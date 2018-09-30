import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from six.moves import range
import copy
from nltk.tokenize import RegexpTokenizer
from PIL import Image

from codebase.utils.config import cfg
from codebase.utils.all_utils import mkdir_p
from codebase.utils.all_utils import build_super_images, build_super_images2
from codebase.utils.all_utils import weights_init, load_params, copy_G_params
from codebase.model import G_DCGAN, G_FC_Net, RNN_Encoder, CNN_Encoder
from codebase.model import D_Net_64, D_Net_128, D_Net_256
from codebase.datasets import prepare_data
from codebase.utils.losses import words_loss, discriminator_loss, generator_loss, KL_loss, discriminator_score
from codebase.utils.prepare import Preparation

# ------------------ Text to image task -------------------------- #

class condGAN(object):

    def __init__(self, output_dir, data_loader, n_words, ixtoword, log):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'model')
            self.image_dir = os.path.join(output_dir, 'image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        self.log = log
        self.update_interval = 100
        self.img_save_interval = 400
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.D_lr = cfg.TRAIN.DISCRIMINATOR_LR
        self.G_lr = cfg.TRAIN.GENERATOR_LR
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

        self.train_log = np.empty((0, 3))

        self.log.add("CUDA status: {}".format(cfg.CUDA))
        self.log.add("GPU ID: {}".format(cfg.GPU_ID))
        self.log.add("Init condGAN ... ")


    def build_models(self, build_discr=True):
        ''' Building models '''

        # Building Text and Image encoders
        if cfg.TRAIN.FLAG and (cfg.TRAIN.NET_E_TEXT == '' or cfg.TRAIN.NET_E_IMG == ''):
            self.log.add('Error: no pretrained text-image encoders')
            return

        # Loading image encoder
        image_encoder = None
        if cfg.TRAIN.FLAG and cfg.TRAIN.NET_E_IMG != '':
            image_encoder = CNN_Encoder(cfg.TEXT.EMBEDDING_DIM)
            state_dict = torch.load(cfg.TRAIN.NET_E_IMG, map_location=lambda storage, loc: storage)
            image_encoder.load_state_dict(state_dict)
            for p in image_encoder.parameters():
                p.requires_grad = False
            self.log.add('Load image encoder from: {}'.format(cfg.TRAIN.NET_E_IMG))
            image_encoder.eval()

        # Loading text encoder
        text_encoder = RNN_Encoder(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM, nlayers=cfg.TEXT.RNN_LAYERS)
        state_dict = torch.load(cfg.TRAIN.NET_E_TEXT, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        self.log.add('Load text encoder from: {}'.format(cfg.TRAIN.NET_E_TEXT))
        text_encoder.eval()

        # Building generators and discriminators
        netsD = []
        if cfg.GAN.B_DCGAN:
            if build_discr:
                if cfg.TREE.BRANCH_NUM ==1:
                    D_Net = D_Net_64
                elif cfg.TREE.BRANCH_NUM == 2:
                    D_Net = D_Net_128
                else:  # cfg.TREE.BRANCH_NUM == 3:
                    D_Net = D_Net_256
                # TODO: elif cfg.TREE.BRANCH_NUM > 3:
                netsD = [D_Net(b_jcu=False)]
            netG = G_DCGAN()
        else:
            if build_discr:
                if cfg.TREE.BRANCH_NUM > 0:
                    netsD.append(D_Net_64())
                if cfg.TREE.BRANCH_NUM > 1:
                    netsD.append(D_Net_128())
                if cfg.TREE.BRANCH_NUM > 2:
                    netsD.append(D_Net_256())
                # TODO: if cfg.TREE.BRANCH_NUM > 3:
            netG = G_FC_Net()

        number_nets = cfg.TREE.BRANCH_NUM
        netG.apply(weights_init)
        self.log.add('G_net: {}'.format(netG), False)

        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
            self.log.add('D_net[{}]: {}'.format(i, netsD[i]), False)
        self.log.add('Number of nets G/D: {}'.format(number_nets))

        # Loading pretrained generators and discriminators
        epoch = 0
        if cfg.TRAIN.NET_G != '':
            state_dict = torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            self.log.add('Load G from: {}'.format(cfg.TRAIN.NET_G))
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            if cfg.TRAIN.B_NET_D:
                for i in range(len(netsD)):
                    if cfg.TRAIN.NET_D[i] != '':
                        Dname = cfg.TRAIN.NET_D[i]
                        self.log.add('Load D{} from: {}'.format(i, Dname))
                        state_dict = torch.load(Dname, map_location=lambda storage, loc: storage)
                        netsD[i].load_state_dict(state_dict)


        # Using CUDA
        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            netG.cuda()
            for i in range(len(netsD)):
                netsD[i].cuda()
        return [text_encoder, image_encoder, netG, netsD, epoch]


    def define_optimizers(self, netG, netsD):
        '''Init optimizers for generator and discriminator'''

        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(),
                             lr=self.D_lr,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        optimizerG = optim.Adam(netG.parameters(),
                                lr=self.G_lr,
                                betas=(0.5, 0.999))

        return optimizerG, optimizersD


    def prepare_labels(self):
        '''Init labels'''
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels


    def set_requires_grad_value(self, models_list, brequires):
        ''''''
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires


    def train(self):
        text_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        gen_iterations = 0
        for epoch in range(start_epoch, self.max_epoch):
            self.epoch_start_time = time.time()
            start_time = time.time()

            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netsD, True)

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
                imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, _, mu, logvar = netG(noise, sent_emb, words_embs, mask)

                #######################################################
                # (3) Update D network
                ######################################################
                errD_total = 0
                D_logs = ''
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                              sent_emb, real_labels, fake_labels)
                    # backward and update parameters
                    errD.backward()
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += 'D%d_loss: %.4f ' % (i, errD.data[0])

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                step += 1
                gen_iterations += 1

                # do not need to compute gradient for Ds
                # self.set_requires_grad_value(netsD, False)
                netG.zero_grad()
                errG_total, G_logs = \
                    generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                                   words_embs, sent_emb, match_labels, cap_lens, class_ids)
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs += 'KL_loss: %.4f ' % kl_loss.data[0]

                # Backward and update parameters
                errG_total.backward()
                optimizerG.step()

                # Updating NetG_avg params
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % self.update_interval == 0:
                    elapsed = time.time() - start_time
                    self.log.add(
                        '| Epoch {:3d} | bt {:3d}/{:3d} | ms/bt {:5.2f} | Time {:5.2f}s '
                            .format(epoch, step, len(self.data_loader), elapsed * 1000. / self.update_interval,
                                    elapsed))
                    self.log.add('| ' + D_logs + '\n| ' + G_logs)
                    start_time = time.time()

            # save images
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            self.save_img_results(netG, fixed_noise, sent_emb,
                                          words_embs, mask, image_encoder,
                                          captions, cap_lens, epoch, name='average')
            load_params(netG, backup_para)
            #
            self.save_img_results(netG, fixed_noise, sent_emb,
                                     words_embs, mask, image_encoder,
                                           captions, cap_lens,
                                           epoch, name='current')

            # Show results of current epoch
            self.log.add('-' * 89)
            self.log.add(
                '| End epoch {:3d}/{:3d} | bt {:3d} | G_loss: {:5.4f} D_loss: {:5.4f} | G_lr: {:5.4f} D_lr: {:5.4f} | Time: {:5.2f}s'.format(
                    epoch, self.max_epoch, self.num_batches, errG_total.data[0], errD_total.data[0],
                    self.G_lr, self.D_lr, time.time() - self.epoch_start_time))
            self.log.add('-' * 89)

            # Save models
            self.save_model(epoch, netG, avg_param_G, netsD, errG_total.data[0], errD_total.data[0])


        self.save_model(self.max_epoch, netG, avg_param_G, netsD, errG_total.data[0], errD_total.data[0])


    def save_model(self, epoch, netG, avg_param_G,  netsD, G_loss, D_loss, force=False):
        '''Saving model'''

        loss_str = np.array([epoch, G_loss, D_loss])
        self.train_log = np.vstack([self.train_log, loss_str])
        mins = np.min(self.train_log, axis=0)
        min_g_loss = mins[1]
        min_d_loss = mins[2]
        self.log.add("| G_loss min: {:5.4f} D_loss min: {:5.4f}".format(min_g_loss, min_d_loss))

        # G_avg_loss <= min_g_avg_loss or
        if ( G_loss <= min_g_loss or epoch % self.snapshot_interval == 0 or force):

            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save(netG.state_dict(),
                       '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
            load_params(netG, backup_para)

            #
            for i in range(len(netsD)):
                netD = netsD[i]
                torch.save(netD.state_dict(),
                           '%s/netD%d_epoch_%d.pth' % (self.model_dir, i, epoch))
            self.log.add('| Saved G/Ds models')
        self.log.add('-' * 89) 


    def save_img_results(self, netG, noise, sent_emb, words_embs, mask, image_encoder, captions, cap_lens,
                         gen_iterations, name='current'):
        '''Save generator results'''

        self.log.add("| Saving {} result images ... ".format(name))

        fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = build_super_images(img, captions, self.ixtoword,
                                            attn_maps, att_sze, lr_imgs=lr_img)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png' % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, cap_lens,
                                    None, self.batch_size)
        img_set, _ = build_super_images(fake_imgs[i].detach().cpu(),
                                        captions, self.ixtoword, att_maps, att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png' % (self.image_dir, name, gen_iterations)
            im.save(fullpath)


    def sampling(self, split_dir):
        '''Generate images from training set text annotations'''

        if cfg.TRAIN.NET_G == '':
            self.log.add('Error: the path for morels is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'

            text_encoder, image_encoder, netG, netsD, _ = self.build_models(build_discr=False)


            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            if cfg.CUDA:
                noise = noise.cuda()

            model_dir = cfg.TRAIN.NET_G
            state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            netG.load_state_dict(state_dict)
            self.log.add('Load G from: ', model_dir)

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s/%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)

            cnt = 0

            for _ in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
                for step, data in enumerate(self.data_loader, 0):
                    cnt += batch_size
                    if step % 100 == 0:
                        self.log.add('step: ', step)
                    # if step > 50:
                    #     break

                    imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                    mask = (captions == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)
                    for j in range(batch_size):
                        s_tmp = '%s/single/%s' % (save_dir, keys[j])
                        folder = s_tmp[:s_tmp.rfind('/')]
                        if not os.path.isdir(folder):
                            self.log.add('Make a new folder: ', folder)
                            mkdir_p(folder)
                        k = -1
                        # for k in range(len(fake_imgs)):
                        im = fake_imgs[k][j].data.cpu().numpy()
                        # [-1, 1] --> [0, 255]
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)
                        fullpath = '%s_s%d.png' % (s_tmp, k)
                        im.save(fullpath)


    def generate_single_image(self, images, filenames, save_dir, split_dir, sentenceID=0):
        '''Generate single image'''

        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' % \
                    (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                self.log.add('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
            # range from [-1, 1] to [0, 1]
            # img = (images[i] + 1.0) / 2
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)


    def prepare_dict_from_files(self, wordtoix, tries=1, threshold=0.05):
        '''generate images from example sentences'''

        filepath = '%s/example_filenames.txt' % (cfg.DATA_DIR)
        data_dic = {}
        with open(filepath, "r") as f:
            filenames = f.read().split('\n')
            for name in filenames:
                if len(name) == 0:
                    continue
                filepath = '%s/%s.txt' % (cfg.DATA_DIR, name)
                with open(filepath, "r") as f:
                    self.log.add('Load from:', name)
                    sentences = f.read().split('\n')
                    captions, cap_lens = self.tokenize_sentences(sentences, wordtoix)

                cap_array, cap_lens, sorted_indices = self.process_captions(captions, cap_lens)

                key = name[(name.rfind('/') + 1):]
                data_dic[key] = [cap_array, cap_lens, sorted_indices]

        return self.generate_images(data_dic, tries, threshold)


    def tokenize_sentences(self, sentences, wordtoix):
        '''Making a list of tokens for each sentence'''

        captions = []
        cap_lens = []
        for sent in sentences:
            if len(sent) == 0:
                continue
            sent = sent.replace("\ufffd\ufffd", " ")
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(sent.lower())
            if len(tokens) == 0:
                print('sent', sent)
                continue

            rev = []
            for t in tokens:
                t = t.encode('ascii', 'ignore').decode('ascii')
                if len(t) > 0 and t in wordtoix:
                    rev.append(wordtoix[t])
            captions.append(rev)
            cap_lens.append(len(rev))

        return captions, cap_lens


    def process_captions(self, captions, cap_lens, sort=True):
        '''Processing captions'''

        sorted_indices = np.argsort(cap_lens)[::-1]
        max_len = np.max(cap_lens)
        cap_lens = np.asarray(cap_lens)
        cap_lens = cap_lens[sorted_indices]
        cap_array = np.zeros((len(captions), max_len), dtype='int64')
        for i in range(len(captions)):
            if sort:
                idx = sorted_indices[i]
            else:
                idx = i
            cap = captions[idx]
            c_len = len(cap)
            cap_array[i, :c_len] = cap

        return  cap_array, cap_lens, sorted_indices


    def generate_images(self, data_dic, tries, threshold):
        '''Generate examples'''

        if cfg.TRAIN.NET_G == '':
            self.log.add('Error: model not found!')
        else:

            s_tmp = cfg.SAVE_DIR
            text_encoder, image_encoder, netG, netsD, _ = self.build_models()
            netG.eval()
            netsD[2].eval()

            generated_images = []

            with torch.no_grad():
                for key in data_dic:

                    save_dir = '%s/%s' % (s_tmp, key)
                    mkdir_p(save_dir)
                    captions, cap_lens, sorted_indices = data_dic[key]


                    batch_size = captions.shape[0]
                    nz = cfg.GAN.Z_DIM
                    captions = Variable(torch.from_numpy(captions))
                    cap_lens = Variable(torch.from_numpy(cap_lens))

                    if cfg.CUDA:
                        captions = captions.cuda()
                        cap_lens = cap_lens.cuda()

                    for i in range(1):  # 16
                        noise = Variable(torch.FloatTensor(batch_size, nz))
                        if cfg.CUDA:
                            noise = noise.cuda()

                        image_name = '%s/%d_s' % (save_dir, i)
                        images = self.generate_image(text_encoder, netG, netsD, batch_size, captions, cap_lens, sorted_indices, noise,
                                            image_name, tries, threshold)
                        generated_images.append(images)

            return generated_images


    def generate_image(self, text_encoder, netG, netsD, batch_size, captions, cap_lens, sorted_indices, noise, image_name, tries, threshold):
        '''Generate single image'''

        images = {0: [], 1: [], 2: []}

        #######################################################
        # (1) Extract text embeddings
        ######################################################

        hidden = text_encoder.init_hidden(batch_size)
        # words_embs: batch_size x nef x seq_len
        # sent_emb: batch_size x nef

        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
        mask = (captions == 0)
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))

        #######################################################
        # (2) Generate fake images
        ######################################################

        prep = Preparation()

        for tr in range(tries):
            images[0].append([])
            images[1].append([])
            images[2].append([])

            generating = True
            iter = 0
            while generating:
                prep.set_random_seed()
                noise.data.normal_(0, 1)
                fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
                score = discriminator_score(netsD[2], fake_imgs[2], sent_emb, fake_labels)

                iter += 1
                if (score > threshold or iter >= 10):
                    generating = False


            # G attention
            cap_lens_np = cap_lens.cpu().data.numpy()
            for j in range(batch_size):
                save_name = image_name + '_{}'.format(sorted_indices[j])

                for k in range(len(fake_imgs)):
                    im = fake_imgs[k][j].data.cpu().numpy()
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    # print('im', im.shape)
                    im = np.transpose(im, (1, 2, 0))
                    # print('im', im.shape)
                    im = Image.fromarray(im)
                    fullpath = '%s_g%d_%d.png' % (save_name, k, tr)
                    im.save(fullpath)
                    images[k][tr].append(fullpath)

                for k in range(len(attention_maps)):
                    if len(fake_imgs) > 1:
                        im = fake_imgs[k + 1].detach().cpu()
                    else:
                        im = fake_imgs[0].detach().cpu()
                    attn_maps = attention_maps[k]
                    att_sze = attn_maps.size(2)
                    img_set, sentences = build_super_images2(im[j].unsqueeze(0),
                                            captions[j].unsqueeze(0),
                                            [cap_lens_np[j]], self.ixtoword,
                                            [attn_maps[j]], att_sze)
                    if img_set is not None:
                        im = Image.fromarray(img_set)
                        fullpath = '%s_a%d_%d.png' % (save_name, k, tr)
                        im.save(fullpath)

        return images
