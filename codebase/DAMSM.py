import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image

from codebase.datasets import prepare_data
from codebase.utils.all_utils import mkdir_p
from codebase.utils.config import cfg
from codebase.model import RNN_Encoder, CNN_Encoder
from codebase.utils.losses import sent_loss, words_loss
from codebase.utils.all_utils import build_super_images, build_super_images2


# ------------------ DAMSM Net ----------------- #

class DAMSM(object):

    def __init__(self, output_dir, data_loader, data_loader_val,  n_words, ixtoword, log):

        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'model')
            self.image_dir = os.path.join(output_dir, 'image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        if cfg.GPU_ID >= 0:
            torch.cuda.set_device(cfg.GPU_ID)
            cudnn.benchmark = True

        self.log = log
        self.update_interval = 200
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        #self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.data_loader_val = data_loader_val
        self.text_encoder, self.image_encoder, self.labels, self.start_epoch = self.build_models()

        self.train_log = np.empty((0,5))

        self.log.add("CUDA status: {}".format(cfg.CUDA))
        self.log.add("GPU ID: {}".format(cfg.GPU_ID))
        self.log.add("Init DAMSM ... ")


    def build_models(self):
        '''Build models'''

        text_encoder = RNN_Encoder(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM, nlayers=cfg.TEXT.RNN_LAYERS)
        image_encoder = CNN_Encoder(cfg.TEXT.EMBEDDING_DIM)
        labels = Variable(torch.LongTensor(range(self.batch_size)))
        start_epoch = 0

        if cfg.TRAIN.NET_E_TEXT != '' and cfg.TRAIN.NET_E_IMG != '':
            state_dict = torch.load(cfg.TRAIN.NET_E_TEXT)
            text_encoder.load_state_dict(state_dict)
            self.log.add('Load text encoder: {}'.format(cfg.TRAIN.NET_E_TEXT))

            state_dict = torch.load(cfg.TRAIN.NET_E_IMG)
            image_encoder.load_state_dict(state_dict)
            self.log.add('Load image encoder: {}'.format(cfg.TRAIN.NET_E_IMG))

            istart = cfg.TRAIN.NET_E_TEXT.rfind('_') + 1
            iend = cfg.TRAIN.NET_E_TEXT.rfind('.')
            start_epoch = cfg.TRAIN.NET_E_TEXT[istart:iend]
            start_epoch = int(start_epoch) + 1
            self.log.add('Start_epoch: {}'.format(start_epoch))

        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            labels = labels.cuda()

        return text_encoder, image_encoder, labels, start_epoch



    def train(self):
        '''Training models'''

        para = list(self.text_encoder.parameters())
        for v in self.image_encoder.parameters():
            if v.requires_grad:
                para.append(v)

        try:
            lr = cfg.TRAIN.ENCODER_LR
            for epoch in range(self.start_epoch, cfg.TRAIN.MAX_EPOCH):
                self.optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
                self.epoch_start_time = time.time()
                w_loss_train = 0.
                s_loss_train = 0.

                _, s_loss_step, w_loss_step = self.train_step(epoch)
                w_loss_train += w_loss_step
                s_loss_train += s_loss_step

                self.log.add('-' * 89)
                if len(self.data_loader_val) > 0:
                    s_loss_val, w_loss_val = self.evaluate()
                    self.log.add('| End epoch {:3d}/{:3d} | train loss {:5.2f} {:5.2f} | valid loss '
                                 '{:5.2f} {:5.2f} | lr {:.5f}| Time {:5.2f}s'
                                 .format(epoch, self.max_epoch, s_loss_train, w_loss_train, s_loss_val, w_loss_val, lr,
                                         time.time() - self.epoch_start_time))
                    self.log.add('-' * 89)


                if lr > cfg.TRAIN.ENCODER_LR / 10.:
                    lr *= 0.98

                # Saving models
                self.save_models(epoch, s_loss_train, s_loss_val, w_loss_train, w_loss_val)

        except KeyboardInterrupt:
            self.log.add('-' * 89)
            self.log.add('Exiting from training early')


    '''
    def train_1(self):
        

        para = list(self.text_encoder.parameters())
        for v in self.image_encoder.parameters():
            if v.requires_grad:
                para.append(v)

        self.log.add("Start training ... ")
        self.log.add("Training params: {}".format(para))
        try:
            lr = cfg.TRAIN.ENCODER_LR
            self.log.add('-' * 89)

            for epoch in range(self.start_epoch, cfg.TRAIN.MAX_EPOCH):
                self.optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
                self.epoch_start_time = time.time()
                w_loss_train = 0.
                s_loss_train = 0.

                _, s_loss_step, w_loss_step = self.train_step(epoch)
                w_loss_train += w_loss_step
                s_loss_train += s_loss_step

                self.log.add('-' * 89)
                s_loss_val, w_loss_val = 0., 0.
                if len(self.data_loader_val) > 0:
                    s_loss_val, w_loss_val = self.evaluate()
                    self.log.add('| End epoch {:3d}/{:3d} | train loss {:5.2f} {:5.2f} | valid loss '
                          '{:5.2f} {:5.2f} | lr {:.5f}| Time {:5.2f}s'
                          .format(epoch, self.max_epoch, s_loss_train, w_loss_train, s_loss_val, w_loss_val, lr, time.time() - self.epoch_start_time))
                    self.log.add('-' * 89)

                # LR decay
                if lr > cfg.TRAIN.ENCODER_LR / 10.:
                    lr *= 0.98

                # Saving models
                self.save_models(epoch, s_loss_train, s_loss_val, w_loss_train, w_loss_val)

            self.log.add('Train_log: [Epoch, S_loss, W_loss] \n{}'.format(self.train_log))

        except KeyboardInterrupt:
            self.log.add('-' * 89)
            self.log.add('Exiting from training early')
    '''


    def train_step(self, epoch):

        self.image_encoder.train()
        self.text_encoder.train()
        s_total_loss0 = 0
        s_total_loss1 = 0
        s_loss_step = 0.
        w_total_loss0 = 0
        w_total_loss1 = 0
        w_loss_step = 0.

        batch_num = len(self.data_loader)
        count = (epoch + 1) * batch_num

        start_time = time.time()
        for step, data in enumerate(self.data_loader, 0):

            # self.save_raw_data(step, data)  # DEBUG

            self.text_encoder.zero_grad()
            self.image_encoder.zero_grad()

            imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

            # words_features: batch_size x nef x 17 x 17
            # sent_code: batch_size x nef
            words_features, sent_code = self.image_encoder(imgs[-1])
            # --> batch_size x nef x 17*17
            nef, att_sze = words_features.size(1), words_features.size(2)
            # words_features = words_features.view(batch_size, nef, -1)

            hidden = self.text_encoder.init_hidden(self.batch_size)
            # words_emb: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_emb, sent_emb = self.text_encoder(captions, cap_lens, hidden)

            w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, self.labels,
                                                     cap_lens, class_ids, self.batch_size)
            w_total_loss0 += w_loss0.data
            w_total_loss1 += w_loss1.data
            loss = w_loss0 + w_loss1

            s_loss0, s_loss1 = \
                sent_loss(sent_code, sent_emb, self.labels, class_ids, self.batch_size)
            loss += s_loss0 + s_loss1
            s_total_loss0 += s_loss0.data
            s_total_loss1 += s_loss1.data
            #
            w_loss_step += w_loss0.data + w_loss1.data
            s_loss_step += s_loss0.data + s_loss1.data
            loss.backward()
            #
            # `clip_grad_norm` helps prevent
            # the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(self.text_encoder.parameters(), cfg.TRAIN.RNN_GRAD_CLIP)
            self.optimizer.step()

            if step % self.update_interval == 0:
                count = epoch * batch_num + step

                s_cur_loss0 = s_total_loss0[0] / self.update_interval
                s_cur_loss1 = s_total_loss1[0] / self.update_interval

                w_cur_loss0 = w_total_loss0[0] / self.update_interval
                w_cur_loss1 = w_total_loss1[0] / self.update_interval

                elapsed = time.time() - start_time
                self.log.add('| Epoch {:3d} | bt {:3d}/{:3d} | ms/bt {:5.2f} | S_loss {:2.4f} {:2.4f} | W_loss {:2.4f} {:5.4f} | Time {:5.2f}s'
                      .format(epoch,  step, len(self.data_loader),
                              elapsed * 1000. / self.update_interval,
                              s_cur_loss0, s_cur_loss1, w_cur_loss0, w_cur_loss1, elapsed))

                s_total_loss0 = 0
                s_total_loss1 = 0
                w_total_loss0 = 0
                w_total_loss1 = 0
                start_time = time.time()

                # Attention Maps
                img_set, _ = build_super_images(imgs[-1].cpu(), captions,
                                       self.ixtoword, attn_maps, att_sze)
                if img_set is not None:
                    im = Image.fromarray(img_set)
                    fullpath = '%s/attention_maps_%d_%d.png' % (self.image_dir, epoch, step)
                    im.save(fullpath)

        s_loss_step /= batch_num
        w_loss_step /= batch_num

        return count, s_loss_step, w_loss_step


    def save_models(self, epoch, s_loss_train, s_loss_val, w_loss_train, w_loss_val):
        loss_str = np.array([epoch, s_loss_train, s_loss_val, w_loss_train, w_loss_val])
        self.train_log = np.vstack([self.train_log, loss_str])
        mins = np.min(self.train_log, axis=0)
        min_s_loss = mins[2]
        min_w_loss = mins[4]
        self.log.add ("| S_loss min: {:6.4f} W_loss min: {:5.4f}".format(min_s_loss, min_w_loss))

        if (s_loss_val <= min_s_loss and w_loss_val <= min_w_loss):
            torch.save(self.image_encoder.state_dict(),
                       '%s/image_encoder_%d.pth' % (self.model_dir, epoch))
            torch.save(self.text_encoder.state_dict(),
                       '%s/text_encoder_%d.pth' % (self.model_dir, epoch))
            self.log.add('| Save encoder models')
            self.log.add('-' * 89)


    def save_raw_data(self, step, data):
        '''Saving raw images - for debugging'''
        real_imgs = data[0][0].cpu()
        real_imgs.add_(1).div_(2).mul_(255)
        real_imgs = real_imgs.data.numpy().astype(np.uint8)
        real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))

        im = Image.fromarray(real_imgs[0])
        fullpath = '%s/data_%d.png' % (self.image_dir, step)
        im.save(fullpath)


    def evaluate(self):
        '''Model evaluation'''

        self.log.add("| Calculating validation error")
        self.image_encoder.eval()
        self.text_encoder.eval()
        s_total_loss = 0
        w_total_loss = 0
        for step, data in enumerate(self.data_loader_val, 0):
            real_imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

            words_features, sent_code = self.image_encoder(real_imgs[-1])
            # nef = words_features.size(1)
            # words_features = words_features.view(batch_size, nef, -1)

            hidden = self.text_encoder.init_hidden(self.batch_size)
            words_emb, sent_emb = self.text_encoder(captions, cap_lens, hidden)

            w_loss0, w_loss1, attn = words_loss(words_features, words_emb, self.labels,
                                                cap_lens, class_ids, self.batch_size)
            w_total_loss += (w_loss0 + w_loss1).data

            s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, self.labels, class_ids, self.batch_size)
            s_total_loss += (s_loss0 + s_loss1).data

            if step == 50:
                break

        s_cur_loss = s_total_loss.item()/ step
        w_cur_loss = w_total_loss.item() / step

        return s_cur_loss, w_cur_loss

