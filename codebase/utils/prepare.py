import argparse
import random
import pprint
import datetime
import dateutil.tz
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from codebase.utils.config import cfg, cfg_from_file


class Preparation():

    def parse_arguments(self, default_cfg):
        args = self.get_args(default_cfg)
        if args.cfg_file is not None:
            cfg_from_file(args.cfg_file)

        if args.gpu_id == -1:
            cfg.CUDA = False
        else:
            cfg.GPU_ID = args.gpu_id

        if args.data_dir != '':
            cfg.DATA_DIR = args.data_dir
        cfg.MANUAL_SEED = args.manualSeed

        pprint.pprint(cfg)
        return cfg

    def set_config(self, filename):
        cfg_from_file(filename)
        return cfg

    def show_config(self):
        pprint.pprint(cfg)

    def get_args(self, default_cfg):
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--cfg', dest='cfg_file',
                            help='optional config file',
                            default=default_cfg, type=str)
        parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
        parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
        parser.add_argument('--manualSeed', type=int, help='manual seed')
        args = parser.parse_args()
        return args

    def set_random_seed(self):
        # Setting random seed
        manual_seed = cfg.MANUAL_SEED
        # if not cfg.TRAIN.FLAG:
        #     manual_seed = 100
        # el
        if cfg.MANUAL_SEED is None:
            manual_seed = random.randint(1, 10000)
        random.seed(manual_seed)
        np.random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        if cfg.CUDA:
            torch.cuda.manual_seed_all(manual_seed)

    def set_output_dir(self):
        # Setting output
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        output_dir = 'output/%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
        return output_dir

    def set_cuda(self):
        if cfg.GPU_ID >= 0:
            torch.cuda.set_device(cfg.GPU_ID)
            cudnn.benchmark = True