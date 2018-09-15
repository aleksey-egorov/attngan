import os
import sys
import time

from codebase.utils.log import Log
from codebase.utils.prepare import Preparation
from codebase.datasets import TextDataset, DataPreprocess
from codebase.DAMSM import DAMSM

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

if __name__ == "__main__":

    # Preparing environment
    prep = Preparation()
    cfg = prep.parse_arguments('cfg/damsm_bird.yml')
    prep.set_random_seed()
    output_dir = prep.set_output_dir()
    prep.set_cuda()

    # Init log
    log = Log(output_dir)
    log.add('Using config: {} \n'.format(cfg), False)

    # Get data loader
    dataprep = DataPreprocess(log)
    image_transform = dataprep.image_transform()
    dataloader, dataset = dataprep.get_dataloader('train', image_transform, True)
    dataloader_val, dataset_val = dataprep.get_dataloader('test', image_transform, True)

    # Train model
    model = DAMSM(output_dir, dataloader, dataloader_val, dataset.n_words, dataset.ixtoword, log)
    start_t = time.time()
    model.train()
    end_t = time.time()
    log.add('Total time for training: {}'.format(end_t - start_t))

