import os
import sys
import time

from codebase.utils.log import Log
from codebase.datasets import TextDataset, DataPreprocess
from codebase.utils.prepare import Preparation
from codebase.condGAN import condGAN

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

if __name__ == "__main__":

    # Preparing environment
    prep = Preparation()
    cfg = prep.parse_arguments('cfg/eval_bird.yml')
    prep.set_random_seed()
    output_dir = prep.set_output_dir()
    prep.set_cuda()

    # Init log
    log = Log(output_dir)
    log.add('Using config: {} \n'.format(cfg), False)

    # Get data loader
    dataprep = DataPreprocess(log)
    image_transform = dataprep.image_transform()
    dataloader, dataset = dataprep.get_dataloader('test', image_transform, True)

    # Define models and go to evaluate
    model = condGAN(output_dir, dataloader, dataset.n_words, dataset.ixtoword, log)
    start_t = time.time()

    # Generate images from pre-extracted embeddings
    if cfg.B_VALIDATION:
        model.sampling('test')  # generate images for the whole valid dataset
    else:
        model.prepare_dict_from_files(dataset.wordtoix)  # generate images for customized captions
    end_t = time.time()
    log.add('Total time for evaluation: {}'.format(end_t - start_t))
