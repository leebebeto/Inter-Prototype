from learner import Face_learner
import argparse
import torch
import numpy as np
import random
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inter-Prototype (BMVC 2021)')

    parser.add_argument("--seed", help='seed', default=4885, type=int)
    parser.add_argument("--wandb", help='whether to use wandb', action='store_true')
    parser.add_argument("--tensorboard", help='whether to use tensorboard', action='store_true')

    # model
    parser.add_argument("--net_mode", help="which network, [ir, ir_se, mobilefacenet]",default='ir_se', type=str)
    parser.add_argument("--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument("--embedding_size", help='embedding_size', default=512, type=int)
    parser.add_argument("--drop_ratio", help="ratio of drop out", default=0.6, type=float)
    parser.add_argument("--device", help="cuda or cpu", default='cuda', type=str)
    parser.add_argument("--loss", help="Arcface", default='Arcface', type=str)
    parser.add_argument("--model_dir", help="path for loading a pretrained model",default='', type=str)

    args = parser.parse_args()
    args.home = os.path.expanduser('~')
    args.train = False

    # fix random seeds
    random_seed = args.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    print('Official pytorch code of "Improving Face Recognition with Large Age Gaps by Learning to Distinguish Children (BMVC 2021)"')
    # init learner
    learner = Face_learner(args)
    print('evaluation starts....')

    learner.model.eval()
    learner.evaluate()
