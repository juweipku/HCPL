import math
import os
import random
import shutil
import numpy as np
import torch
import sys
sys.path.append('../..')


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

