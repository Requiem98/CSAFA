import warnings

with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore', message = "IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html")
    
    from pathlib import Path
    import sys
    from tqdm import tqdm
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image
    import torch
    from torch import nn
    from torch.nn.parameter import Parameter
    from torch.utils.data import Dataset, DataLoader, Sampler, default_collate
    import torchvision.transforms as T
    from torchvision.io import read_image, write_png
    from torchvision import transforms
    from torchvision.transforms import functional as F
    from torchvision.utils import make_grid
    import torch.nn.functional as f
    from torch.distributions.uniform import Uniform
    import pickle
    import glob
    import xml.etree.ElementTree as ET
    import os
    from imageio import imread, imsave
    from collections import defaultdict
    import time
    from sklearn.model_selection import train_test_split
    import math
    import lightning
    import lightning.pytorch as pl
    from tensorboard import program
    from pytorch_lightning.loggers import TensorBoardLogger
    from lightning.pytorch.cli import LightningCLI
    import lightning.pytorch.callbacks as clb
    import functools
    from functools import partial
    import cv2
    import shutil
    import torchmetrics
    from collections import OrderedDict
    from torchvision.models import vgg16
    from torchvision.models import VGG16_Weights
    import timm
    from timm.models.vision_transformer import VisionTransformer as ViT
    from timm.models.vision_transformer import vit_base_patch16_224
    #losses
    from torch.nn.functional import mse_loss as mse
    from torch.nn.functional import l1_loss as mae
    from torch.nn.functional import binary_cross_entropy_with_logits as bce_l
    from jsonargparse import lazy_instance

local_home = Path(__file__).parent.resolve()
os.chdir(local_home)
torch.hub.set_dir(f"{local_home}/cache")

CKP_DIR = "./output/lightning_logs/"

if not os.path.exists(CKP_DIR):
    os.makedirs(CKP_DIR)







