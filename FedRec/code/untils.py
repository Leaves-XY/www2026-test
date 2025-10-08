import re
import os
import gzip
import datetime
import numpy as np
import pandas as pd
from model.FMLP import FMLPRec
from model.SASRec import SASRec
from model.BSARec import BSARec
from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
def add_noise(weights, lambd):
    with torch.no_grad():
        for k in weights.keys():
            noise = torch.distributions.laplace.Laplace(0, lambd).sample(weights[k].shape)
            weights[k] += noise.to(weights[k].device)
    return weights
def ensure_dir ( path ):
    if not os.path.exists (path):
        os.makedirs (path)
def get_local_time ( ):
    cur = datetime.datetime.now ()
    cur = cur.strftime ('%b-%d-%Y_%H-%M-%S')
    return cur
def check_data_dir ( root_path, data_filename ):
    data_dir = root_path
    ensure_dir (data_dir)
    data_dir = os.path.join (data_dir, data_filename)
    assert os.path.exists (data_dir), f'{data_filename} in {root_path} not exists'
    return data_dir
def parse ( path ):
    g = gzip.open (path, 'rb')
    for l in g:
        yield eval (l)
class Logger:
    def __init__ ( self, config, desc = None ):
        log_root = config ['log_path']
        log_root = os.path.join (log_root, config ['model'])
        log_root = os.path.join (log_root, config ['dataset'])
        ensure_dir (log_root)
        current_time = datetime.datetime.now().strftime('%m%d-%H%M')
        logfilename = f'{current_time}_{config ["model"]}_{config ["algorithm"]}_{config ["dataset"]}_alpha{config["decor_alpha"]}_ratio{config["kd_ratio"]}_topk{config["top_k_ratio"]}.log'
        logfilepath = os.path.join (log_root, logfilename)
        self.filename = logfilepath
        f = open (logfilepath, 'w', encoding = 'utf-8')
        f.write (str (config) + '
')
        f.flush ()
        f.close ()
    def info ( self, s = None ):
        print (s)
        f = open (self.filename, 'a', encoding = 'utf-8')
        f.write (f'[{get_local_time ()}] - {s}
')
        f.flush ()
        f.close ()
def getModel(config, item_maxid):
    if config['model']=='SASRec':
        return SASRec(config, item_maxid)
    elif config['model']=='FMLP':
        return FMLPRec(config, item_maxid)
    elif config['model']=='BSARec':
        return BSARec(config, item_maxid)
    elif config['model']=='BERT4Rec':
        from model.BERT4Rec import BERT
        return BERT(config, item_maxid)
    else:
        raise ValueError(f"Unknown model: {config['model']}")
class FedDecorrLoss(nn.Module):
    def __init__(self):
        super(FedDecorrLoss, self).__init__()
        self.eps = 1e-8
    def _off_diagonal(self, mat):
        n, m = mat.shape
        assert n == m
        return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    def forward(self, x):
        N, C = x.shape
        if N == 1:
            return 0.0
        x = x - x.mean(dim=0, keepdim=True)
        x = x / torch.sqrt(self.eps + x.var(dim=0, keepdim=True))
        corr_mat = torch.matmul(x.t(), x)
        loss = (self._off_diagonal(corr_mat).pow(2)).mean()
        loss = loss / N
        return loss