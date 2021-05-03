import json
import os
from torch.utils.data import Dataset
from .IndexedDataset import make_dataset,MMapIndexedDataset
import random
import numpy as np

class MultiDocDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.max_len = config.getint('train', 'max_len')
        path = config.get('data', '%s_data' % mode)
        flist = config.get('data', '%s_files' % mode).split(',')
        self.datasets = [MMapIndexedDataset(os.path.join(path, f), False) for f in flist]
        self.lens = [len(d) for d in self.datasets]
        self.length = sum(self.lens)
        self.idlist = np.arange(0, self.length)
        np.random.shuffle(self.idlist)
    
    def get_index_i(self, idx):
        ridx = int(self.idlist[idx])
        sent = None
        for i in range(len(self.lens)):
            if ridx >= self.lens[i]:
                ridx -= self.lens[i]
            else:
                sent = self.datasets[i][ridx]
        if sent is None:
            raise ValueError('Index is larger than the number of data')
        for i in range(max(sent.shape[0] - 52, 0), sent.shape[0]):
            if sent[i] == 102:
                break
        return sent[:i+1]
    
    def __getitem__(self, idx):
        sent = self.get_index_i(idx)
        while sent.shape[0] < self.max_len - 50:
            ridx = random.randint(0, self.length - 1)
            rsent = self.get_index_i(ridx)
            sent = np.concatenate([sent, rsent])[:self.max_len]
        return sent
    
    def __len__(self):
        return self.length