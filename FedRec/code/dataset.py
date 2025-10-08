import sys
import copy
import torch
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from multiprocessing import Process, Queue
from torch.utils.data import Dataset
from scipy import sparse


def build_index(dataset_path):
    ui_mat = np.loadtxt(dataset_path, dtype=np.int32)
    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()
    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]
    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])
    return u2i_index, i2u_index


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def get_itemnum(dataset_path):
    itemnum = 0
    with open(dataset_path, 'r') as f:
        for line in f:
            _, i = line.rstrip().split(' ')
            i = int(i)
            itemnum = max(i, itemnum)
    return itemnum


def data_partition(dataset_path):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    with open(dataset_path, 'r') as f:
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


class ClientsDataset(Dataset):
    def __init__(self, data_path, maxlen):
        self.maxlen = maxlen
        self.user_train, self.user_valid, self.user_test, self.usernum, self.itemnum = data_partition(data_path)
        self.user_set = list(self.user_train.keys())
        self.item_set = np.arange(1, self.itemnum + 1)
        self.item_maxid = self.itemnum
        cc = 0.0
        for u in self.user_train:
            cc += len(self.user_train[u])
        print('average sequence length: %.2f' % (cc / len(self.user_train)))
        self.seq = {}
        self.seq_len = {}
        self.train_seq = {}
        self.valid_seq = {}
        for u in self.user_set:
            items = np.array(self.user_train[u])
            if len(items) > maxlen + 1:
                items = items[-(maxlen + 1):]
            pad_len = max(0, maxlen + 1 - len(items))
            train_s = np.pad(items[:-1], (pad_len, 0))
            valid_s = np.pad(items[1:], (pad_len, 0))
            self.seq_len[u] = len(items) - 1
            self.seq[u] = items
            self.valid_seq[u] = valid_s
            self.train_seq[u] = train_s
        self.item_prob = np.ones([self.item_maxid + 1], dtype=np.float32)

    def __len__(self):
        return len(self.user_set)

    def __getitem__(self, user_id):
        input_seq = self.train_seq[user_id]
        target_seq = self.valid_seq[user_id]
        input_len = self.seq_len[user_id]
        return input_seq, target_seq, input_len

    def get_maxid(self):
        return self.itemnum

    def get_user_set(self):
        return self.user_set

    def get_usernum(self):
        return self.usernum

    def get_itemnum(self):
        return self.itemnum

    def get_dataset(self):
        return [self.user_train, self.user_valid, self.user_test, self.usernum, self.itemnum]


def evaluate(model, dataset, maxlen, neg_num, k, full_eval=False, device='cuda'):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    NDCG, HT, valid_user = 0.0, 0.0, 0
    users = range(1, usernum + 1) if usernum <= 10000 else random.sample(range(1, usernum + 1), 10000)
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1:
            continue
        seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        rated = set(train[u]) | set(valid[u])
        rated.add(0)
        if full_eval:
            item_idx = [test[u][0]] + [i for i in range(1, itemnum + 1) if i not in rated]
        else:
            item_idx = [test[u][0]]
            for _ in range(neg_num):
                t = np.random.randint(1, itemnum + 1)
                while t in rated:
                    t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])[0]
        rank = predictions.argsort().argsort()[0].item()
        valid_user += 1
        if rank < k:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, maxlen, neg_num, k, full_eval=False, device='cuda'):
    [train, valid, _, usernum, itemnum] = copy.deepcopy(dataset)
    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1:
            continue
        seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        rated = set(train[u])
        rated.add(0)
        if full_eval:
            item_idx = [valid[u][0]] + [i for i in range(1, itemnum + 1) if i not in rated]
        else:
            item_idx = [valid[u][0]]
            for _ in range(neg_num):
                t = np.random.randint(1, itemnum + 1)
                while t in rated:
                    t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]
        rank = predictions.argsort().argsort()[0].item()
        valid_user += 1
        if rank < k:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    return NDCG / valid_user, HT / valid_user
