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
    """
    构建用户-物品索引，基于pmixer的build_index函数

    参数:
    - dataset_path: 数据文件路径

    返回:
    - u2i_index: 用户到物品的索引 {user_id: [item_ids]}
    - i2u_index: 物品到用户的索引 {item_id: [user_ids]}
    """
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
    """
    生成不在集合s中的随机数，基于pmixer的random_neq函数

    参数:
    - l: 下界
    - r: 上界
    - s: 排除的集合

    返回:
    - t: 不在集合s中的随机数
    """
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def get_itemnum(dataset_path):
    """
    只获取数据集中的物品数量
    """
    itemnum = 0
    with open(dataset_path, 'r') as f:
        for line in f:
            _, i = line.rstrip().split(' ')
            i = int(i)
            itemnum = max(i, itemnum)
    return itemnum


def data_partition(dataset_path):
    """
    数据分割函数，基于pmixer的data_partition函数
    将用户序列分为训练集、验证集和测试集

    参数:
    - dataset_path: 数据文件路径

    返回:
    - [user_train, user_valid, user_test, usernum, itemnum]
    """
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}

    # 读取数据文件
    with open(dataset_path, 'r') as f:
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

    # 分割数据
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


class ClientsDataset (Dataset):
    """
    为联邦学习中的客户端（用户）准备训练数据的 PyTorch Dataset 类。
    实现真正的联邦学习数据隔离，每个客户端只能访问自己的数据
    """

    def __init__(self, data_path,  maxlen):
        """
        初始化数据集。

        参数:
        - data_path: 训练数据文件的路径 (txt格式，每行为 "user_id item_id")
        - maxlen: 序列最大长度
        """
        self.maxlen = maxlen

        # 使用data_partition函数处理数据
        self.user_train, self.user_valid, self.user_test, self.usernum, self.itemnum = data_partition(data_path)

        # 获取所有用户ID和物品ID
        self.user_set = list(self.user_train.keys())
        self.item_set = np.arange(1, self.itemnum + 1)  # 物品ID从1开始
        self.item_maxid = self.itemnum

        # 计算平均序列长度
        cc = 0.0
        for u in self.user_train:
            cc += len(self.user_train[u])
        print('average sequence length: %.2f' % (cc / len(self.user_train)))

        # 预处理每个用户的序列数据
        self.seq = {}  # 每个用户的完整交互序列
        self.seq_len = {}  # 每个用户序列的实际长度
        self.train_seq = {}  # 输入序列 (前n-1个物品)
        self.valid_seq = {}  # 目标序列 (后n-1个物品)

        # 遍历所有用户，处理并存储他们的序列数据
        for u in self.user_set:
            items = np.array(self.user_train[u])
            # train_s 是输入序列，即 item[0] 到 item[n-2]
            # valid_s 是目标序列，即 item[1] 到 item[n-1]

            # 处理序列长度，确保不超过maxlen
            if len(items) > maxlen + 1:
                # 如果序列太长，截取最后maxlen+1个元素
                items = items[-(maxlen + 1):]

            # 计算需要填充的长度
            pad_len = max(0, maxlen + 1 - len(items))

            # 前填充：在序列前面添加0
            train_s = np.pad(items[:-1], (pad_len, 0))
            valid_s = np.pad(items[1:], (pad_len, 0))

            self.seq_len[u] = len(items) - 1  # 实际序列长度为 n-1
            self.seq[u] = items
            self.valid_seq[u] = valid_s
            self.train_seq[u] = train_s

        # 初始化物品的采样概率（用于负采样），默认为均匀分布
        self.item_prob = np.ones([self.item_maxid + 1], dtype=np.float32)

    def __len__(self):
        """返回用户的总数。"""
        return len(self.user_set)

    def __getitem__(self, user_id):
        """
        根据 user_id 获取一个训练样本。
        注意：负采样现在在训练时动态生成，这里只返回基础数据
        返回: 输入序列, 目标序列, 序列长度
        """
        input_seq = self.train_seq[user_id]
        target_seq = self.valid_seq[user_id]
        input_len = self.seq_len[user_id]

        return input_seq, target_seq, input_len

    def get_maxid(self):
        """返回数据集中最大的物品ID。"""
        return self.itemnum

    def get_user_set(self):
        """返回所有用户的ID集合。"""
        return self.user_set

    def get_usernum(self):
        """返回用户数量"""
        return self.usernum

    def get_itemnum(self):
        """返回物品数量"""
        return self.itemnum

    def get_dataset(self):
        """返回数据集分割结果"""
        return [self.user_train, self.user_valid, self.user_test, self.usernum, self.itemnum]



def evaluate(model, dataset, maxlen, neg_num, k, full_eval=False, device='cuda'):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG, HT, valid_user = 0.0, 0.0, 0

    users = range(1, usernum + 1) if usernum <= 10000 else random.sample(range(1, usernum + 1), 10000)

    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        # 构造序列
        seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        seq[idx] = valid[u][0]   # 倒数第二个放在序列最后
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        # 过滤：训练+验证
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

        # 预测
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])[0]
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1
        if rank < k:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, maxlen, neg_num, k, full_eval=False,device='cuda'):
    """
    在验证集上评估模型，基于pmixer的evaluate_valid函数

    参数:
    - model: 训练好的模型
    - dataset: 数据集分割结果 [user_train, user_valid, user_test, usernum, itemnum]
    - maxlen: 序列最大长度
    - device: 设备

    返回:
    - (NDCG@10, HR@10)
    """
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
            # 全量评估：候选集合 = 全部物品 - 用户历史
            item_idx = [valid [u] [0]] + [i for i in range (1, itemnum + 1) if i not in rated]
        else:
            # 负采样评估：候选集合 = 正样本 + 100 随机负样本
            item_idx = [valid [u] [0]]
            for _ in range (neg_num):
                t = np.random.randint (1, itemnum + 1)
                while t in rated:
                    t = np.random.randint (1, itemnum + 1)
                item_idx.append (t)

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


def evaluate_for_bert(model, dataset, maxlen, neg_num, k, full_eval=False, device='cuda'):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG, HT, valid_user = 0.0, 0.0, 0
    users = range(1, usernum + 1) if usernum <= 10000 else random.sample(range(1, usernum + 1), 10000)

    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        # CORRECTED: Sequence for prediction should contain the user's FULL history
        # (training items + validation item)
        seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1

        # Add the validation item to the history first
        if valid[u]:
            seq[idx] = valid[u][0]
            idx -= 1

        # Add training items
        for i in reversed(train[u]):
            if idx == -1: break
            seq[idx] = i
            idx -= 1

        rated = set(train[u]) | set(valid[u])
        rated.add(0)

        ground_truth_item = test[u][0]

        if full_eval:
            item_idx = [ground_truth_item] + [i for i in range(1, itemnum + 1) if i not in rated]
        else:
            item_idx = [ground_truth_item]
            for _ in range(neg_num):
                t = np.random.randint(1, itemnum + 1)
                while t in rated:
                    t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])

        # ROBUST FIX: Ensure predictions is a 1D tensor before ranking.
        # .squeeze() removes dimensions of size 1, e.g., (1, 1081) -> (1081,)
        predictions = predictions.squeeze()

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1
        if rank < k:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    # Avoid division by zero if no valid users were found
    return (NDCG / valid_user, HT / valid_user) if valid_user > 0 else (0.0, 0.0)


def evaluate_valid_for_bert(model, dataset, maxlen, neg_num, k, full_eval=False, device='cuda'):
    [train, valid, _, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    users = range(1, usernum + 1) if usernum <= 10000 else random.sample(range(1, usernum + 1), 10000)

    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1:
            continue

        # CORRECTED: The sequence for prediction should ONLY contain the training history.
        seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(train[u]):
            if idx == -1: break
            seq[idx] = i
            idx -= 1

        rated = set(train[u])
        rated.add(0)

        ground_truth_item = valid[u][0]

        if full_eval:
            item_idx = [ground_truth_item] + [i for i in range(1, itemnum + 1) if i not in rated]
        else:
            item_idx = [ground_truth_item]
            for _ in range(neg_num):
                t = np.random.randint(1, itemnum + 1)
                while t in rated:
                    t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])

        # ROBUST FIX: Ensure predictions is a 1D tensor before ranking.
        # .squeeze() removes dimensions of size 1, e.g., (1, 1081) -> (1081,)
        predictions = predictions.squeeze()

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1
        if rank < k:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    # Avoid division by zero if no valid users were found
    return (NDCG / valid_user, HT / valid_user) if valid_user > 0 else (0.0, 0.0)
