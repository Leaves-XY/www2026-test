import numpy as np
import bottleneck as bn


def NDCG_binary_at_k_batch ( X_pred, heldout_batch, k = 5 ):
    '''
    批量计算二进制相关性下的 NDCG@k (归一化折损累计增益)。
    这是一个衡量排名质量的指标，考虑了推荐物品的位置。
    假设：heldout_batch 中的 0 表示不相关，非 0 表示相关。

    参数:
    - X_pred: (batch_users, num_items) 形状的预测得分矩阵。
    - heldout_batch: (batch_users, num_items) 形状的真实标签矩阵 (测试集)。
    - k: 推荐列表的长度。

    返回:
    - 每个用户的 NDCG@k 分数数组。
    '''
    batch_users = X_pred.shape [0]
    # 使用 bottleneck.argpartition 高效地找到每个用户得分最高的前 k 个物品的索引
    # argpartition 并不完全排序，只是将 top-k 的元素放在前 k 个位置，速度更快
    idx_topk_part = bn.argpartition (-X_pred, k, axis = 1)
    # 获取这 top-k 个物品的实际预测得分
    topk_part = X_pred [np.arange (batch_users) [:, np.newaxis], idx_topk_part [:, :k]]
    # 对这 top-k 个物品的得分进行完全排序，得到最终的 top-k 索引
    idx_part = np.argsort (-topk_part, axis = 1)
    idx_topk = idx_topk_part [np.arange (batch_users) [:, np.newaxis], idx_part]

    # 创建一个折损(discount)模板，位置越靠后，折损越大
    # tp = [1/log2(2), 1/log2(3), ..., 1/log2(k+1)]
    tp = 1. / np.log2 (np.arange (2, k + 2))

    # 计算 DCG (Discounted Cumulative Gain)
    # 将 top-k 推荐命中情况 (0或1) 与折损模板相乘再求和
    DCG = (heldout_batch [np.arange (batch_users) [:, np.newaxis], idx_topk] * tp).sum (axis = 1)

    # 计算 IDCG (Ideal Discounted Cumulative Gain)
    # IDCG 是理想情况下的 DCG，即所有真实相关的物品都排在最前面
    IDCG = np.array ([(tp [:min (n, k)]).sum () for n in heldout_batch.sum (axis = 1).astype (np.int32)])

    # NDCG = DCG / IDCG
    return DCG / IDCG


def Recall_Precision_F1_OneCall_at_k_batch ( X_pred, heldout_batch, k = 5 ):
    '''
    批量计算 Recall@k, Precision@k, F1-score@k, 和 1-call@k。

    参数:
    - X_pred: 预测得分矩阵。
    - heldout_batch: 真实标签矩阵。
    - k: 推荐列表的长度。

    返回:
    - recall, precision, f1, oneCall: 四个评估指标的数组。
    '''
    batch_users = X_pred.shape [0]
    # 找到每个用户得分最高的前 k 个物品的索引
    idx = bn.argpartition (-X_pred, k, axis = 1)
    # 创建一个布尔矩阵，将 top-k 推荐的位置标记为 True
    X_pred_binary = np.zeros_like (X_pred, dtype = bool)
    X_pred_binary [np.arange (batch_users) [:, np.newaxis], idx [:, :k]] = True
    # 创建一个布尔矩阵，表示真实交互的物品
    X_true_binary = (heldout_batch > 0)

    # 计算TP (True Positive)，即推荐命中且真实交互的物品数量
    tmp = (np.logical_and (X_true_binary, X_pred_binary).sum (axis = 1)).astype (np.float32)

    # Recall = TP / (所有真实相关的物品数)
    recall = tmp / X_true_binary.sum (axis = 1)
    # Precision = TP / (推荐的物品数 k)
    precision = tmp / k
    # F1-score 是 recall 和 precision 的调和平均数
    f1 = 2 * recall * precision / (recall + precision)
    # 1-call (或 Hit Rate) 表示 top-k 列表中是否至少命中一个真实物品
    oneCall = (tmp > 0).astype (np.float32)

    return recall, precision, f1, oneCall


def HR_at_k_batch ( X_pred, heldout_batch, k = 5 ):
    '''
    批量计算 Hit Ratio (命中率)@k。
    命中率表示在 Top-k 推荐中至少有一个项目是相关的用户所占的比例。

    参数:
    - X_pred: 预测得分矩阵。
    - heldout_batch: 真实标签矩阵。
    - k: 推荐列表的长度。

    返回:
    - hr: 该批次用户的平均命中率。
    '''
    batch_users = X_pred.shape [0]
    idx = bn.argpartition (-X_pred, k, axis = 1)
    X_pred_binary = np.zeros_like (X_pred, dtype = bool)
    X_pred_binary [np.arange (batch_users) [:, np.newaxis], idx [:, :k]] = True
    X_true_binary = (heldout_batch > 0)

    # 对于每个用户，如果在Top-k推荐中有至少一个相关项目，则为一次“命中”
    hit = np.logical_and (X_true_binary, X_pred_binary).sum (axis = 1) > 0
    # 计算所有用户的平均命中率
    hr = hit.mean ()
    return hr


def AUC_at_k_batch ( X_train, X_pred, heldout_batch ):
    '''
    批量计算 AUC (Area Under the ROC Curve)。
    AUC 衡量的是模型将随机选择的正样本排在随机选择的负样本前面的概率。

    参数:
    - X_train: 训练集交互矩阵，用于识别已交互物品。
    - X_pred: 预测得分矩阵。
    - heldout_batch: 测试集真实标签矩阵。

    返回:
    - aucs: 每个用户的 AUC 分数数组。
    '''
    train_set_num = X_train.sum (axis = 1)
    test_set_num = heldout_batch.sum (axis = 1)
    # 计算每个物品得分的排名（rank）
    rank = np.argsort (np.argsort (X_pred)) + 1

    # AUC 公式的一种计算方式
    # 分子：所有正样本的排名之和，减去一个修正项
    molecular = (heldout_batch * rank).sum (axis = 1) - test_set_num * (
            test_set_num + 1) / 2 - test_set_num * train_set_num
    # 分母：负样本对的总数（即 (总物品数 - 正样本数) * 正样本数）
    denominator = (X_pred.shape [1] - train_set_num - test_set_num) * test_set_num

    aucs = molecular / denominator
    return aucs
