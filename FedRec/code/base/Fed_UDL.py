import numpy as np
import torch
import time  # 添加time模块用于计时
from torch.utils.data import DataLoader

from FedRec.code.dataset import ClientsDataset, evaluate, evaluate_valid
from FedRec.code.metric import NDCG_binary_at_k_batch, AUC_at_k_batch, HR_at_k_batch
from FedRec.code.untils import getModel
import copy
import random


class Clients:
    """
    联邦学习客户端类 - 支持异构设备
    主要改进：
    1. 支持不同设备类型（小型/中型/大型）及其对应嵌入维度
    2. 支持本地梯度提取子矩阵
    3. 支持参数等价性维护
    """

    def __init__(self, config, logger):
        self.neg_num = config['neg_num']
        self.logger = logger
        self.config = config

        # 添加异构设备维度配置
        self.dim_s = config['dim_s']  # 小型设备嵌入维度
        self.dim_m = config['dim_m']  # 中型设备嵌入维度
        self.dim_l = config['dim_l']  # 大型设备嵌入维度

        # 数据路径
        self.data_path = config['datapath'] + config['dataset'] + '/' + config['train_data']
        self.maxlen = config['max_seq_len']
        self.batch_size = config['batch_size']

        # 加载客户端数据集
        self.clients_data = ClientsDataset(self.data_path, maxlen=self.maxlen)
        self.dataset = self.clients_data.get_dataset()
        self.user_train, self.user_valid, self.user_test, self.usernum, self.itemnum = self.dataset

        # 设备选择
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 初始化SASRec模型 - 使用大型设备维度作为基础
        config['embed_size'] = self.dim_l  # 使用最大维度作为基础
        self.model = getModel(config, self.clients_data.get_maxid())
        self.model.to(self.device)

        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'],
                                          betas=(0.9, 0.98), weight_decay=config['l2_reg'])

        # 记录客户端设备类型
        self.device_types = config.get('device_types', {})  # {uid: 's', 'm', 'l'}

        self._init_device_types()

        # 创建三种尺寸的模型
        config_s = self.config.copy()
        config_s['hidden_size'] = self.dim_s
        self.model_s = getModel(config_s, self.clients_data.get_maxid()).to(self.device)

        config_m = self.config.copy()
        config_m['hidden_size'] = self.dim_m
        self.model_m = getModel(config_m, self.clients_data.get_maxid()).to(self.device)

        config_l = self.config.copy()
        config_l['hidden_size'] = self.dim_l
        self.model_l = getModel(config_l, self.clients_data.get_maxid()).to(self.device)

        # Server类需要一个统一的model引用，我们让它引用最大模型
        self.model = self.model_l

        self.logger.info(f"异构设备配置: dim_s={self.dim_s}, dim_m={self.dim_m}, dim_l={self.dim_l}")
        self.logger.info(f"设备类型分布: "
                         f"小型:{sum(1 for t in self.device_types.values() if t == 's')}, "
                         f"中型:{sum(1 for t in self.device_types.values() if t == 'm')}, "
                         f"大型:{sum(1 for t in self.device_types.values() if t == 'l')}")

    def _init_device_types ( self ):
        user_set = self.clients_data.get_user_set ()
        total_users = len (user_set)

        device_split = self.config ['device_split']
        num_s = int (total_users * device_split [0])
        num_m = int (total_users * (device_split [0] + device_split [1]))

        if self.config ['assign_by_interactions']:
            # 获取用户交互次数并排序（升序）
            method = "按交互次序分配"
            user_interactions = {uid: len (self.user_train [uid]) for uid in user_set}
            sorted_users = sorted (user_set, key = lambda uid: user_interactions [uid])
        else:
            # 随机打乱
            method = "随机分配"
            sorted_users = list (user_set)
            np.random.shuffle (sorted_users)

        # 分配设备类型
        for i, uid in enumerate (sorted_users):
            if i < num_s:
                self.device_types [uid] = 's'
            elif i < num_m:
                self.device_types [uid] = 'm'
            else:
                self.device_types [uid] = 'l'

        # 统计结果
        count_s = sum (1 for t in self.device_types.values () if t == 's')
        count_m = sum (1 for t in self.device_types.values () if t == 'm')
        count_l = sum (1 for t in self.device_types.values () if t == 'l')

        self.logger.info (
            f"嵌套设备类型分配完成({method}): "
            f"小型={count_s}({count_s / total_users:.1%}), "
            f"中型={count_m}({count_m / total_users:.1%}), "
            f"大型={count_l}({count_l / total_users:.1%})"
        )

    def get_dim_for_client(self, uid):
        """获取客户端的嵌入维度"""
        dev_type = self.device_types.get(uid, 'l')
        return {
            's': self.dim_s,
            'm': self.dim_m,
            'l': self.dim_l
        }[dev_type]


    def load_server_model_params(self, client_model, server_state_dict):
        """
        将服务器的全局模型参数加载到客户端模型中。
        只在“特征维/隐藏维”上切片，不裁剪索引维（词表数、位置数）。
        """
        client_state_dict = client_model.state_dict()

        for name, server_param in server_state_dict.items():
            if name in client_state_dict:
                client_param = client_state_dict[name]
                # 如果服务器的参数维度比客户端大，则进行切片
                if server_param.shape != client_param.shape:
                    # 从大参数矩阵中裁剪出与小模型匹配的部分
                    slicing_indices = [slice(0, dim) for dim in client_param.shape]
                    client_param.copy_(server_param[tuple(slicing_indices)])
                else:
                    # 如果维度相同，直接复制
                    client_param.copy_(server_param)

    def get_local_grads_for_device(self, uid, gradients):
        """
        根据设备类型提取梯度子矩阵
        考虑嵌套关系：
        - 小型设备：只提取前1/3维度
        - 中型设备：提取前2/3维度
        - 大型设备：提取全部维度
        """
        dev_type = self.device_types.get(uid, 'l')
        device_grads = {}

        for name, grad in gradients.items():
            if grad is None:
                continue

            # 对于物品嵌入，提取设备对应维度的子集
            if 'item_embedding' in name:
                if dev_type == 's':
                    # 小型设备：只取前1/3维度
                    device_grads[name] = grad[:, :self.dim_s].clone()
                elif dev_type == 'm':
                    # 中型设备：取前2/3维度
                    device_grads[name] = grad[:, :self.dim_m].clone()
                else:  # 'l'
                    # 大型设备：取全部维度
                    device_grads[name] = grad.clone()
            else:
                device_grads[name] = grad.clone()

        return device_grads

    def train(self, uids, model_param_state_dict, epoch=0):
        """
        联邦学习训练方法 - 动态负采样版本
        实现标准的联邦学习流程：
        1. 每个客户端独立训练，保护数据隐私
        2. 动态生成负样本，提高学习效果
        3. 分离隐私梯度（不含embedding）和非隐私梯度（含embedding）
        4. 隐私梯度在本地更新，非隐私梯度上传服务器
        5. 返回所有客户端的非隐私梯度字典用于服务器聚合

        参数:
        - uids: 当前批次需要训练的客户端ID列表
        - model_param_state_dict: 从服务器接收的最新全局模型参数
        - epoch: 当前训练轮次，用于动态采样策略调整

        返回:
        - clients_grads: 所有客户端的非隐私梯度字典 {uid: {param_name: grad}}
        - clients_losses: 所有客户端的损失值字典 {uid: loss}
        """

        # 存储客户端梯度和损失
        clients_grads = {}
        clients_losses = {}

        # 每个客户端独立训练
        for uid in uids:
            uid = uid.item()
            dev_type = self.device_types.get(uid, 'l')

            # 1. 根据设备类型选择正确的模型
            if dev_type == 's':
                client_model = self.model_s
            elif dev_type == 'm':
                client_model = self.model_m
            else:  # 'l'
                client_model = self.model_l

            client_model.train()

            # 2. 将服务器参数加载到选定的客户端模型中
            self.load_server_model_params(client_model, model_param_state_dict)

            # 3. 为当前模型创建优化器
            optimizer = torch.optim.Adam(client_model.parameters(), lr=self.config['lr'],
                                         betas=(0.9, 0.98), weight_decay=self.config['l2_reg'])

            # 4. 准备该用户的数据
            input_seq = self.clients_data.train_seq[uid]
            target_seq = self.clients_data.valid_seq[uid]
            input_len = self.clients_data.seq_len[uid]
            # 负采样
            # 去除用户交互过的物品
            cand = np.setdiff1d(self.clients_data.item_set, self.clients_data.seq[uid])
            # 计算物品的采样概率
            prob = self.clients_data.item_prob[cand]
            prob = prob / prob.sum()
            # 随机采样
            neg_seq = np.random.choice(cand, (input_len, 100), p=prob)
            # 填充
            neg_seq = np.pad(neg_seq, ((input_seq.shape[0] - input_len, 0), (0, 0)))
            # 转换为张量
            input_seq = torch.from_numpy(input_seq).unsqueeze(0).to(self.device)
            target_seq = torch.from_numpy(target_seq).unsqueeze(0).to(self.device)
            neg_seq = torch.from_numpy(neg_seq).unsqueeze(0).to(self.device)
            input_len = torch.tensor(input_len).unsqueeze(0).to(self.device)
            max_seq_length = client_model.max_seq_length
            # 处理序列长度限制
            if input_seq.size(1) > max_seq_length:
                input_seq = input_seq[:, -max_seq_length:]
                target_seq = target_seq[:, -max_seq_length:]
                neg_seq = neg_seq[:, -max_seq_length:, :]  # 添加负采样序列截断
                input_len = torch.clamp(input_len, max=max_seq_length)

            # 5. 使用尺寸正确的 client_model 进行训练
            seq_out = client_model(input_seq, input_len)
            padding_mask = (torch.not_equal(input_seq, 0)).float().unsqueeze(-1).to(self.device)

            # 计算损失
            # 根据设备类型选择损失函数
            if dev_type == 's':

                loss = self.model_s.loss_function(seq_out, padding_mask, target_seq, neg_seq, input_len)
            elif dev_type == 'm':

                loss_s=self.model_s.loss_function(seq_out, padding_mask, target_seq, neg_seq, input_len)
                loss_m=self.model_m.loss_function(seq_out, padding_mask, target_seq, neg_seq, input_len)
                loss = loss_s + loss_m
            else:  # 'l'

                loss_s=self.model_s.loss_function(seq_out, padding_mask, target_seq, neg_seq, input_len)
                loss_m=self.model_m.loss_function(seq_out, padding_mask, target_seq, neg_seq, input_len)
                loss_l=self.model_l.loss_function(seq_out, padding_mask, target_seq, neg_seq, input_len)
                loss = loss_s + loss_m + loss_l

            # 保存损失值
            clients_losses[uid] = loss.item()

            # 反向传播计算梯度
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            gradients = {name: param.grad.clone() for name, param in client_model.named_parameters() if
                         param.grad is not None}
            clients_grads[uid] = gradients

        return clients_grads, clients_losses


class Server:
    """
    联邦学习服务器类
    主要功能：
    1. 实现FedAvg梯度聚合算法
    2. 处理多个客户端的梯度平均
    3. 维护全局模型并进行评估
    """

    def __init__(self, config, clients, logger):
        self.clients = clients
        self.config = config
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.early_stop = config['early_stop']
        self.maxlen = config['max_seq_len']
        self.skip_test_eval = config['skip_test_eval']
        self.eval_freq = config['eval_freq']
        self.early_stop_enabled = config['early_stop_enabled']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logger
        self.dataset = self.clients.dataset

        # 获取异构设备维度配置
        self.dim_s = config['dim_s']
        self.dim_m = config['dim_m']
        self.dim_l = config['dim_l']
        
        # 获取评估k值配置
        self.eval_k = config['eval_k']

        # 初始化模型 - 使用大型设备维度
        config['embed_size'] = self.dim_l
        self.model = getModel(config, self.clients.clients_data.get_maxid())
        self.model.to(self.device)

        # 初始化优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'],
                                          betas=(0.9, 0.98), weight_decay=config['l2_reg'])

    # def _padding(self, grad, target_dim):
    #     """梯度填充函数"""
    #     current_dim = grad.shape[1]
    #     if current_dim < target_dim:
    #         padding = torch.zeros((grad.shape[0], target_dim - current_dim), device=self.device)
    #         return torch.cat((grad, padding), dim=1)
    #     return grad

    def aggregate_gradients(self, clients_grads):
        """执行基于填充的梯度聚合 - 支持嵌套设备类型"""
        clients_num = len(clients_grads)
        if clients_num == 0:
            self.logger.warning("没有收到任何客户端梯度")
            return

        aggregated_gradients = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if
                                param.requires_grad}

        # 聚合每个客户端的梯度
        for uid, grads_dict in clients_grads.items():
            for name, grad in grads_dict.items():
                if grad is None:
                    continue

                target_shape = aggregated_gradients[name].shape

                # 如果收到的梯度尺寸不匹配，进行填充
                if grad.shape != target_shape:
                    padded_grad = torch.zeros(target_shape, device=self.device)
                    slicing_indices = tuple(slice(0, dim) for dim in grad.shape)
                    padded_grad[slicing_indices] = grad
                    aggregated_gradients[name] += padded_grad
                else:
                    # 如果尺寸匹配，直接累加
                    aggregated_gradients[name] += grad

        # 计算平均梯度
        for name in aggregated_gradients:
            aggregated_gradients[name] /= clients_num

        # 应用梯度到模型
        for name, param in self.model.named_parameters():
            if name in aggregated_gradients:
                param.grad = aggregated_gradients[name]
            else:
                param.grad = None

    def train(self):
        """
        真正的联邦学习训练循环
        """
        # 创建客户端批次序列
        user_set = self.clients.clients_data.get_user_set()
        uid_seq = []
        for i in range(0, len(user_set), self.batch_size):
            batch_uids = user_set[i:i + self.batch_size]
            uid_seq.append(torch.tensor(batch_uids))

        # 早停相关变量
        best_val_ndcg, best_val_hr = 0.0, 0.0
        best_test_ndcg, best_test_hr = 0.0, 0.0

        # 早停计数器 - 记录连续没有NDCG改善的轮数
        no_improve_count = 0

        # 初始化模型参数
        for name, param in self.model.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass  # 忽略初始化失败的层

        # 设置embedding层的padding位置为0
        if hasattr(self.model, 'position_embedding'):
            self.model.position_embedding.weight.data[0, :] = 0
        if hasattr(self.model, 'item_embedding'):
            self.model.item_embedding.weight.data[0, :] = 0

        T = 0.0
        t0 = time.time()

        # 开始多轮训练
        early_stop_triggered = False  # 添加早停标志
        for epoch in range(self.epochs):
            # 记录epoch开始时间
            epoch_start_time = time.time()

            # 切换模型到训练模式
            self.model.train()

            # 训练统计变量
            batch_count = 0
            epoch_losses = []

            # 遍历所有客户端批次
            for uids in uid_seq:
                # 清零梯度
                self.optimizer.zero_grad()

                # 获取所有客户端的梯度字典和损失值字典
                clients_grads, clients_losses = self.clients.train(uids, self.model.state_dict(), epoch)

                # 收集本批次的损失值
                batch_losses = list(clients_losses.values())
                epoch_losses.extend(batch_losses)

                # 使用FedAvg梯度聚合方法
                self.aggregate_gradients(clients_grads)

                # 更新全局模型参数
                self.optimizer.step()

                # 记录批次统计
                batch_count += 1

            eval_freq = self.eval_freq
            # 在第一个epoch、每eval_freq个epoch或最后一个epoch进行评估
            should_evaluate = (epoch + 1) % eval_freq == 0 or epoch == 0 or epoch == self.epochs - 1
            if should_evaluate:
                self.model.eval()
                # t1是该轮训练时间 T为总时间
                t1 = time.time() - t0
                T += t1
                print('Evaluating', end='')

                t_valid = evaluate_valid(self.model, self.dataset, self.maxlen, self.clients.neg_num, self.eval_k, self.config['full_eval'],self.device)
                self.logger.info(
                    f"传统评估结果 - Epoch {epoch + 1}: NDCG@{self.eval_k}={t_valid[0]:.4f}, HR@{self.eval_k}={t_valid[1]:.4f}")

                # 检查评估结果是否异常
                if t_valid[0] > 1.0 or t_valid[1] > 1.0 or np.isnan(t_valid[0]) or np.isnan(t_valid[1]):
                    self.logger.info(f"检测到异常评估结果: NDCG@{self.eval_k}={t_valid[0]:.4f}, HR@{self.eval_k}={t_valid[1]:.4f}")

                # 早停检查 - 检查NDCG是否有所改善
                if self.early_stop_enabled:
                    if t_valid[0] > best_val_ndcg:
                        # NDCG有改善，重置计数器
                        no_improve_count = 0
                        best_val_ndcg = t_valid[0]
                    else:
                        # NDCG没有改善，增加计数器
                        no_improve_count += 1

                    # 如果连续多轮没有改善，触发早停
                    if no_improve_count >= self.early_stop:
                        self.logger.info(f"早停触发！NDCG在{self.early_stop}轮内没有改善。")
                        early_stop_triggered = True

                # 测试集评估（可选）- 也使用异构评估
                if not self.skip_test_eval:
                    # 这里可以添加测试集的异构评估，暂时使用传统方法
                    t_test = evaluate(self.model, self.dataset, self.maxlen, self.clients.neg_num, self.eval_k, self.config['full_eval'],self.device)

                    # 检查测试集评估结果是否异常
                    if t_test[0] > 1.0 or t_test[1] > 1.0 or np.isnan(t_test[0]) or np.isnan(t_test[1]):
                        self.logger.info(f"检测到异常测试结果: NDCG@{self.eval_k}={t_test[0]:.4f}, HR@{self.eval_k}={t_test[1]:.4f}")

                    # 记录到日志
                    self.logger.info(
                                            'epoch:%d, time: %f(s), valid (NDCG@%d: %.4f, HR@%d: %.4f), test (NDCG@%d: %.4f, HR@%d: %.4f) all_time: %f(s)'
                    % (epoch + 1, t1, self.eval_k, t_valid[0], self.eval_k, t_valid[1], self.eval_k, t_test[0], self.eval_k, t_test[1], T))
                else:
                    # 跳过测试集评估，设置默认值
                    t_test = (0.0, 0.0)
                    # 记录到日志
                    self.logger.info(
                                            'epoch:%d, time: %f(s), valid (NDCG@%d: %.4f, HR@%d: %.4f), test: SKIPPED, all_time: %f(s)'
                    % (epoch + 1, t1, self.eval_k, t_valid[0], self.eval_k, t_valid[1], T))

                # 更新最佳结果
                if not self.skip_test_eval:
                    # 包含测试集评估的情况
                    if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr or t_test[0] > best_test_ndcg or \
                            t_test[1] > best_test_hr:
                        best_val_ndcg = max(t_valid[0], best_val_ndcg)
                        best_val_hr = max(t_valid[1], best_val_hr)
                        best_test_ndcg = max(t_test[0], best_test_ndcg)
                        best_test_hr = max(t_test[1], best_test_hr)
                        self.logger.info(
                            f"新的最佳性能: valid NDCG@{self.eval_k}={best_val_ndcg:.4f}, test NDCG@{self.eval_k}={best_test_ndcg:.4f}")

                else:
                    # 跳过测试集评估的情况，只基于验证集更新
                    if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr:
                        best_val_ndcg = max(t_valid[0], best_val_ndcg)
                        best_val_hr = max(t_valid[1], best_val_hr)
                        self.logger.info(
                            f"新的最佳性能: valid NDCG@{self.eval_k}={best_val_ndcg:.4f}, valid HR@{self.eval_k}={best_val_hr:.4f}")

                t0 = time.time()
                self.model.train()

                # 在评估完成后检查早停标志
                if early_stop_triggered:
                    break

                # 如果早停被触发，跳出训练循环
            if early_stop_triggered:
                break

        # 记录最佳结果
        if not self.skip_test_eval:
            self.logger.info(
                            '[联邦训练] 最佳结果: valid NDCG@{}={:.4f}, HR@{}={:.4f}, test NDCG@{}={:.4f}, HR@{}={:.4f}'.format(
                self.eval_k, best_val_ndcg, self.eval_k, best_val_hr, self.eval_k, best_test_ndcg, self.eval_k, best_test_hr))
        else:
            self.logger.info('[联邦训练] 最佳结果: valid NDCG@{}={:.4f}, HR@{}={:.4f} (测试集评估已跳过)'.format(
                self.eval_k, best_val_ndcg, self.eval_k, best_val_hr))