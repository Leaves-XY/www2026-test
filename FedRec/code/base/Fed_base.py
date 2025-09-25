import numpy as np
import torch
import time  # 添加time模块用于计时
from FedRec.code.dataset import ClientsDataset, evaluate, evaluate_valid, evaluate_for_bert, evaluate_valid_for_bert
from FedRec.code.metric import NDCG_binary_at_k_batch, AUC_at_k_batch, HR_at_k_batch
from FedRec.code.untils import getModel


class Clients:
    """
    联邦学习客户端类 - 支持异构设备
    主要改进：
    1. 支持不同设备类型（小型/中型/大型）及其对应嵌入维度
    2. 支持本地梯度提取子矩阵
    3. 支持参数等价性维护
    """

    def __init__ ( self, config, logger ):
        self.neg_num = config ['neg_num']
        self.logger = logger
        self.config = config

        # 添加异构设备维度配置
        self.dim_s = config ['dim_s']  # 小型设备嵌入维度
        self.dim_m = config ['dim_m']  # 中型设备嵌入维度
        self.dim_l = config ['dim_l']  # 大型设备嵌入维度

        # 数据路径
        self.data_path = config ['datapath'] + config ['dataset'] + '/' + config ['train_data']
        self.maxlen = config ['max_seq_len']
        self.batch_size = config ['batch_size']

        # 加载客户端数据集
        self.clients_data = ClientsDataset (self.data_path,maxlen = self.maxlen)
        self.dataset = self.clients_data.get_dataset ()
        self.user_train, self.user_valid, self.user_test, self.usernum, self.itemnum = self.dataset

        # 设备选择
        self.device = "cuda" if torch.cuda.is_available () else "cpu"

        # 记录客户端设备类型
        self.device_types = {} # {uid: 's', 'm', 'l'}

        self._init_device_types ()

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

        self.logger.info (f"异构设备配置: dim_s={self.dim_s}, dim_m={self.dim_m}, dim_l={self.dim_l}")
        self.logger.info (f"设备类型分布: "
                          f"小型:{sum (1 for t in self.device_types.values () if t == 's')}, "
                          f"中型:{sum (1 for t in self.device_types.values () if t == 'm')}, "
                          f"大型:{sum (1 for t in self.device_types.values () if t == 'l')}")


    def _init_device_types(self):
        user_set = self.clients_data.get_user_set ()
        total_users = len (user_set)

        device_split = self.config['device_split']
        num_s = int(total_users * device_split[0])
        num_m = int(total_users * (device_split[0] + device_split[1]))

        if self.config ['assign_by_interactions']:
            # 获取用户交互次数并排序（升序）
            method="按交互次序分配"
            user_interactions = {uid: len (self.user_train [uid]) for uid in user_set}
            sorted_users = sorted (user_set, key = lambda uid: user_interactions [uid])
        else:
            # 随机打乱
            method="随机分配"
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

    def get_dim_for_client ( self, uid ):
        """获取客户端的嵌入维度"""
        dev_type = self.device_types.get (uid, 'l')
        return {
            's': self.dim_s,
            'm': self.dim_m,
            'l': self.dim_l
        } [dev_type]
    

    def load_server_model_params(self, client_model, server_state_dict):
        """
        将服务器的全局模型参数加载到指定的客户端模型中。
        对于需要适应不同维度的层（如embedding），此函数会自动进行切片。
        """
        # client_state_dict是引用值，后续对 client_param 使用 .copy_() 方法时，会直接修改模型内部的参数值
        client_state_dict = client_model.state_dict()
        
        for name, server_param in server_state_dict.items():
            if name in client_state_dict:
                client_param = client_state_dict[name]
                # 如果服务器的参数维度比客户端大，则进行切片
                if server_param.shape != client_param.shape:
                    # 从大参数矩阵中裁剪出与小模型匹配的部分
                    slicing_indices = [slice(0, dim) for dim in client_param.shape]
                    # 对 server_param 这个张量进行多维切片，得到一个和 client_param 形状一致的子张量。
                    client_param.copy_(server_param[tuple(slicing_indices)])
                else:
                    # 如果维度相同，直接复制
                    client_param.copy_(server_param)

    def train(self, uids, model_param_state_dict, epoch=0):
        """
        联邦学习训练方法 - 动态负采样版本
        实现标准的联邦学习流程：
        1. 每个客户端独立训练，保护数据隐私
        2. 动态生成负样本，提高学习效果
        3. 为每个客户端选择对应尺寸的模型进行训练
        4. 返回所有客户端的梯度字典用于服务器聚合
        """
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
            #去除用户交互过的物品
            cand = np.setdiff1d(self.clients_data.item_set, self.clients_data.seq[uid])
            # 计算物品的采样概率
            prob = self.clients_data.item_prob[cand]
            prob = prob / prob.sum()
            #随机采样
            neg_seq = np.random.choice(cand, (input_len, 100), p=prob)
            #填充
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
                neg_seq = neg_seq[:, -max_seq_length:, :]
                input_len = torch.clamp(input_len, max=max_seq_length)

            # 5. 使用尺寸正确的 client_model 进行训练
            if self.config['model'] == 'BERT4Rec':
                # --- BERT4Rec Data Generation: Masking ---
                mask_prob = self.config['mask_prob']  # BERT4Rec的掩码概率
                input_seq_numpy = input_seq.cpu().numpy().squeeze(0)
                masked_seq_numpy = np.copy(input_seq_numpy)
                labels = np.zeros_like(input_seq_numpy)

                candidate_indices = np.where(masked_seq_numpy > 0)[0]
                num_to_mask = max(1, int(len(candidate_indices) * mask_prob))
                masked_indices = np.random.choice(candidate_indices, num_to_mask, replace=False)

                for index in masked_indices:
                    labels[index] = masked_seq_numpy[index]
                    masked_seq_numpy[index] = client_model.mask_token_id

                masked_seq = torch.from_numpy(masked_seq_numpy).unsqueeze(0).to(self.device)
                labels = torch.from_numpy(labels).unsqueeze(0).to(self.device)
                
                seq_out = client_model(masked_seq)
                loss = client_model.loss_function(seq_out, labels)
            else:
                # 原有的 next-item prediction 逻辑
                seq_out = client_model(input_seq, input_len)
                padding_mask = (torch.not_equal(input_seq, 0)).float().unsqueeze(-1).to(self.device)
                loss = client_model.loss_function(seq_out, padding_mask, target_seq, neg_seq, input_len)

            clients_losses[uid] = loss.item()

            # 6. 反向传播并收集梯度
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            gradients = {name: param.grad.clone() for name, param in client_model.named_parameters() if param.grad is not None}
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
        config['hidden_size'] = self.dim_l
        self.model = getModel(config, self.clients.clients_data.get_maxid())
        self.model.to(self.device)

        # 初始化优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'],
                                          betas=(0.9, 0.98), weight_decay=config['l2_reg'])

        # --- 通信开销计算模块初始化 ---
        self._init_comm_cost_calculation()

        logger.info("服务器初始化完成")

    def _get_param_size(self, model_or_params):
        """计算模型或参数字典的大小（字节）"""
        if isinstance(model_or_params, torch.nn.Module):
            params = model_or_params.parameters()
        elif isinstance(model_or_params, dict):
            params = model_or_params.values()
        else:
            params = model_or_params

        size = 0
        for p in params:
            if isinstance(p, torch.Tensor):
                size += p.nelement() * p.element_size()
        return size

    def _get_layer_size(self, model, layer_name):
        """获取模型特定层的大小（字节）"""
        size = 0
        for name, param in model.named_parameters():
            if layer_name in name:
                size += param.nelement() * param.element_size()
        return size

    def _init_comm_cost_calculation(self):
        """预计算并存储模型/层的尺寸，用于开销统计"""
        self.comm_costs = {'downlink': 0, 'uplink': 0}
        self.logger.info("初始化通信开销计算器...")

        # 预计算三种尺寸模型的总参数大小
        self.size_p_s = self._get_param_size(self.clients.model_s)
        self.size_p_m = self._get_param_size(self.clients.model_m)
        self.size_p_l = self._get_param_size(self.clients.model_l)

        # 预计算三种尺寸模型 item_embedding 层的大小
        self.size_e_s = self._get_layer_size(self.clients.model_s, 'item_embedding.weight')
        self.size_e_m = self._get_layer_size(self.clients.model_m, 'item_embedding.weight')
        self.size_e_l = self._get_layer_size(self.clients.model_l, 'item_embedding.weight')

        self.logger.info(f"模型总尺寸 (MB): S={self.size_p_s / 1e6:.3f}, M={self.size_p_m / 1e6:.3f}, L={self.size_p_l / 1e6:.3f}")
        self.logger.info(f"Embedding层尺寸 (MB): S={self.size_e_s / 1e6:.3f}, M={self.size_e_m / 1e6:.3f}, L={self.size_e_l / 1e6:.3f}")



    def aggregate_gradients(self, clients_grads):
        """执行基于填充的梯度聚合 - 支持异构设备"""
        clients_num = len(clients_grads)
        if clients_num == 0:
            self.logger.info("没有收到任何客户端梯度")
            return

        # 初始化一个空的聚合梯度字典
        aggregated_gradients = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}

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


    def train ( self ):
        """
        真正的联邦学习训练循环
        """
        # 创建客户端批次序列
        user_set = self.clients.clients_data.get_user_set ()
        uid_seq = []
        for i in range (0, len (user_set), self.batch_size):
            batch_uids = user_set [i:i + self.batch_size]
            uid_seq.append (torch.tensor (batch_uids))

        # 早停相关变量
        best_val_ndcg, best_val_hr = 0.0, 0.0
        best_test_ndcg, best_test_hr = 0.0, 0.0

        # 早停计数器 - 记录连续没有NDCG改善的轮数
        no_improve_count = 0

        # 初始化模型参数
        for name, param in self.model.named_parameters ():
            try:
                torch.nn.init.xavier_normal_ (param.data)
            except:
                pass  # 忽略初始化失败的层

        # 设置embedding层的padding位置为0
        if hasattr (self.model, 'position_embedding'):
            self.model.position_embedding.weight.data [0, :] = 0
        if hasattr (self.model, 'item_embedding'):
            self.model.item_embedding.weight.data [0, :] = 0

        T = 0.0
        t0 = time.time ()

        # 开始多轮训练
        early_stop_triggered = False  # 添加早停标志
        for epoch in range (self.epochs):
            # 记录epoch开始时间
            epoch_start_time = time.time ()

            # 切换模型到训练模式
            self.model.train ()

            # 训练统计变量
            batch_count = 0
            epoch_losses = []

            # 遍历所有客户端批次
            for uids in uid_seq:
                # --- 通信开销计算: 下行链路 ---
                # 根据每个客户端的实际模型尺寸计算下行开销
                downlink_cost_batch = 0
                for uid in uids:
                    dev_type = self.clients.device_types.get(uid.item(), 'l')
                    if dev_type == 's':
                        downlink_cost_batch += self.size_p_s
                    elif dev_type == 'm':
                        downlink_cost_batch += self.size_p_m
                    else:
                        downlink_cost_batch += self.size_p_l
                self.comm_costs['downlink'] += downlink_cost_batch
                # ---

                # 清零梯度
                self.optimizer.zero_grad ()

                # 获取所有客户端的梯度字典和损失值字典
                clients_grads, clients_losses = self.clients.train (uids, self.model.state_dict (), epoch)

                # --- 通信开销计算: 上行链路 ---
                # 计算每个客户端上传的梯度大小
                uplink_cost_batch = 0
                for uid in uids:
                    dev_type = self.clients.device_types.get(uid.item(), 'l')
                    if dev_type == 's':
                        uplink_cost_batch += self.size_p_s
                    elif dev_type == 'm':
                        uplink_cost_batch += self.size_p_m
                    else:
                        uplink_cost_batch += self.size_p_l
                self.comm_costs['uplink'] += uplink_cost_batch
                # ---

                # 收集本批次的损失值
                batch_losses = list (clients_losses.values ())
                epoch_losses.extend (batch_losses)

                # 使用FedAvg梯度聚合方法
                self.aggregate_gradients (clients_grads)

                # 更新全局模型参数
                self.optimizer.step ()

                # 记录批次统计
                batch_count += 1

            eval_freq = self.eval_freq
            # 在第一个epoch、每eval_freq个epoch或最后一个epoch进行评估
            should_evaluate = (epoch + 1) % eval_freq == 0 or epoch == 0 or epoch == self.epochs - 1
            if should_evaluate:
                self.model.eval ()
                # t1是该轮训练时间 T为总时间
                t1 = time.time () - t0
                T += t1
                print ('Evaluating', end = '')


                if self.config['model'] == 'BERT4Rec':
                    t_valid = evaluate_valid_for_bert(self.model, self.dataset, self.maxlen, self.clients.neg_num, self.eval_k, self.config['full_eval'],self.device)
                else:
                    t_valid = evaluate_valid(self.model, self.dataset, self.maxlen, self.clients.neg_num, self.eval_k, self.config['full_eval'],self.device)

                # --- 通信开销计算: 日志记录 ---
                down_mb = self.comm_costs['downlink'] / (1024**2)
                up_mb = self.comm_costs['uplink'] / (1024**2)
                total_mb = down_mb + up_mb
                self.logger.info(f"COMMUNICATION COST - Epoch {epoch + 1}: "
                                 f"Downlink={down_mb:.3f}MB, Uplink={up_mb:.3f}MB, Total={total_mb:.3f}MB")
                # ---

                self.logger.info (
                    f"传统评估结果 - Epoch {epoch + 1}: NDCG@{self.eval_k}={t_valid [0]:.4f}, HR@{self.eval_k}={t_valid [1]:.4f}")

                # 检查评估结果是否异常
                if t_valid [0] >= 0.99 or t_valid [1] >= 0.99 or np.isnan (t_valid [0]) or np.isnan (t_valid [1]):
                    self.logger.info (f"检测到异常评估结果: NDCG@{self.eval_k}={t_valid [0]:.4f}, HR@{self.eval_k}={t_valid [1]:.4f}")

                # 早停检查 - 检查NDCG是否有所改善
                if self.early_stop_enabled:
                    if t_valid [0] > best_val_ndcg:
                        # NDCG有改善，重置计数器
                        no_improve_count = 0
                        best_val_ndcg = t_valid [0]
                    else:
                        # NDCG没有改善，增加计数器
                        no_improve_count += 1

                    # 如果连续多轮没有改善，触发早停
                    if no_improve_count >= self.early_stop:
                        self.logger.info (f"早停触发！NDCG在{self.early_stop}轮内没有改善。")
                        early_stop_triggered = True

                # 测试集评估（可选）- 也使用异构评估
                if not self.skip_test_eval:
                    # 这里可以添加测试集的异构评估，暂时使用传统方法
                    if self.config['model'] == 'BERT4Rec':
                        t_test = evaluate_for_bert(self.model, self.dataset, self.maxlen, self.clients.neg_num, self.eval_k, self.config['full_eval'], self.device)
                    else:
                        t_test = evaluate (self.model, self.dataset, self.maxlen, self.clients.neg_num, self.eval_k, self.config['full_eval'], self.device)

                    # 检查测试集评估结果是否异常
                    if t_test [0] >= 1.0 or t_test [1] > 1.0 or np.isnan (t_test [0]) or np.isnan (t_test [1]):
                        self.logger.info (f"检测到异常测试结果: NDCG@{self.eval_k}={t_test [0]:.4f}, HR@{self.eval_k}={t_test [1]:.4f}")

                    # 记录到日志
                    self.logger.info (
                        'epoch:%d, time: %f(s), valid (NDCG@%d: %.4f, HR@%d: %.4f), test (NDCG@%d: %.4f, HR@%d: %.4f) all_time: %f(s)'
                        % (epoch + 1, t1, self.eval_k, t_valid [0], self.eval_k, t_valid [1], self.eval_k, t_test [0], self.eval_k, t_test [1], T))
                else:
                    # 跳过测试集评估，设置默认值
                    t_test = (0.0, 0.0)
                    # 记录到日志
                    self.logger.info (
                        'epoch:%d, time: %f(s), valid (NDCG@%d: %.4f, HR@%d: %.4f), test: SKIPPED, all_time: %f(s)'
                        % (epoch + 1, t1, self.eval_k, t_valid [0], self.eval_k, t_valid [1], T))

                # 更新最佳结果
                if not self.skip_test_eval:
                    # 包含测试集评估的情况
                    if t_valid [0] > best_val_ndcg or t_valid [1] > best_val_hr or t_test [0] > best_test_ndcg or \
                            t_test [1] > best_test_hr:
                        best_val_ndcg = max (t_valid [0], best_val_ndcg)
                        best_val_hr = max (t_valid [1], best_val_hr)
                        best_test_ndcg = max (t_test [0], best_test_ndcg)
                        best_test_hr = max (t_test [1], best_test_hr)
                        self.logger.info (
                            f"新的最佳性能: valid NDCG@{self.eval_k}={best_val_ndcg:.4f}, test NDCG@{self.eval_k}={best_test_ndcg:.4f}")

                else:
                    # 跳过测试集评估的情况，只基于验证集更新
                    if t_valid [0] > best_val_ndcg or t_valid [1] > best_val_hr:
                        best_val_ndcg = max (t_valid [0], best_val_ndcg)
                        best_val_hr = max (t_valid [1], best_val_hr)
                        self.logger.info (
                            f"新的最佳性能: valid NDCG@{self.eval_k}={best_val_ndcg:.4f}, valid HR@{self.eval_k}={best_val_hr:.4f}")

                t0 = time.time ()
                self.model.train ()

                # 在评估完成后检查早停标志
                if early_stop_triggered:
                    break

            # 如果早停被触发，跳出训练循环
            if early_stop_triggered:
                break

        # 记录最佳结果
        if not self.skip_test_eval:
            self.logger.info (
                '[联邦训练] 最佳结果: valid NDCG@{}={:.4f}, HR@{}={:.4f}, test NDCG@{}={:.4f}, HR@{}={:.4f}'.format (
                    self.eval_k, best_val_ndcg, self.eval_k, best_val_hr, self.eval_k, best_test_ndcg, self.eval_k, best_test_hr))
        else:
            self.logger.info ('[联邦训练] 最佳结果: valid NDCG@{}={:.4f}, HR@{}={:.4f} (测试集评估已跳过)'.format (
                self.eval_k, best_val_ndcg, self.eval_k, best_val_hr))

