import numpy as np
import torch
import time
from ..dataset import ClientsDataset, evaluate, evaluate_valid
from ..metric import NDCG_binary_at_k_batch, AUC_at_k_batch, HR_at_k_batch
from ..untils import getModel, FedDecorrLoss


class Clients:
    """
    联邦学习客户端类 - 支持Top-k梯度选择
    主要改进：
    1. 支持不同设备类型（小型/中型/大型）及其对应嵌入维度
    2. 支持本地梯度Top-k选择，只传输最重要的k个梯度分量
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

        # Top-k配置
        self.top_k_ratio = config['top_k_ratio']  # Top-k比例，默认保留10%的梯度
        self.top_k_method = config['top_k_method']  # 'global' 或 'layer-wise'
        self.min_k = config['min_k']  # 每层最少保留的梯度数量

        # Top-k统计
        self.top_k_stats = {
            'total_gradients': 0,
            'selected_gradients': 0,
            'compression_ratio': 0.0
        }

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

        # 记录客户端设备类型
        self.device_types = {}  # {uid: 's', 'm', 'l'}

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
        self.logger.info(f"Top-k配置: 比例={self.top_k_ratio}, 方法={self.top_k_method}, 最小k={self.min_k}")
        self.logger.info(f"设备类型分布: "
                         f"小型:{sum(1 for t in self.device_types.values() if t == 's')}, "
                         f"中型:{sum(1 for t in self.device_types.values() if t == 'm')}, "
                         f"大型:{sum(1 for t in self.device_types.values() if t == 'l')}")

    def _init_device_types(self):
        user_set = self.clients_data.get_user_set()
        total_users = len(user_set)

        device_split = self.config['device_split']
        num_s = int(total_users * device_split[0])
        num_m = int(total_users * (device_split[0] + device_split[1]))

        if self.config['assign_by_interactions']:
            method = "按交互次序分配"
            user_interactions = {uid: len(self.user_train[uid]) for uid in user_set}
            sorted_users = sorted(user_set, key=lambda uid: user_interactions[uid])
        else:
            method = "随机分配"
            sorted_users = list(user_set)
            np.random.shuffle(sorted_users)

        for i, uid in enumerate(sorted_users):
            if i < num_s:
                self.device_types[uid] = 's'
            elif i < num_m:
                self.device_types[uid] = 'm'
            else:
                self.device_types[uid] = 'l'

        count_s = sum(1 for t in self.device_types.values() if t == 's')
        count_m = sum(1 for t in self.device_types.values() if t == 'm')
        count_l = sum(1 for t in self.device_types.values() if t == 'l')

        self.logger.info(
            f"嵌套设备类型分配完成({method}): "
            f"小型={count_s}({count_s / total_users:.1%}), "
            f"中型={count_m}({count_m / total_users:.1%}), "
            f"大型={count_l}({count_l / total_users:.1%})"
        )

    def get_dim_for_client(self, uid):
        dev_type = self.device_types.get(uid, 'l')
        return {
            's': self.dim_s,
            'm': self.dim_m,
            'l': self.dim_l
        }[dev_type]

    def load_server_model_params(self, client_model, server_state_dict):
        client_state_dict = client_model.state_dict()

        for name, server_param in server_state_dict.items():
            if name in client_state_dict:
                client_param = client_state_dict[name]
                if server_param.shape != client_param.shape:
                    slicing_indices = [slice(0, dim) for dim in client_param.shape]
                    client_param.copy_(server_param[tuple(slicing_indices)])
                else:
                    client_param.copy_(server_param)

    def apply_top_k_selection(self, gradients_dict):
        """
        对梯度字典应用Top-k选择
        返回压缩后的梯度字典和索引信息
        """
        compressed_gradients = {}
        gradient_indices = {}

        if self.top_k_method == 'global':
            # --- 修正后的全局Top-k逻辑 ---
            # 1. 收集并展平所有梯度，同时记录各层信息
            all_gradients_flat = []
            layer_info = []  # 存储层名称和原始形状
            for name, grad in gradients_dict.items():
                if grad is not None:
                    all_gradients_flat.append(grad.flatten())
                    layer_info.append({'name': name, 'shape': grad.shape})

            if not all_gradients_flat:
                return compressed_gradients, gradient_indices

            # 2. 合并成一个全局梯度向量
            combined_gradients = torch.cat(all_gradients_flat)

            # 3. 计算 Top-k 并获取全局索引
            k = max(self.min_k, int(len(combined_gradients) * self.top_k_ratio))
            
            # 处理k=0的边缘情况
            if k == 0:
                for info in layer_info:
                    compressed_gradients[info['name']] = torch.zeros(info['shape'], device=self.device)
                    gradient_indices[info['name']] = torch.tensor([], dtype=torch.long, device=self.device)
                return compressed_gradients, gradient_indices

            _, top_indices_global = torch.topk(torch.abs(combined_gradients), k)

            # 4. 创建一个稀疏的全局梯度向量，只保留Top-k的值
            sparse_combined_gradients = torch.zeros_like(combined_gradients)
            sparse_combined_gradients[top_indices_global] = combined_gradients[top_indices_global]

            # 5. 将稀疏向量切片并重建回各层的原始形状
            start_idx = 0
            for i, info in enumerate(layer_info):
                grad_size = all_gradients_flat[i].numel()
                
                # 从稀疏全局向量中提取当前层的部分
                layer_grad_sparse_flat = sparse_combined_gradients[start_idx : start_idx + grad_size]
                
                # 重塑为原始形状
                compressed_gradients[info['name']] = layer_grad_sparse_flat.reshape(info['shape'])
                
                # 找到非零元素的局部索引并展平
                gradient_indices[info['name']] = torch.nonzero(layer_grad_sparse_flat, as_tuple=False).flatten()

                start_idx += grad_size

        else:  # layer-wise (逐层选择)
            for name, grad in gradients_dict.items():
                if grad is not None:
                    grad_flat = grad.flatten()
                    num_elements = grad_flat.numel()

                    if num_elements == 0:
                        k = 0
                    else:
                        k = max(self.min_k, int(num_elements * self.top_k_ratio))
                        k = min(k, num_elements) # 确保k不超过元素总数

                    if k == 0:
                        compressed_grad = torch.zeros_like(grad_flat)
                        top_indices = torch.tensor([], dtype=torch.long, device=self.device)
                    else:
                        _, top_indices = torch.topk(torch.abs(grad_flat), k)
                        compressed_grad = torch.zeros_like(grad_flat)
                        compressed_grad[top_indices] = grad_flat[top_indices]

                    compressed_gradients[name] = compressed_grad.reshape(grad.shape)
                    gradient_indices[name] = top_indices

        return compressed_gradients, gradient_indices

    def sync_models_from_server(self, server):
        """从服务器同步蒸馏后的模型参数"""
        # self.logger.info("同步服务器蒸馏后的模型参数...")

        # 小模型同步
        for param, server_param in zip(self.model_s.parameters(), server.model_s.parameters()):
            param.data.copy_(server_param.data)

        # 中模型同步
        for param, server_param in zip(self.model_m.parameters(), server.model_m.parameters()):
            param.data.copy_(server_param.data)

        # 大模型同步
        for param, server_param in zip(self.model.parameters(), server.model.parameters()):
            param.data.copy_(server_param.data)

    def train(self, uids, model_param_state_dict, epoch=0):
        self.sync_models_from_server(self)  # 同步模型
        clients_grads = {}
        clients_losses = {}
        clients_indices = {}  # 存储每个客户端的梯度索引信息

        for uid in uids:
            uid = uid.item()
            dev_type = self.device_types.get(uid, 'l')

            if dev_type == 's':
                client_model = self.model_s
            elif dev_type == 'm':
                client_model = self.model_m
            else:  # 'l'
                client_model = self.model_l

            client_model.train()
            alpha = self.config['decor_alpha']

            self.load_server_model_params(client_model, model_param_state_dict)

            optimizer = torch.optim.Adam(client_model.parameters(), lr=self.config['lr'],
                                         betas=(0.9, 0.98), weight_decay=self.config['l2_reg'])

            input_seq = self.clients_data.train_seq[uid]
            target_seq = self.clients_data.valid_seq[uid]
            input_len = self.clients_data.seq_len[uid]
            cand = np.setdiff1d(self.clients_data.item_set, self.clients_data.seq[uid])
            prob = self.clients_data.item_prob[cand]
            prob = prob / prob.sum()
            neg_seq = np.random.choice(cand, (input_len, 100), p=prob)
            neg_seq = np.pad(neg_seq, ((input_seq.shape[0] - input_len, 0), (0, 0)))
            input_seq = torch.from_numpy(input_seq).unsqueeze(0).to(self.device)
            target_seq = torch.from_numpy(target_seq).unsqueeze(0).to(self.device)
            neg_seq = torch.from_numpy(neg_seq).unsqueeze(0).to(self.device)
            input_len = torch.tensor(input_len).unsqueeze(0).to(self.device)
            max_seq_length = client_model.max_seq_length
            if input_seq.size(1) > max_seq_length:
                input_seq = input_seq[:, -max_seq_length:]
                target_seq = target_seq[:, -max_seq_length:]
                neg_seq = neg_seq[:, -max_seq_length:, :]
                input_len = torch.clamp(input_len, max=max_seq_length)

            seq_out = client_model(input_seq, input_len)
            padding_mask = (torch.not_equal(input_seq, 0)).float().unsqueeze(-1).to(self.device)

            feddecorr = FedDecorrLoss ()

            # 计算损失
            # 根据设备类型选择损失函数
            if dev_type == 's':
                # 小型设备：只计算小型模型损失（使用完整用户特征）
                loss = self.model_s.loss_function(seq_out, padding_mask, target_seq, neg_seq, input_len)
            elif dev_type == 'm':
                # 中型设备：
                # 1. 计算小型模型部分损失（使用小型特征子集 Ns）
                loss_s = self.model_s.loss_function (seq_out, padding_mask, target_seq, neg_seq, input_len)
                # 2. 计算中型模型整体损失（使用完整用户特征）
                loss_m = self.model_m.loss_function (seq_out, padding_mask, target_seq, neg_seq, input_len)
                # 添加正则化损失（仅对高维部分）
                loss_reg = feddecorr(client_model.item_embedding.weight[:, self.dim_s:])
                loss = loss_s + loss_m + alpha * loss_reg
            else:  # 'l'
                # 大型设备：
                # 1. 计算小型模型部分损失（使用小型特征子集 Ns）
                loss_s = self.model_s.loss_function (seq_out, padding_mask, target_seq, neg_seq, input_len)
                # 2. 计算中型模型整体损失（使用完整用户特征）
                loss_m = self.model_m.loss_function (seq_out, padding_mask, target_seq, neg_seq, input_len)
                # 3. 计算大型模型整体损失（使用完整用户特征）
                loss_l = self.model_l.loss_function (seq_out, padding_mask, target_seq, neg_seq, input_len)
                # 添加正则化损失（仅对高维部分）

                loss_reg = feddecorr (client_model.item_embedding.weight [:, self.dim_s:])
                loss =loss_l + loss_s + loss_m + alpha * loss_reg

            clients_losses[uid] = loss.item()

            optimizer.zero_grad()
            loss.backward()

            # 收集所有梯度
            gradients = {name: param.grad.clone() for name, param in client_model.named_parameters() if
                         param.grad is not None}

            # 应用Top-k选择
            compressed_gradients, gradient_indices = self.apply_top_k_selection(gradients)

            # 更新模型参数为压缩后的梯度
            for name, param in client_model.named_parameters():
                if name in compressed_gradients:
                    param.grad = compressed_gradients[name]

            optimizer.step()

            # 存储压缩后的梯度和索引信息
            clients_grads[uid] = compressed_gradients
            clients_indices[uid] = gradient_indices

        # 更新Top-k统计信息
        total_gradients = sum(len(grad.flatten()) for grads in clients_grads.values() for grad in grads.values())
        selected_gradients = sum(
            len(indices) for indices_list in clients_indices.values() for indices in indices_list.values())

        self.top_k_stats['total_gradients'] = total_gradients
        self.top_k_stats['selected_gradients'] = selected_gradients
        if total_gradients > 0:
            self.top_k_stats['compression_ratio'] = selected_gradients / total_gradients

        return clients_grads, clients_losses, clients_indices


class Server:
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
        self.dim_s = config['dim_s']
        self.dim_m = config['dim_m']
        self.dim_l = config['dim_l']
        self.eval_k = config['eval_k']

        config['hidden_size'] = self.dim_l
        # 添加知识蒸馏相关参数
        self.kd_ratio = config['kd_ratio']  # 蒸馏物品采样比例
        self.kd_lr = config['kd_lr']  # 蒸馏学习率
        self.distill_epochs = config['distill_epochs']  # 蒸馏轮次

        # 修正1：创建正确的配置对象
        # 创建小模型配置
        config_s = config.copy()
        config_s['hidden_size'] = self.dim_s
        # 创建中模型配置
        config_m = config.copy()
        config_m['hidden_size'] = self.dim_m
        # 创建大模型配置（使用全局模型配置）
        config_l = config.copy()
        config_l['hidden_size'] = self.dim_l

        # 修正2：正确初始化三种尺寸的模型
        self.model_s = getModel(config_s, self.clients.clients_data.get_maxid())
        self.model_m = getModel(config_m, self.clients.clients_data.get_maxid())
        self.model_l = getModel(config_l, self.clients.clients_data.get_maxid())

        # 修正3：将模型转移到设备并初始化参数
        for model in [self.model_s, self.model_m, self.model_l]:
            model.to(self.device)
            # 初始化模型参数
            for name, param in model.named_parameters():
                try:
                    torch.nn.init.xavier_normal_(param.data)
                except:
                    pass  # 忽略初始化失败的层

        # 修正4：确保全局模型与蒸馏大模型一致
        self.model = self.model_l  # 让全局模型引用蒸馏大模型
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=config['lr'],
                                          betas=(0.9, 0.98),
                                          weight_decay=config['l2_reg'])

        # 创建蒸馏优化器，包含所有三个模型的参数
        kd_params = list(self.model_s.parameters()) + list(self.model_m.parameters()) + list(self.model_l.parameters())
        self.kd_optimizer = torch.optim.Adam(kd_params, lr=self.kd_lr)

        logger.info("初始化知识蒸馏模块完成")
        logger.info("服务器初始化完成")

        # --- 通信开销计算模块初始化 ---
        self._init_comm_cost_calculation()

        # 添加累积通信开销变量
        self.cumulative_uplink_cost = 0.0
        self.cumulative_downlink_cost = 0.0

    def _knowledge_distillation(self):
        """执行基于关系的集成自知识蒸馏 - 修复尺寸不匹配问题"""
        self.logger.info("开始知识蒸馏步骤...")

        # 1. 准备三个模型
        models = {
            's': self.model_s,
            'm': self.model_m,
            'l': self.model_l
        }

        # 2. 随机选择蒸馏物品子集
        item_set = self.clients.clients_data.item_set
        kd_items = self._select_distill_items(item_set, self.kd_ratio)
        kd_size = len(kd_items)
        self.logger.info(f"蒸馏物品子集大小: {kd_size}个物品")

        # 3. 将物品ID转换为嵌入索引
        # 确保索引在有效范围内
        max_index = self.clients.itemnum - 1
        kd_index = torch.from_numpy(np.clip(kd_items - 1, 0, max_index)).long().to(self.device)

        # 4. 计算每个模型的物品嵌入 - 只针对蒸馏物品子集
        similarity_matrices = {}
        with torch.no_grad():
            for model_type, model in models.items():
                # 只计算蒸馏物品子集的嵌入
                embeddings = model.item_embedding(kd_index)

                # 归一化处理
                norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
                norm_emb = embeddings / norms

                # 计算物品间的余弦相似度
                sim_matrix = torch.mm(norm_emb, norm_emb.t())
                similarity_matrices[model_type] = sim_matrix

        # 5. 计算集成相似度 - 确保所有矩阵尺寸一致
        ens_matrix = (similarity_matrices['s'] +
                      similarity_matrices['m'] +
                      similarity_matrices['l']) / 3.0

        # 6. 蒸馏训练
        for epoch_idx in range(self.distill_epochs):
            self.kd_optimizer.zero_grad()

            # 重新计算当前模型的相似度
            current_similarities = {}
            for model_type, model in models.items():
                embeddings = model.item_embedding(kd_index)
                norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
                norm_emb = embeddings / norms
                sim_matrix = torch.mm(norm_emb, norm_emb.t())
                current_similarities[model_type] = sim_matrix

            # 计算蒸馏损失 - 使用MSE
            loss = 0
            for model_type in models:
                loss += torch.nn.functional.mse_loss(
                    current_similarities[model_type],
                    ens_matrix.detach()
                )

            # 反向传播
            loss.backward()
            self.kd_optimizer.step()

            self.logger.info(f"蒸馏轮次 [{epoch_idx + 1}/{self.distill_epochs}] 损失: {loss.item():.4f}")

        # 7. 更新客户端模型
        self._update_client_models()

        self.logger.info("知识蒸馏步骤完成")

    def _select_distill_items ( self, item_set, ratio ):
        """
        从全体物品 item_set 中按“交互频率”加权，采样蒸馏物品子集。
        - 忽略 padding(0)
        - 权重全零时回退到均匀采样
        """
        dataset = self.clients.clients_data

        # 1) 统计频率（过滤 padding=0）
        freq = {}
        for user_id, items_list in dataset.train_seq.items ():
            for it in items_list:
                if it == 0:
                    continue
                freq [it] = freq.get (it, 0) + 1

        # 2) 去掉 0，并且只在 item_set 内部采样
        valid_items = [it for it in item_set if it != 0]

        # 3) 构造采样权重；未知物品给个最小权重 1
        weights = np.array ([freq.get (it, 1) for it in valid_items], dtype = np.float32)
        s = weights.sum ()

        # 4) 计算采样规模
        kd_size = max (1, int (len (valid_items) * ratio))

        if s <= 0 or not np.isfinite (s):
            # 回退：均匀采样
            return np.random.choice (valid_items, kd_size, replace = False)

        weights /= s
        return np.random.choice (valid_items, kd_size, replace = False, p = weights)

    def _update_client_models(self):
        """将蒸馏后的模型参数同步到客户端"""
        # 小模型更新
        client_s = self.clients.model_s
        for param, server_param in zip(client_s.parameters(), self.model_s.parameters()):
            param.data.copy_(server_param.data)

        # 中模型更新
        client_m = self.clients.model_m
        for param, server_param in zip(client_m.parameters(), self.model_m.parameters()):
            param.data.copy_(server_param.data)

        # 大模型更新（全局模型）
        client_l = self.clients.model
        for param, server_param in zip(client_l.parameters(), self.model_l.parameters()):
            param.data.copy_(server_param.data)

    def _init_comm_cost_calculation(self):
        """预计算并存储模型尺寸，用于开销统计"""
        self.logger.info("初始化通信开销计算器...")
        # 预计算三种尺寸模型的总参数大小 (MB)
        self.size_p_s = self._calculate_dense_params_size(self.clients.model_s.state_dict())
        self.size_p_m = self._calculate_dense_params_size(self.clients.model_m.state_dict())
        self.size_p_l = self._calculate_dense_params_size(self.clients.model_l.state_dict())

        self.logger.info(f"预计算模型尺寸 (MB): "
                         f"S={self.size_p_s:.4f}, "
                         f"M={self.size_p_m:.4f}, "
                         f"L={self.size_p_l:.4f}")

    def aggregate_gradients(self, clients_grads, clients_indices):
        clients_num = len(clients_grads)
        if clients_num == 0:
            self.logger.info("没有收到任何客户端梯度")
            return

        aggregated_gradients = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if
                                param.requires_grad}

        for uid, grads_dict in clients_grads.items():
            indices_dict = clients_indices.get(uid, {})

            for name, grad in grads_dict.items():
                if grad is None:
                    continue

                target_shape = aggregated_gradients[name].shape

                if grad.shape != target_shape:
                    padded_grad = torch.zeros(target_shape, device=self.device)
                    slicing_indices = tuple(slice(0, dim) for dim in grad.shape)
                    padded_grad[slicing_indices] = grad
                    aggregated_gradients[name] += padded_grad
                else:
                    aggregated_gradients[name] += grad

        for name in aggregated_gradients:
            aggregated_gradients[name] /= clients_num

        for name, param in self.model.named_parameters():
            if name in aggregated_gradients:
                param.grad = aggregated_gradients[name]
            else:
                param.grad = None

    def _calculate_dense_params_size(self, params_dict):
        """计算密集参数字典的总大小（MB）"""
        total_size = 0
        for param in params_dict.values():
            if isinstance(param, torch.Tensor):
                total_size += param.nelement() * param.element_size()
        return total_size / (1024 * 1024)  # 返回MB

    def _calculate_sparse_params_size(self, indices_dict):
        """计算稀疏梯度（值+索引）的总大小（MB）"""
        total_elements = 0
        for indices in indices_dict.values():
            total_elements += len(indices)
        # 假设传输每个值(float32=4字节)和其对应的索引(int64=8字节)
        # 实际传输中索引可能会被压缩，但这里作为一种上界估计
        total_size = total_elements * (4 + 8)
        return total_size / (1024 * 1024)  # 返回MB

    def train(self):
        self.clients.sync_models_from_server(self)  # 传递服务器实例
        user_set = self.clients.clients_data.get_user_set()
        uid_seq = []
        for i in range(0, len(user_set), self.batch_size):
            batch_uids = user_set[i:i + self.batch_size]
            uid_seq.append(torch.tensor(batch_uids))

        best_val_ndcg, best_val_hr = 0.0, 0.0
        best_test_ndcg, best_test_hr = 0.0, 0.0

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

        no_improve_count = 0
        early_stop_triggered = False  # 添加早停标志
        distill_freq=self.config['distill_freq']
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self.model.train()

            # --- 通信开销计算初始化 ---
            total_uplink_cost = 0
            total_downlink_cost = 0
            # -------------------------

            batch_count = 0
            epoch_losses = []

            # 获取要发送的全局模型参数
            model_state_to_send = self.model.state_dict()

            for uids in uid_seq:
                self.optimizer.zero_grad()

                # --- 计算下行链路开销 ---
                # 根据每个客户端的实际模型尺寸计算下行开销
                downlink_cost_batch = 0
                for uid in uids:
                    dev_type = self.clients.device_types.get(uid.item(), 'l')
                    if dev_type == 's':
                        downlink_cost_batch += self.size_p_s
                    elif dev_type == 'm':
                        downlink_cost_batch += self.size_p_m
                    else:  # 'l'
                        downlink_cost_batch += self.size_p_l
                total_downlink_cost += downlink_cost_batch
                # -------------------------

                clients_grads, clients_losses, clients_indices = self.clients.train(uids, model_state_to_send, epoch)

                # --- 计算上行链路开销 ---
                # 客户端上传稀疏梯度（值+索引）
                for client_uid, indices_dict in clients_indices.items():
                    total_uplink_cost += self._calculate_sparse_params_size(indices_dict)
                # -------------------------

                batch_losses = list(clients_losses.values())
                epoch_losses.extend(batch_losses)

                self.aggregate_gradients(clients_grads, clients_indices)

                self.optimizer.step()

                batch_count += 1

            if (epoch + 1) % distill_freq == 0:
                self._knowledge_distillation()
            # --- 更新累积通信开销 ---
            self.cumulative_uplink_cost += total_uplink_cost
            self.cumulative_downlink_cost += total_downlink_cost

            # --- 记录通信开销（当前epoch和累积总量） ---

            self.logger.info(f"累积通信开销: "
                             f"上行 = {self.cumulative_uplink_cost:.4f} MB, "
                             f"下行 = {self.cumulative_downlink_cost:.4f} MB, "
                             f"总计 = {self.cumulative_uplink_cost + self.cumulative_downlink_cost:.4f} MB")
            # --------------------

            eval_freq = self.eval_freq
            should_evaluate = (epoch + 1) % eval_freq == 0 or epoch == 0 or epoch == self.epochs - 1
            if should_evaluate:
                self.model.eval()
                # t1是该轮训练时间 T为总时间
                t1 = time.time() - t0
                T += t1
                print('Evaluating', end='')

                t_valid = evaluate_valid(self.model, self.dataset, self.maxlen, self.clients.neg_num, self.eval_k,
                                         self.config['full_eval'], self.device)
                self.logger.info(
                    f"传统评估结果 - Epoch {epoch + 1}: NDCG@{self.eval_k}={t_valid[0]:.4f}, HR@{self.eval_k}={t_valid[1]:.4f}")

                # 输出Top-k统计信息
                stats = self.clients.top_k_stats
                if stats['total_gradients'] > 0:
                    compression_ratio = stats['compression_ratio']
                    self.logger.info(f"Top-k压缩统计: 总梯度数={stats['total_gradients']}, "
                                     f"选择梯度数={stats['selected_gradients']}, "
                                     f"压缩比例={compression_ratio:.2%}")

                # 检查评估结果是否异常
                if t_valid[0] > 1.0 or t_valid[1] > 1.0 or np.isnan(t_valid[0]) or np.isnan(t_valid[1]):
                    self.logger.info(
                        f"检测到异常评估结果: NDCG@{self.eval_k}={t_valid[0]:.4f}, HR@{self.eval_k}={t_valid[1]:.4f}")

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
                    t_test = evaluate(self.model, self.dataset, self.maxlen, self.clients.neg_num, self.eval_k,
                                      self.config['full_eval'], self.device)

                    # 检查测试集评估结果是否异常
                    if t_test[0] > 1.0 or t_test[1] > 1.0 or np.isnan(t_test[0]) or np.isnan(t_test[1]):
                        self.logger.info(
                            f"检测到异常测试结果: NDCG@{self.eval_k}={t_test[0]:.4f}, HR@{self.eval_k}={t_test[1]:.4f}")

                    # 记录到日志
                    self.logger.info(
                        'epoch:%d, time: %f(s), valid (NDCG@%d: %.4f, HR@%d: %.4f), test (NDCG@%d: %.4f, HR@%d: %.4f) all_time: %f(s)'
                        % (epoch + 1, t1, self.eval_k, t_valid[0], self.eval_k, t_valid[1], self.eval_k, t_test[0],
                           self.eval_k, t_test[1], T))
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
                    self.eval_k, best_val_ndcg, self.eval_k, best_val_hr, self.eval_k, best_test_ndcg, self.eval_k,
                    best_test_hr))
        else:
            self.logger.info('[联邦训练] 最佳结果: valid NDCG@{}={:.4f}, HR@{}={:.4f} (测试集评估已跳过)'.format(
                self.eval_k, best_val_ndcg, self.eval_k, best_val_hr))