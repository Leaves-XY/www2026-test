import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


class FilterLayer (nn.Module):
    def __init__ ( self, hidden_size, max_seq_length, hidden_dropout_prob, layer_norm_eps ):
        super (FilterLayer, self).__init__ ()
        self.complex_weight = nn.Parameter (
            torch.randn (1, max_seq_length // 2 + 1, hidden_size, 2, dtype = torch.float32) * 0.02)
        self.out_dropout = nn.Dropout (hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm (hidden_size, eps = layer_norm_eps)

    def forward ( self, input_tensor ):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft (input_tensor, dim = 1, norm = 'ortho')
        weight = torch.view_as_complex (self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft (x, n = seq_len, dim = 1, norm = 'ortho')
        hidden_states = self.out_dropout (sequence_emb_fft)
        hidden_states = self.LayerNorm (hidden_states + input_tensor)

        return hidden_states


class FeedForward (nn.Module):
    """
    前馈神经网络模块
    Transformer中的FFN层，由两个线性变换和一个非线性激活函数组成
    支持 Pre-LN 和 Post-LN 架构
    """

    def __init__ (
            self,
            hidden_size,  # 隐藏层大小
            inner_size,  # 内部层大小
            hidden_dropout_prob,  # dropout概率
            hidden_act,  # 激活函数类型
            layer_norm_eps,  # Layer Normalization的epsilon值

    ):
        super (FeedForward, self).__init__ ()

        self.dense_1 = nn.Linear (hidden_size, inner_size)  # 第一个线性变换层
        self.intermediate_act_fn = self.get_hidden_act (hidden_act)  # 激活函数

        self.dense_2 = nn.Linear (inner_size, hidden_size)  # 第二个线性变换层
        self.LayerNorm = nn.LayerNorm (hidden_size, eps = layer_norm_eps)  # Layer Normalization
        self.dropout = nn.Dropout (hidden_dropout_prob)  # Dropout层

    def get_hidden_act ( self, act ):
        """
        获取指定的激活函数

        参数:
        - act: 激活函数名称

        返回:
        - 激活函数
        """
        ACT2FN = {
            "gelu": self.gelu,
            "relu": F.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN [act]

    def gelu ( self, x ):
        """
        GELU激活函数的实现
        GELU(x) = x * Φ(x)，其中Φ是标准正态分布的累积分布函数

        参数:
        - x: 输入张量

        返回:
        - 应用GELU后的张量
        """
        return x * 0.5 * (1.0 + torch.erf (x / math.sqrt (2.0)))

    def swish ( self, x ):
        """
        Swish激活函数的实现
        Swish(x) = x * sigmoid(x)

        参数:
        - x: 输入张量

        返回:
        - 应用Swish后的张量
        """
        return x * torch.sigmoid (x)

    def forward ( self, input_tensor ):
        """
        前向传播函数，支持 Pre-LN 和 Post-LN 架构

        参数:
        - input_tensor: 输入张量

        返回:
        - hidden_states: 经过前馈网络处理后的张量
        """

            # Post-LN 架构：直接使用输入
        hidden_states = self.dense_1 (input_tensor)  # 第一个线性变换
        hidden_states = self.intermediate_act_fn (hidden_states)  # 应用激活函数
        hidden_states = self.dense_2 (hidden_states)  # 第二个线性变换
        hidden_states = self.dropout (hidden_states)  # 应用dropout

            # Post-LN 架构：残差连接后进行 Layer Normalization
        hidden_states = self.LayerNorm (hidden_states + input_tensor)

        return hidden_states


class TransformerLayer (nn.Module):
    """
    Transformer层
    包含一个多头注意力子层和一个前馈网络子层
    支持 Pre-LN 和 Post-LN 架构
    """

    def __init__ (
            self,
            hidden_size,  # 隐藏层大小
            max_seq_length,
            intermediate_size,  # 前馈网络内部大小
            hidden_dropout_prob,  # 隐藏层dropout概率
            hidden_act,  # 激活函数类型
            layer_norm_eps,  # Layer Normalization的epsilon值
                ):

        super (TransformerLayer, self).__init__ ()
        self.filter_layer = FilterLayer (hidden_size, max_seq_length, hidden_dropout_prob, layer_norm_eps)
        # 前馈网络子层
        self.feed_forward = FeedForward (
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,

        )

    def forward ( self, hidden_states, attention_mask ):
        """
        前向传播函数

        参数:
        - hidden_states: 输入隐藏状态
        - attention_mask: 注意力掩码

        返回:
        - feedforward_output: 经过完整Transformer层处理后的输出
        """
        # 首先通过多头注意力层
        attention_output = self.filter_layer (hidden_states)
        # 然后通过前馈网络层
        feedforward_output = self.feed_forward (attention_output)
        return feedforward_output


class TransformerEncoder (nn.Module):
    """
    Transformer编码器
    由多个Transformer层堆叠组成
    支持 Pre-LN 和 Post-LN 架构
    """

    def __init__ (
            self,
            n_layers = 2,  # Transformer层数量
            hidden_size = 64,  # 隐藏层大小
            max_seq_length = 50,
            inner_size = 256,  # 前馈网络内部大小
            hidden_dropout_prob = 0.5,  # 隐藏层dropout概率
            hidden_act = "gelu",  # 激活函数类型
            layer_norm_eps = 1e-12,  # Layer Normalization的epsilon值
             ):  # 是否使用Pre-LN架构
        super (TransformerEncoder, self).__init__ ()
        # 创建指定数量的Transformer层
        layer = TransformerLayer (
            hidden_size,
            max_seq_length,
            inner_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        # 使用ModuleList存储多个相同结构的Transformer层
        self.layer = nn.ModuleList ([copy.deepcopy (layer) for _ in range (n_layers)])

    def forward ( self, hidden_states, attention_mask, output_all_encoded_layers = True ):
        """
        前向传播函数

        参数:
        - hidden_states: 输入隐藏状态
        - attention_mask: 注意力掩码
        - output_all_encoded_layers: 是否输出所有编码层的结果

        返回:
        - all_encoder_layers: 所有编码层的输出列表(如果output_all_encoded_layers为True)或最后一层的输出
        """
        all_encoder_layers = []  # 存储每一层的输出
        # 逐层处理输入
        for layer_module in self.layer:
            hidden_states = layer_module (hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append (hidden_states)
        # 如果不需要输出所有层的结果，只保存最后一层的输出
        if not output_all_encoded_layers:
            all_encoder_layers.append (hidden_states)
        return all_encoder_layers


class FMLPRec (nn.Module):
    """
    FMLP 模型类
    """

    def __init__ ( self, config, item_maxid ):
        super (FMLPRec, self).__init__ ()

        # 模型配置参数
        self.n_layers = config ['num_layers']

        self.hidden_size = config['hidden_size']

        self.inner_size = config ['inner_size']

        self.hidden_dropout_prob = config ['dropout']
        self.hidden_act = 'gelu'
        self.layer_norm_eps = 1e-8
        self.max_seq_length = config ['max_seq_len']


        # 设备配置
        self.device = "cuda" if torch.cuda.is_available () else "cpu"

        # 嵌入层 - 调整为与参考项目SAS.torch一致
        self.n_items = item_maxid + 1
        self.item_embedding = nn.Embedding (self.n_items, self.hidden_size, padding_idx = 0)
        # 位置编码使用maxlen+1，与参考项目一致
        self.position_embedding = nn.Embedding (self.max_seq_length + 1, self.hidden_size, padding_idx = 0)

        # Transformer编码器
        self.trm_encoder = TransformerEncoder (
            n_layers = self.n_layers,
            hidden_size = self.hidden_size,
            max_seq_length = self.max_seq_length,
            inner_size = self.inner_size,
            hidden_dropout_prob = self.hidden_dropout_prob,
            hidden_act = self.hidden_act,
            layer_norm_eps = self.layer_norm_eps,

        )

        # LayerNorm和Dropout - 添加embedding dropout与参考项目一致
        self.LayerNorm = nn.LayerNorm (self.hidden_size, eps = self.layer_norm_eps)
        self.dropout = nn.Dropout (self.hidden_dropout_prob)
        self.emb_dropout = nn.Dropout (self.hidden_dropout_prob)  # 对应参考项目的emb_dropout

        # 初始化参数
        self.init_param ()

    def init_param ( self ):
        """初始化模型参数 - 调整为与参考项目SAS.torch一致"""
        # 使用xavier_normal_初始化所有参数，与参考项目一致
        for name, param in self.named_parameters ():
            try:
                nn.init.xavier_normal_ (param.data)
            except:
                pass  # 忽略初始化失败的层

        # 设置padding位置的embedding为0，与参考项目一致
        self.position_embedding.weight.data [0, :] = 0
        self.item_embedding.weight.data [0, :] = 0

    def get_attention_mask ( self, item_seq, bidirectional = False ):
        """
        生成注意力掩码
        item_seq: 物品序列，形状为(batch_size, seq_len)
        bidirectional: 是否使用双向注意力，默认为False(单向/因果注意力)
        """
        attention_mask = item_seq != 0  # 非0位置为True，表示非padding位置
        extended_attention_mask = attention_mask.unsqueeze (1).unsqueeze (2)  # 扩展维度，变为(batch_size, 1, 1, seq_len)
        if not bidirectional:
            # 使用下三角矩阵，实现因果注意力（当前位置只能看到之前的位置）
            extended_attention_mask = torch.tril (
                extended_attention_mask.expand ((-1, -1, item_seq.size (-1), -1))
            )
        # 将True/False转换为0.0/-10000.0，用于在softmax前屏蔽某些位置
        extended_attention_mask = torch.where (extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

    def forward ( self, x, x_lens ):
        """
        定义模型的前向传播过程。

        参数:
        - x: 输入的物品序列 (batch_size, seq_len)。
        - x_lens: 每个序列的实际长度 (batch_size)。

        返回:
        - seq_out: 模型对序列中每个位置的下一个物品的预测得分。
        """
        item_seq = x

        # 确保序列长度不超过模型定义的最大长度
        if item_seq.size (1) > self.max_seq_length:
            item_seq = item_seq [:, -self.max_seq_length:]
            # 如果序列被截断，也需要调整序列长度
            x_lens = torch.clamp (x_lens, max = self.max_seq_length)

        # 生成位置编码 - 确保位置ID不超过max_seq_length
        position_ids = torch.arange (
            item_seq.size (1), dtype = torch.long, device = item_seq.device
        )
        position_ids = position_ids.unsqueeze (0).expand_as (item_seq)

        # 确保position_ids不超出范围
        position_ids = torch.clamp (position_ids, 0, self.max_seq_length - 1)
        position_embedding = self.position_embedding (position_ids)

        # 物品嵌入 - 确保item_seq中的ID不超出范围
        item_seq = torch.clamp (item_seq, 0, self.n_items - 1)
        item_emb = self.item_embedding (item_seq)

        # 输入嵌入 = 物品嵌入 + 位置嵌入
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm (input_emb)
        input_emb = self.dropout (input_emb)

        # 生成注意力掩码
        extended_attention_mask = self.get_attention_mask (item_seq)

        # Transformer编码
        trm_output = self.trm_encoder (
            input_emb, extended_attention_mask, output_all_encoded_layers = True
        )
        output = trm_output [-1]  # 取最后一层的输出

        # 计算每个物品的预测得分
        seq_output = torch.matmul (output, self.item_embedding.weight.transpose (0, 1))

        return seq_output

    def loss_function ( self, seq_out, padding_mask, target, neg, seq_len ):
        """
        计算模型的损失函数，使用二元交叉熵(Binary Cross Entropy)损失
        (已删除TOP1损失功能，简化实现)

        参数:
        - seq_out: 模型的输出得分。
        - padding_mask: 用于忽略填充位置的掩码。
        - target: 目标物品ID (正样本)。
        - neg: 负采样物品序列 (负样本)。
        - seq_len: 序列长度 (未使用，保留接口兼容性)。

        返回:
        - loss: 计算出的损失值。
        """
        # 提取目标物品(正样本)的预测分数
        target_output = torch.gather (seq_out, 2, target.unsqueeze (-1).long ())

        # 使用二元交叉熵损失
        # 处理负样本，可能是多个负样本
        if neg.dim () == 3:
            # 多个负样本的情况
            neg_output = torch.gather (seq_out, 2, neg.long ())

            # 计算正样本损失：-log(sigmoid(pos_score))
            pos_loss = -torch.log (torch.sigmoid (target_output))

            # 计算多个负样本的损失并平均：-log(1-sigmoid(neg_score))
            neg_loss = -torch.log (1 - torch.sigmoid (neg_output)).mean (dim = -1, keepdim = True)
        else:
            # 单个负样本的情况
            neg_output = torch.gather (seq_out, 2, neg.unsqueeze (-1).long ())

            # 计算正样本损失
            pos_loss = -torch.log (torch.sigmoid (target_output))

            # 计算负样本损失
            neg_loss = -torch.log (1 - torch.sigmoid (neg_output))

        # 合并正负样本损失
        loss = pos_loss + neg_loss

        # 应用掩码，确保只计算非填充位置的损失
        loss = loss * padding_mask.unsqueeze (-1)

        # 计算平均损失 (只考虑非padding位置)
        non_zero_elements = padding_mask.sum ()
        if non_zero_elements > 0:
            loss = loss.sum () / non_zero_elements
        else:
            loss = loss.sum ()

        return loss


    def log2feats ( self, log_seqs ):
        """
        将序列转换为特征表示，基于pmixer的log2feats方法

        参数:
        - log_seqs: 输入序列 (batch_size, seq_len)

        返回:
        - log_feats: 序列特征表示 (batch_size, seq_len, hidden_size)
        """
        # 转换为tensor
        if not isinstance (log_seqs, torch.Tensor):
            log_seqs = torch.LongTensor (log_seqs).to (self.device)

        # 物品嵌入
        seqs = self.item_embedding (log_seqs)
        seqs *= self.item_embedding.embedding_dim ** 0.5

        # 位置嵌入 - 调整为与参考项目SAS.torch完全一致的方式
        import numpy as np
        poss = np.tile (np.arange (1, log_seqs.shape [1] + 1), [log_seqs.shape [0], 1])
        poss *= (log_seqs.cpu ().numpy () != 0)  # 只对非padding位置添加位置编码
        seqs += self.position_embedding (torch.LongTensor (poss).to (self.device))
        seqs = self.emb_dropout (seqs)  # 使用embedding dropout，与参考项目一致

        # 生成注意力掩码
        extended_attention_mask = self.get_attention_mask (log_seqs)

        # Transformer编码
        trm_output = self.trm_encoder (
            seqs, extended_attention_mask, output_all_encoded_layers = True
        )


        log_feats = self.LayerNorm (trm_output [-1])  # 保持与原实现一致

        return log_feats


    def predict ( self, user_ids, log_seqs, item_indices ):
        """
        基于pmixer的predict方法，用于推理

        参数:
        - user_ids: 用户ID (batch_size,)
        - log_seqs: 输入序列 (batch_size, seq_len)
        - item_indices: 候选物品ID列表 (num_items,)

        返回:
        - logits: 预测得分 (batch_size, num_items)
        """
        log_feats = self.log2feats (log_seqs)  # user_ids暂时未使用

        final_feat = log_feats [:, -1, :]  # 只使用最后一个位置的特征

        # 转换为tensor
        if not isinstance (item_indices, torch.Tensor):
            item_indices = torch.LongTensor (item_indices).to (self.device)

        item_embs = self.item_embedding (item_indices)  # (num_items, hidden_size)

        logits = item_embs.matmul (final_feat.unsqueeze (-1)).squeeze (-1)  # (batch_size, num_items)

        return logits
