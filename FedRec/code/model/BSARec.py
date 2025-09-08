
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制模块
    该模块实现了Transformer架构中的核心组件，允许模型在不同的表示子空间中关注序列的不同部分
    支持 Pre-LN 和 Post-LN 架构
    """

    def __init__(
            self,
            n_heads,  # 注意力头的数量
            hidden_size,  # 隐藏层维度
            hidden_dropout_prob,  # 隐藏层dropout概率
            attn_dropout_prob,  # 注意力dropout概率
            layer_norm_eps,  # Layer Normalization的epsilon值
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads  # 注意力头数量
        self.attention_head_size = int(hidden_size / n_heads)  # 每个注意力头的维度大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 所有注意力头的总维度
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)  # 用于缩放注意力分数

        # 定义查询、键、值的线性变换
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)  # 用于将注意力分数转换为概率分布
        self.attn_dropout = nn.Dropout(attn_dropout_prob)  # 注意力dropout层

        # 输出投影层和Layer Normalization
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        """
        重塑张量为多头注意力格式

        参数:
        - x: 输入张量

        返回:
        - 重塑后的张量，适合多头注意力计算
        """
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        """
        前向传播函数，支持 Pre-LN 和 Post-LN 架构

        参数:
        - input_tensor: 输入张量，形状为 [batch_size, seq_len, hidden_size]
        - attention_mask: 注意力掩码，用于屏蔽某些位置的注意力

        返回:
        - hidden_states: 经过多头注意力处理后的张量
        """

        # Post-LN 架构：直接使用输入
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        # 重塑张量并调整维度顺序以便进行批量矩阵乘法
        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # 计算注意力分数（查询和键的点积）
        attention_scores = torch.matmul(query_layer, key_layer)

        # 缩放注意力分数
        attention_scores = attention_scores / self.sqrt_attention_head_size
        # 应用注意力掩码
        attention_scores = attention_scores + attention_mask

        # 将分数归一化为概率分布
        attention_probs = self.softmax(attention_scores)
        # 应用dropout - 这会随机丢弃整个token的注意力信息
        attention_probs = self.attn_dropout(attention_probs)

        # 计算加权值向量
        context_layer = torch.matmul(attention_probs, value_layer)
        # 调整维度顺序，并合并多头结果
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 输出投影和dropout
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)

        # Post-LN 架构：残差连接后进行 Layer Normalization
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FeedForward(nn.Module):
    """
    前馈神经网络模块
    Transformer中的FFN层，由两个线性变换和一个非线性激活函数组成
    支持 Pre-LN 和 Post-LN 架构
    """

    def __init__(
            self,
            hidden_size,  # 隐藏层大小
            inner_size,  # 内部层大小
            hidden_dropout_prob,  # dropout概率
            hidden_act,  # 激活函数类型
            layer_norm_eps,  # Layer Normalization的epsilon值
    ):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)  # 第一个线性变换层
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)  # 激活函数

        self.dense_2 = nn.Linear(inner_size, hidden_size)  # 第二个线性变换层
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)  # Layer Normalization
        self.dropout = nn.Dropout(hidden_dropout_prob)  # Dropout层

    def get_hidden_act(self, act):
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
        return ACT2FN[act]

    def gelu(self, x):
        """
        GELU激活函数的实现
        GELU(x) = x * Φ(x)，其中Φ是标准正态分布的累积分布函数

        参数:
        - x: 输入张量

        返回:
        - 应用GELU后的张量
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        """
        Swish激活函数的实现
        Swish(x) = x * sigmoid(x)

        参数:
        - x: 输入张量

        返回:
        - 应用Swish后的张量
        """
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        """
        前向传播函数，支持 Pre-LN 和 Post-LN 架构

        参数:
        - input_tensor: 输入张量

        返回:
        - hidden_states: 经过前馈网络处理后的张量
        """

            # Post-LN 架构：直接使用输入
        hidden_states = self.dense_1(input_tensor)  # 第一个线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)  # 应用激活函数
        hidden_states = self.dense_2(hidden_states)  # 第二个线性变换
        hidden_states = self.dropout(hidden_states)  # 应用dropout


        # Post-LN 架构：残差连接后进行 Layer Normalization
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FrequencyLayer(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, layer_norm_eps, c):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.c = c // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        low_pass = x[:]
        low_pass[:, self.c:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass + (self.sqrt_beta ** 2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class BSARecLayer(nn.Module):
    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps, c, alpha,
          ):
        super(BSARecLayer, self).__init__()
        self.filter_layer = FrequencyLayer(hidden_size, hidden_dropout_prob, layer_norm_eps, c)
        self.attention_layer = MultiHeadAttention(n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob,
                                                  layer_norm_eps)
        self.alpha = alpha

    def forward(self, input_tensor, attention_mask):
        dsp = self.filter_layer(input_tensor)
        gsp = self.attention_layer(input_tensor, attention_mask)
        hidden_states = self.alpha * dsp + (1 - self.alpha) * gsp

        return hidden_states


class BSARecBlock(nn.Module):
    def __init__(self, n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
                 layer_norm_eps, c, alpha):
        super(BSARecBlock, self).__init__()
        self.layer = BSARecLayer(n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps, c,
                                 alpha)
        self.feed_forward = FeedForward(hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps,
                                        )

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output


class BSARecEncoder(nn.Module):
    def __init__(self, n_layers=2, n_heads=2, hidden_size=64, inner_size=256, hidden_dropout_prob=0.5,
                 attn_dropout_prob=0.5, hidden_act="gelu", layer_norm_eps=1e-12, c=10, alpha=0.5):
        super(BSARecEncoder, self).__init__()
        block = BSARecBlock(n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
                            layer_norm_eps, c, alpha)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BSARec(nn.Module):
    """
    BSARec Model
    """

    def __init__(self, config, item_maxid):
        super(BSARec, self).__init__()

        # config
        self.n_layers = config['num_layers']
        self.n_heads = config['num_heads']
        self.hidden_size = config['hidden_size']
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['dropout']
        self.attn_dropout_prob = config['dropout']
        self.hidden_act = 'gelu'
        self.layer_norm_eps = 1e-8
        self.max_seq_length = config['max_seq_len']
        self.c = config['c']
        self.alpha = config['alpha']

        # device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # embedding
        self.n_items = item_maxid + 1
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size, padding_idx=0)

        # encoder
        self.encoder = BSARecEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            c=self.c,
            alpha=self.alpha,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.emb_dropout = nn.Dropout(self.hidden_dropout_prob)

        self.init_param()

    def init_param(self):
        for name, param in self.named_parameters():
            try:
                nn.init.xavier_normal_(param.data)
            except:
                pass
        self.position_embedding.weight.data[0, :] = 0
        self.item_embedding.weight.data[0, :] = 0

    def get_attention_mask(self, item_seq, bidirectional=False):
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

    def forward(self, x, x_lens):
        item_seq = x
        if item_seq.size(1) > self.max_seq_length:
            item_seq = item_seq[:, -self.max_seq_length:]
            x_lens = torch.clamp(x_lens, max=self.max_seq_length)

        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_ids = torch.clamp(position_ids, 0, self.max_seq_length - 1)
        position_embedding = self.position_embedding(position_ids)

        item_seq = torch.clamp(item_seq, 0, self.n_items - 1)
        item_emb = self.item_embedding(item_seq)

        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        seq_output = torch.matmul(output, self.item_embedding.weight.transpose(0, 1))
        return seq_output


    def loss_function(self, seq_out, padding_mask, target, neg, seq_len):
        target_output = torch.gather(seq_out, 2, target.unsqueeze(-1).long())
        if neg.dim() == 3:
            neg_output = torch.gather(seq_out, 2, neg.long())
            pos_loss = -torch.log(torch.sigmoid(target_output))
            neg_loss = -torch.log(1 - torch.sigmoid(neg_output)).mean(dim=-1, keepdim=True)
        else:
            neg_output = torch.gather(seq_out, 2, neg.unsqueeze(-1).long())
            pos_loss = -torch.log(torch.sigmoid(target_output))
            neg_loss = -torch.log(1 - torch.sigmoid(neg_output))

        loss = pos_loss + neg_loss
        loss = loss * padding_mask.unsqueeze(-1)
        non_zero_elements = padding_mask.sum()
        if non_zero_elements > 0:
            loss = loss.sum() / non_zero_elements
        else:
            loss = loss.sum()
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
        seqs *= self.hidden_size ** 0.5

        # 位置嵌入 - 调整为与参考项目SAS.torch完全一致的方式
        import numpy as np
        poss = np.tile (np.arange (1, log_seqs.shape [1] + 1), [log_seqs.shape [0], 1])
        poss *= (log_seqs.cpu ().numpy () != 0)  # 只对非padding位置添加位置编码
        seqs += self.position_embedding (torch.LongTensor (poss).to (self.device))
        seqs = self.emb_dropout (seqs)  # 使用embedding dropout，与参考项目一致

        # 生成注意力掩码
        extended_attention_mask = self.get_attention_mask (log_seqs)

        # Transformer编码
        trm_output = self.encoder (
            seqs, extended_attention_mask, output_all_encoded_layers = True
        )


        # Post-LN 架构：每层已经应用了 LayerNorm，这里可以选择是否再次应用
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
