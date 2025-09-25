# --- START OF CORRECTED FILE BERT4Rec.py ---

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    @property
    def weight(self):
        return self.pe.weight

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class BERTEmbedding(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 max_len,
                 dropout=0.1):
        super().__init__()
        self.item_embedding = TokenEmbedding(vocab_size=vocab_size,
                                             embed_size=embed_size)
        self.position_embedding = PositionalEmbedding(max_len=max_len,
                                                      d_model=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.item_embedding(sequence) + self.position_embedding(sequence)
        return self.dropout(x)


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model)
                                            for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [
            layer(x).view(batch_size, -1,
                          self.h, self.d_k).transpose(1, 2)
            for layer, x in zip(self.linear_layers, (query, key, value))
        ]
        x, attn = self.attention(query, key, value,
                                 mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1,
                                                self.h * self.d_k)
        return self.output_linear(x)


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
                                         (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads,
                                              d_model=hidden,
                                              dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden,
                                                    d_ff=feed_forward_hidden,
                                                    dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden,
                                                 dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden,
                                                  dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x,
                                lambda _x: self.attention.forward(_x, _x, _x,
                                                                  mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class BERT(nn.Module):
    def __init__(self, config, item_maxid):
        super().__init__()

        max_len = config['max_seq_len']
        n_layers = config['num_layers']
        heads = config['num_heads']
        hidden = config['hidden_size']
        dropout = config['dropout']

        self.max_seq_length = max_len
        # CORRECTED: vocab_size = item_maxid + 1 (for items) + 1 (for padding 0) + 1 (for [MASK])
        self.vocab_size = item_maxid + 2
        # ADDED: Define mask token id. It's the last token in our vocabulary.
        self.mask_token_id = item_maxid + 1
        self.hidden = hidden
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embedding = BERTEmbedding(vocab_size=self.vocab_size,
                                       embed_size=self.hidden,
                                       max_len=max_len,
                                       dropout=dropout)
        self.item_embedding = self.embedding.item_embedding
        self.position_embedding = self.embedding.position_embedding

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden, heads, hidden * 4, dropout)
            for _ in range(n_layers)
        ])

        # ADDED: Define loss function here, CrossEntropyLoss is standard for MIP
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        self.init_weights()

    def forward(self, x):  # Removed x_lens as it is not used
        mask = (x > 0).unsqueeze(1).unsqueeze(2)
        x = self.embedding(x)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        logits = torch.matmul(x, self.embedding.item_embedding.weight.transpose(0, 1))
        return logits

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                torch.nn.init.xavier_normal_(param.data)
            elif 'bias' in name:
                param.data.zero_()
        self.item_embedding.weight.data[0].fill_(0)
        self.position_embedding.pe.weight.data[0].fill_(0)
        # It's also good practice to initialize the [MASK] embedding
        mask_emb_init = torch.randn(self.hidden)
        self.item_embedding.weight.data[self.mask_token_id] = mask_emb_init

    # CORRECTED: loss_function for Masked Item Prediction
    def loss_function(self, seq_out, labels):
        """
        Calculates the loss for the Masked Item Prediction task.
        :param seq_out: The output of the BERT model, shape: (batch_size, seq_len, vocab_size)
        :param labels: The true item IDs for the masked positions, shape: (batch_size, seq_len)
                       Non-masked positions should have an ignore_index (e.g., 0).
        """
        # Flatten the outputs and labels
        # seq_out.view(-1, self.vocab_size) will have shape (batch_size * seq_len, vocab_size)
        # labels.view(-1) will have shape (batch_size * seq_len)
        loss = self.criterion(seq_out.view(-1, self.vocab_size), labels.view(-1).long())
        return loss

    def log2feats(self, log_seqs):
        if not isinstance(log_seqs, torch.Tensor):
            log_seqs = torch.LongTensor(log_seqs).to(self.device)
        mask = (log_seqs > 0).unsqueeze(1).unsqueeze(2)
        seqs = self.embedding(log_seqs)
        for transformer in self.transformer_blocks:
            seqs = transformer.forward(seqs, mask)
        return seqs

    # CORRECTED: predict method for evaluation using [MASK] token
    def predict(self, user_ids, log_seqs, item_indices):
        """
        Predict scores by appending a [MASK] token to the end of the sequence.
        Handles the case where the sequence is full.
        """
        if not isinstance(log_seqs, torch.Tensor):
            log_seqs = torch.from_numpy(log_seqs).type(torch.long)
        log_seqs = log_seqs.to(self.device)

        # Calculate the actual length of each sequence in the batch
        seq_len = torch.sum(log_seqs != 0, dim=1)

        # Create a new sequence tensor for prediction
        pred_seqs = torch.zeros_like(log_seqs)
        pred_seqs.copy_(log_seqs)

        # For sequences that are full, we need to make space for the [MASK] token.
        # We shift the sequence to the left by one, discarding the oldest item.
        full_seq_mask = (seq_len >= self.max_seq_length)
        if torch.any(full_seq_mask):
            pred_seqs[full_seq_mask, :-1] = log_seqs[full_seq_mask, 1:]

        # The index where we'll place the [MASK] token.
        # For full sequences, it's the last index (maxlen-1).
        # For others, it's the index right after the last real item.
        mask_indices = torch.clamp(seq_len, max=self.max_seq_length - 1)

        # Place the [MASK] token at the correct position for each sequence
        pred_seqs[torch.arange(log_seqs.size(0)), mask_indices] = self.mask_token_id

        # Get the feature representation for the prediction sequence
        log_feats = self.log2feats(pred_seqs)

        # The final feature is the one at the position of the [MASK] token
        final_feat = log_feats[torch.arange(log_seqs.size(0)), mask_indices]

        if not isinstance(item_indices, torch.Tensor):
            item_indices = torch.LongTensor(item_indices).to(self.device)
        item_embs = self.item_embedding(item_indices)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits