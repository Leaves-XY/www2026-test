# --- START OF MODIFIED FILE BERT4Rec.py ---

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self,
                 vocab_size,
                 embed_size,
                 max_len,
                 dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size,
                                    embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len,
                                            d_model=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        # MODIFIED: Removed genres embedding
        x = self.token(sequence) + self.position(sequence)
        return self.dropout(x)


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

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
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model)
                                            for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            layer(x).view(batch_size, -1,
                          self.h, self.d_k).transpose(1, 2)
            for layer, x in zip(self.linear_layers, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value,
                                 mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1,
                                                self.h * self.d_k)

        return self.output_linear(x)


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
                                         (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

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
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

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
    # MODIFIED: Changed constructor signature to be compatible with the framework
    def __init__(self, config, item_maxid):
        super().__init__()

        # Parse parameters from config dictionary
        max_len = config['max_seq_len']
        n_layers = config['num_layers']
        heads = config['num_heads']
        hidden = config['hidden_size']
        dropout = config['dropout']

        # Set vocab_size based on item_maxid
        self.max_seq_length = max_len
        self.vocab_size = item_maxid + 2  # +1 for padding, +1 for mask token (if any, good practice)
        self.hidden = hidden
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=self.vocab_size,
                                       embed_size=self.hidden,
                                       max_len=max_len,
                                       dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden, heads, hidden * 4, dropout)
            for _ in range(n_layers)
        ])

        # MODIFIED: No separate output layer, will use weight tying with token embedding
        # self.out = nn.Linear(hidden, num_items + 1)
        self.init_weights()

    # MODIFIED: Changed forward signature to be compatible
    def forward(self, x, x_lens=None):
        # Attention mask for BERT is bidirectional, masking only padding tokens (0)
        # The mask shape should be (batch_size, 1, 1, seq_len) for MultiHeadedAttention
        mask = (x > 0).unsqueeze(1).unsqueeze(2)

        # embedding the indexed sequence to sequence of vectors
        # MODIFIED: Removed genres argument
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        # MODIFIED: Use weight tying for output layer
        # This computes the dot product between transformer output and all item embeddings
        logits = torch.matmul(x, self.embedding.token.weight.transpose(0, 1))

        return logits

    def init_weights(self):
        # Initialize parameters
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                torch.nn.init.xavier_normal_(param.data)
            elif 'bias' in name:
                param.data.zero_()
        # Set embedding for padding_idx to zero
        self.embedding.token.weight.data[0].fill_(0)
        self.embedding.position.pe.weight.data[0].fill_(0)

    # ADDED: loss_function method copied from SASRec for compatibility
    def loss_function(self, seq_out, padding_mask, target, neg, seq_len):
        # Extract target item (positive sample) scores
        target_output = torch.gather(seq_out, 2, target.unsqueeze(-1).long())

        # BPR-like loss with binary cross-entropy
        if neg.dim() == 3:
            # For multiple negative samples
            neg_output = torch.gather(seq_out, 2, neg.long())
            pos_loss = -torch.log(torch.sigmoid(target_output))
            neg_loss = -torch.log(1 - torch.sigmoid(neg_output)).mean(dim=-1, keepdim=True)
        else:
            # For a single negative sample
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

    # ADDED: log2feats method, similar to SASRec, for prediction
    def log2feats(self, log_seqs):
        """
        Converts sequence to feature representation for prediction.
        """
        if not isinstance(log_seqs, torch.Tensor):
            log_seqs = torch.LongTensor(log_seqs).to(self.device)

        # Create attention mask
        mask = (log_seqs > 0).unsqueeze(1).unsqueeze(2)

        # Get embeddings
        seqs = self.embedding(log_seqs)

        # Pass through transformer blocks
        for transformer in self.transformer_blocks:
            seqs = transformer.forward(seqs, mask)

        return seqs

    # ADDED: predict method for evaluation, adapted from SASRec
    def predict(self, user_ids, log_seqs, item_indices):
        """
        Predict scores for candidate items during evaluation.
        """
        log_feats = self.log2feats(log_seqs)

        # Use the representation of the last item in the sequence
        final_feat = log_feats[:, -1, :]

        if not isinstance(item_indices, torch.Tensor):
            item_indices = torch.LongTensor(item_indices).to(self.device)

        # Get embeddings of candidate items
        item_embs = self.embedding.token(item_indices)

        # Compute dot product scores
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits

# --- END OF MODIFIED FILE BERT4Rec.py ---