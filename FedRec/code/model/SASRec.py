import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
class MultiHeadAttention (nn.Module):
    def __init__ (self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,):
        super (MultiHeadAttention, self).__init__ ()
        if hidden_size % n_heads != 0:
            raise ValueError ("The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (hidden_size, n_heads))
        self.num_attention_heads = n_heads
        self.attention_head_size = int (hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt (self.attention_head_size)
        self.query = nn.Linear (hidden_size, self.all_head_size)
        self.key = nn.Linear (hidden_size, self.all_head_size)
        self.value = nn.Linear (hidden_size, self.all_head_size)
        self.softmax = nn.Softmax (dim = -1)
        self.attn_dropout = nn.Dropout (attn_dropout_prob)
        self.dense = nn.Linear (hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm (hidden_size, eps = layer_norm_eps)
        self.out_dropout = nn.Dropout (hidden_dropout_prob)
    def transpose_for_scores ( self, x ):
        new_x_shape = x.size () [:-1] + (self.num_attention_heads, self.attention_head_size,)
        x = x.view (*new_x_shape)
        return x
    def forward ( self, input_tensor, attention_mask ):
        mixed_query_layer = self.query (input_tensor)
        mixed_key_layer = self.key (input_tensor)
        mixed_value_layer = self.value (input_tensor)
        query_layer = self.transpose_for_scores (mixed_query_layer).permute (0, 2, 1, 3)
        key_layer = self.transpose_for_scores (mixed_key_layer).permute (0, 2, 3, 1)
        value_layer = self.transpose_for_scores (mixed_value_layer).permute (0, 2, 1, 3)
        attention_scores = torch.matmul (query_layer, key_layer)
        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask
        attention_probs = self.softmax (attention_scores)
        attention_probs = self.attn_dropout (attention_probs)
        context_layer = torch.matmul (attention_probs, value_layer)
        context_layer = context_layer.permute (0, 2, 1, 3).contiguous ()
        new_context_layer_shape = context_layer.size () [:-2] + (self.all_head_size,)
        context_layer = context_layer.view (*new_context_layer_shape)
        hidden_states = self.dense (context_layer)
        hidden_states = self.out_dropout (hidden_states)
        hidden_states = self.LayerNorm (hidden_states + input_tensor)
        return hidden_states
class FeedForward (nn.Module):
    def __init__ (self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps,):
        super (FeedForward, self).__init__ ()
        self.dense_1 = nn.Linear (hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act (hidden_act)
        self.dense_2 = nn.Linear (inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm (hidden_size, eps = layer_norm_eps)
        self.dropout = nn.Dropout (hidden_dropout_prob)
    def get_hidden_act ( self, act ):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": F.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN [act]
    def gelu ( self, x ):
        return x * 0.5 * (1.0 + torch.erf (x / math.sqrt (2.0)))
    def swish ( self, x ):
        return x * torch.sigmoid (x)
    def forward ( self, input_tensor ):
        hidden_states = self.dense_1 (input_tensor)
        hidden_states = self.intermediate_act_fn (hidden_states)
        hidden_states = self.dense_2 (hidden_states)
        hidden_states = self.dropout (hidden_states)
        hidden_states = self.LayerNorm (hidden_states + input_tensor)
        return hidden_states
class TransformerLayer (nn.Module):
    def __init__ (self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps,):
        super (TransformerLayer, self).__init__ ()
        self.multi_head_attention = MultiHeadAttention (n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps)
        self.feed_forward = FeedForward (hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps,)
    def forward ( self, hidden_states, attention_mask ):
        attention_output = self.multi_head_attention (hidden_states, attention_mask)
        feedforward_output = self.feed_forward (attention_output)
        return feedforward_output
class TransformerEncoder (nn.Module):
    def __init__ (self, n_layers = 2, n_heads = 2, hidden_size = 64, inner_size = 256, hidden_dropout_prob = 0.5, attn_dropout_prob = 0.5, hidden_act = "gelu", layer_norm_eps = 1e-12,):
        super (TransformerEncoder, self).__init__ ()
        layer = TransformerLayer (n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps,)
        self.layer = nn.ModuleList ([copy.deepcopy (layer) for _ in range (n_layers)])
    def forward ( self, hidden_states, attention_mask, output_all_encoded_layers = True ):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module (hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append (hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append (hidden_states)
        return all_encoder_layers
class SASRec (nn.Module):
    def __init__ ( self, config, item_maxid ):
        super (SASRec, self).__init__ ()
        self.config=config
        self.n_layers = config['num_layers']
        self.n_heads = config['num_heads']
        self.hidden_size = config['hidden_size']
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['dropout']
        self.attn_dropout_prob = config['dropout']
        self.hidden_act = 'gelu'
        self.layer_norm_eps = 1e-8
        self.max_seq_length = config['max_seq_len']
        self.loss_type = 'bce'
        self.device = "cuda" if torch.cuda.is_available () else "cpu"
        self.n_items = item_maxid + 1
        self.item_embedding = nn.Embedding (self.n_items, self.hidden_size, padding_idx = 0)
        self.position_embedding = nn.Embedding (self.max_seq_length + 1, self.hidden_size, padding_idx = 0)
        self.trm_encoder = TransformerEncoder (n_layers = self.n_layers, n_heads = self.n_heads, hidden_size = self.hidden_size, inner_size = self.inner_size, hidden_dropout_prob = self.hidden_dropout_prob, attn_dropout_prob = self.attn_dropout_prob, hidden_act = self.hidden_act, layer_norm_eps = self.layer_norm_eps,)
        self.LayerNorm = nn.LayerNorm (self.hidden_size, eps = self.layer_norm_eps)
        self.dropout = nn.Dropout (self.hidden_dropout_prob)
        self.emb_dropout = nn.Dropout (self.hidden_dropout_prob)
        self.init_param ()
    def init_param ( self ):
        for name, param in self.named_parameters():
            try:
                nn.init.xavier_normal_(param.data)
            except:
                pass
        self.position_embedding.weight.data[0, :] = 0
        self.item_embedding.weight.data[0, :] = 0
    def get_attention_mask ( self, item_seq, bidirectional = False ):
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze (1).unsqueeze (2)
        if not bidirectional:
            extended_attention_mask = torch.tril (extended_attention_mask.expand ((-1, -1, item_seq.size (-1), -1)))
        extended_attention_mask = torch.where (extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask
    def forward ( self, x, x_lens ):
        item_seq = x
        if item_seq.size (1) > self.max_seq_length:
            item_seq = item_seq [:, -self.max_seq_length:]
            x_lens = torch.clamp (x_lens, max = self.max_seq_length)
        position_ids = torch.arange (item_seq.size (1), dtype = torch.long, device = item_seq.device)
        position_ids = position_ids.unsqueeze (0).expand_as (item_seq)
        position_ids = torch.clamp (position_ids, 0, self.max_seq_length - 1)
        position_embedding = self.position_embedding (position_ids)
        item_seq = torch.clamp (item_seq, 0, self.n_items - 1)
        item_emb = self.item_embedding (item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm (input_emb)
        input_emb = self.dropout (input_emb)
        extended_attention_mask = self.get_attention_mask (item_seq)
        trm_output = self.trm_encoder (input_emb, extended_attention_mask, output_all_encoded_layers = True)
        output = trm_output [-1]
        seq_output = torch.matmul (output, self.item_embedding.weight.transpose (0, 1))
        return seq_output
    def loss_function ( self, seq_out, padding_mask, target, neg, seq_len ):
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
    def log2feats(self, log_seqs):
        if not isinstance(log_seqs, torch.Tensor):
            log_seqs = torch.LongTensor(log_seqs).to(self.device)
        seqs = self.item_embedding(log_seqs)
        seqs *= self.item_embedding.embedding_dim ** 0.5
        import numpy as np
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        poss *= (log_seqs.cpu().numpy() != 0)
        seqs += self.position_embedding(torch.LongTensor(poss).to(self.device))
        seqs = self.emb_dropout(seqs)
        extended_attention_mask = self.get_attention_mask(log_seqs)
        trm_output = self.trm_encoder(seqs, extended_attention_mask, output_all_encoded_layers=True)
        log_feats = self.LayerNorm(trm_output[-1])
        return log_feats
    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        if not isinstance(item_indices, torch.Tensor):
            item_indices = torch.LongTensor(item_indices).to(self.device)
        item_embs = self.item_embedding(item_indices)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits
