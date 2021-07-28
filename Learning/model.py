import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Value(nn.Module):
    def __init__(self, config):
        super(Value, self).__init__()
        self.dim_embed = config.dim_embed
        self.dim_hidden = config.dim_hidden

        self.fc1 = nn.Linear(self.dim_embed, self.dim_hidden, bias = False)
    
    def forward(self, x):
        x = self.fc1(x)
        return x

class Key(nn.Module):
    def __init__(self, config):
        super(Key, self).__init__()
        self.dim_embed = config.dim_embed
        self.dim_hidden = config.dim_hidden

        self.fc1 = nn.Linear(self.dim_embed, self.dim_hidden, bias = False)
       
    def forward(self, x):
        x = self.fc1(x)
        return x

class Query(nn.Module):
    def __init__(self, config):
        super(Query, self).__init__()
        self.dim_embed = config.dim_embed
        self.dim_hidden = config.dim_hidden

        self.fc1 = nn.Linear(self.dim_embed, self.dim_hidden, bias = False)
    
    def forward(self, x):
        x = self.fc1(x)
        return x


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        self.max_len = config.max_len
        self.dim_embed = config.dim_embed

        pe = th.zeros(self.max_len, self.dim_embed)
        position = th.arange(0, self.max_len, dtype=th.float).unsqueeze(1)
        
        div_term = th.exp(th.arange(0, self.dim_embed, 2).float() * (-th.log(10000.0) / self.dim_embed))
        
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, seq_len):
        x = self.pe[:seq_len, :].squeeze(1)
        return x  


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
    
    def forward(self, Q, K, V, attn_mask=None):
        """
        Attention(Q, K, V) = norm(QK)V
        """
        a = th.matmul(Q, K.transpose(2,1).float())
        a /= th.sqrt(th.tensor(Q.shape[-1]).float()) # scaled
        
        # Mask(opt.)
        if attn_mask is not None:
            a.masked_fill_(attn_mask, -1e9)

        attn_p = th.softmax(a, -1) # (num_q_seq, num_k_seq)
        attn_v = th.matvul(a, V) # (num_q_seq, dim_hidden)
        return attn_v, attn_p
    

class MultiHeadAttention(th.nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.batch_size = config.batch_size
        self.dim_hidden = config.dim_hidden
        self.dim_head = config.dim_head
        self.num_heads = config.num_heads

        self.W_Q = Query(self.dim_hidden, self.dim_head * self.num_heads)
        self.W_K = Key(self.dim_hidden, self.dim_head * self.num_heads)
        self.W_V = Value(self.dim_hidden, self.dim_head * self.num_heads)
        self.scaled_dot_attn = Attention(config)
        self.fc1 = nn.Linear(self.dim_head * self.num_heads, self.dim_hidden)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, Q, K, V, attn_mask=None):
        # (batch_size, num_heads, num_q_seq, dim_head)
        q_s = self.W_Q(Q).view(self.batch_size, -1, self.num_heads, self.dim_head).transpose(1,2)
        # (batch_size, num_heads, num_k_seq, dim_head)
        k_s = self.W_K(K).view(self.batch_size, -1, self.num_heads, self.dim_head).transpose(1,2)
        # (batch_size, num_heads, num_v_seq, dim_head)
        v_s = self.W_V(V).view(self.batch_size, -1, self.num_heads, self.dim_head).transpose(1,2)

        # |TODO| check
        # (batch_size, num_heads, num_q_seq, n_k_seq)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # |TODO| check shape
        # (batch_size, num_heads, num_q_seq, dim_head), (batch_size, num_heads, num_q_seq, num_k_seq)
        attn_v, attn_p = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
        # (batch_size, num_heads, num_q_seq, num_heads * dim_head)
        attn_v = attn_v.transpose(1, 2).contiguous().view(self.batch_size, -1, self.num_heads * self.dim_head)
        # (batch_size, num_q_seq, dim_hidden)
        output = self.fc1(attn_v)
        output = self.dropout(output)

        # (batch_size, num_q_seq, dim_hidden), (batch_size, num_heads, num_q_seq, num_k_seq)
        return output, attn_p
    

class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim_hidden = config.dim_hidden
        self.dim_ffn = config.dim_ffn

        self.conv1 = nn.Conv1d(in_channels=self.dim_hidden, out_channels=self.dim_ffn, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.dim_ffn, out_channels=self.dim_hidden, kernel_size=1)
        # |TODO| How to change by config?
        self.act_fn = F.gelu # original: ReLU

    def forward(self, inputs):
        # (batch_size, dim_ffn, num_seq)
        output = self.act_fn(self.conv1(inputs.transpose(1, 2)))
        # (batch_size, num_seq, dim_hidden)
        output = self.conv2(output).transpose(1, 2)

        return output


class GPT2DecoderLayer(nn.Module):
    def __init__(self, config):
        super(GPT2DecoderLayer, self).__init__()
        self.config = config

        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.dim_hidden)
        self.ffn = FeedForwardNetwork(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.dim_hidden)
    
    def forward(self, x, attn_mask):
                # (batch_size, num_dec_seq, dim_hidden), (batch_size, num_heads, num_dec_seq, num_dec_seq)
        self_attn_out, self_attn_prob = self.self_attn(x, x, x, attn_mask)
        self_attn_out = self.layer_norm1(x + self_attn_out)

        # (batch_size, num_dec_seq, dim_hidden)
        ffn_out = self.ffn(self_attn_out)
        ffn_outputs = self.layer_norm2(self_attn_out + ffn_out)

        return ffn_out, self_attn_prob


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim_observation = config.dim_observation
        self.dim_action = config.dim_action
        self.dim_embed = config.dim_embed
        self.dim_hidden = config.dim_hidden
        self.num_layers = config.num_layers
        self.action_tanh = config.action_tanh

        self.embed_observation = nn.Linear(self.dim_observation, self.dim_embed)
        self.embed_action = nn.Linear(self.dim_action, self.dim_embed)
        self.pos_embed = PositionalEncoding(self.dim_embed)
        self.ln = nn.LayerNorm(self.dim_hidden)

        self.layers = nn.ModuleList([GPT2DecoderLayer(self.config) for _ in range(self.num_layers)])
    
        self.predict_action = nn.Sequential(*([nn.Linear(self.dim_hidden, self.dim_action)] + ([nn.Tanh()] if self.action_tanh else [])))

    def forward(self, observations, actions, attn_mask=None):
        batch_size, seq_len = observations.shape[0], observations.shape[1]

        if attn_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attn_mask = th.ones((batch_size, seq_len), dtype=th.long)

        time_embeddings = self.pos_embed(seq_len)

        observation_embeddings = self.embed_observation(observations) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        
        # this makes the sequence look like (R_1, o_1, a_1, R_2, o_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = th.stack((observation_embeddings, action_embeddings), dim=1).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_len, self.dim_hidden)
        stacked_inputs = self.ln(stacked_inputs)        

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = th.stack((attn_mask, attn_mask, attn_mask), dim=1).permute(0, 2, 1).reshape(batch_size, 2*seq_len)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        dec_outputs, attn_prob = self.layers(stacked_inputs, stacked_attention_mask)

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        dec_outputs = dec_outputs.reshape(batch_size, seq_len, 2, self.dim_hidden).permute(0, 2, 1, 3)

        # get predictions
        # return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        # state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(dec_outputs[:,1])  # predict next action given state

        return action_preds
