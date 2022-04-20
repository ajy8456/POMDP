import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Value(nn.Module):
    def __init__(self, dim_hidden, dim_embed):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(dim_embed, dim_hidden, bias = False)
    
    def forward(self, x):
        x = self.fc1(x)
        return x

class Key(nn.Module):
    def __init__(self, dim_hidden, dim_embed):
        super(Key, self).__init__()
        self.fc1 = nn.Linear(dim_embed, dim_hidden, bias = False)
       
    def forward(self, x):
        x = self.fc1(x)
        return x

class Query(nn.Module):
    def __init__(self, dim_hidden, dim_embed):
        super(Query, self).__init__()
        self.fc1 = nn.Linear(dim_embed, dim_hidden, bias = False)
    
    def forward(self, x):
        x = self.fc1(x)
        return x


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()

        pe = th.zeros(config.max_len, config.dim_embed)
        position = th.arange(0, config.max_len, dtype=th.float).unsqueeze(1)
        
        div_term = th.exp(th.arange(0, config.dim_embed, 2).float() * (-math.log(10000.0) / config.dim_embed))
        
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1).to(config.device)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x  


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, Q, K, V, attn_mask=None):
        """
        Attention(Q, K, V) = norm(QK)V
        """

        # if self.config.print_in_out:
        #     print("Input of Attention Q:", Q)
        #     print("Input of Attention K:", K)
        #     print("Input of Attention V:", V)

        a = th.matmul(Q, K.transpose(-1,-2).float())

        # if self.config.print_in_out:
        #     print("Output of a:", a)

        a /= th.sqrt(th.tensor(Q.shape[-1]).float()) # scaled

        # if self.config.print_in_out:
        #     print("After scaled:", a)
        
        # Mask(opt.)
        if attn_mask is not None:
            a.masked_fill_(attn_mask, -1e9)

        # if self.config.print_in_out:
        #     print("After masked:", a)

        attn_p = th.softmax(a, -1) # (num_q_seq, num_k_seq)

        # if self.config.print_in_out:
        #     print("Attention score:", attn_p)

        attn_v = th.matmul(self.dropout(attn_p), V) # (num_q_seq, dim_hidden)

        # if self.config.print_in_out:
        #     print("Attention value:", attn_v)

        return attn_v, attn_p
    

class MultiHeadAttention(th.nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.config = config
        self.dim_hidden = config.dim_hidden
        self.dim_head = config.dim_head
        self.num_heads = config.num_heads

        self.W_Q = Query(self.dim_hidden, self.dim_head * self.num_heads)
        self.W_K = Key(self.dim_hidden, self.dim_head * self.num_heads)
        self.W_V = Value(self.dim_hidden, self.dim_head * self.num_heads)
        self.scaled_dot_attn = Attention(config)
        self.fc1 = nn.Linear(self.dim_head * self.num_heads, self.dim_hidden)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, attn_mask=None):
        batch_size = x.shape[0]

        # if self.config.print_in_out:
        #     print("Input of MultiHeadAttention:", x)

        # (batch_size, num_heads, num_q_seq, dim_head)
        q_s = self.W_Q(x).view(batch_size, -1, self.num_heads, self.dim_head).transpose(1,2)

        # if self.config.print_in_out:
        #     print("Query:", q_s)

        # (batch_size, num_heads, num_k_seq, dim_head)
        k_s = self.W_K(x).view(batch_size, -1, self.num_heads, self.dim_head).transpose(1,2)

        # if self.config.print_in_out:
        #     print("Key:", k_s)

        # (batch_size, num_heads, num_v_seq, dim_head)
        v_s = self.W_V(x).view(batch_size, -1, self.num_heads, self.dim_head).transpose(1,2)

        # if self.config.print_in_out:
        #     print("Value:", v_s)

        # |TODO| check
        # (batch_size, num_heads, num_q_seq, n_k_seq)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).transpose(0,1)

        # |TODO| check shape
        # (batch_size, num_heads, num_q_seq, dim_head), (batch_size, num_heads, num_q_seq, num_k_seq
        attn_v, attn_p = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)

        # if self.config.print_in_out:
        #     print("Output of Attention value:", attn_v)
        #     print("Output of Attention score:", attn_p)

        # (batch_size, num_heads, num_q_seq, num_heads * dim_head)
        attn_v = attn_v.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_head)
        # (batch_size, num_q_seq, dim_hidden)
        output = self.fc1(attn_v)
        
        # if self.config.print_in_out:
        #     print("Output of FC1:", output)

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
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, attn_mask):
        # (batch_size, num_dec_seq, dim_hidden), (batch_size, num_heads, num_dec_seq, num_dec_seq)

        # if self.config.print_in_out:
        #     print("Input of MultiHeadAttention:", x, attn_mask)

        self_attn_out, self_attn_prob = self.self_attn(x, attn_mask)

        # if self.config.print_in_out:
        #     print("Output of MultiHeadAttention:", self_attn_out, self_attn_prob)

        self_attn_out = self.layer_norm1(x + self.dropout(self_attn_out))

        # if self.config.print_in_out:
        #     print("Output of LN1:", self_attn_out)

        # (batch_size, num_dec_seq, dim_hidden)
        ffn_out = self.ffn(self_attn_out)

        # if self.config.print_in_out:
        #     print("Output of FFN:", ffn_out)

        ffn_outputs = self.layer_norm2(self_attn_out + self.dropout(ffn_out))

        # if self.config.print_in_out:
        #     print("Output of LN2:", ffn_out)

        return ffn_outputs, self_attn_prob


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim_observation = config.dim_observation
        self.dim_action = config.dim_action
        self.dim_state = config.dim_state
        self.dim_reward = config.dim_reward
        self.dim_embed = config.dim_embed
        self.dim_hidden = config.dim_hidden
        self.num_layers = config.num_layers
        self.action_tanh = config.action_tanh
        self.max_len = config.max_len
        self.seq_len = config.seq_len

        # self.embed_observation = nn.Linear(self.dim_observation, self.dim_embed)
        # self.embed_action = nn.Linear(self.dim_action, self.dim_embed)
        if self.config.use_reward:
            if self.config.use_mask_padding:
                self.embed = nn.Linear(self.dim_observation + self.dim_action + self.dim_reward + 1, self.dim_embed)
            else:
                self.embed = nn.Linear(self.dim_observation + self.dim_action + self.dim_reward, self.dim_embed)
            # self.predict_reward = nn.Linear(self.seq_len * self.dim_hidden, self.dim_reward)
        else:
            if self.config.use_mask_padding:
                self.embed = nn.Linear(self.dim_observation + self.dim_action + 1, self.dim_embed)
            else:
                self.embed = nn.Linear(self.dim_observation + self.dim_action, self.dim_embed)
        
        if self.config.randomize:
            if self.config.use_mask_padding:
                self.embed_goal = nn.Linear(self.dim_state + 1, self.dim_embed)
            else:
                self.embed_goal = nn.Linear(self.dim_state, self.dim_embed)

        # select trainable/fixed positional encoding
        if self.config.train_pos_en:
            self.embed_timestep = nn.Embedding(self.max_len, self.dim_embed)
        else:
            self.pos_embed = PositionalEncoding(self.config)
        
        self.dropout = nn.Dropout(config.dropout)
        self.ln = nn.LayerNorm(self.dim_hidden)

        self.layers = []
        for _ in range(self.num_layers):
            self.layers.append(GPT2DecoderLayer(self.config))
        # |NOTE| need!!!! https://michigusa-nlp.tistory.com/26
        self.layers = nn.ModuleList(self.layers)

        # self.ln2 = nn.LayerNorm(self.dim_hidden)

        if config.model == 'CVAE' or 'ValueNet' or 'PolicyValueNet':
            # self.fc_condi = nn.Sequential(*([nn.Linear(self.seq_len * self.dim_hidden, self.config.dim_condition)] + ([nn.Tanh()] if self.action_tanh else [])))
            if config.randomize:
                self.fc_condi = nn.Linear((self.seq_len + 1) * self.dim_hidden, self.config.dim_condition)
            else:
                self.fc_condi = nn.Linear(self.seq_len * self.dim_hidden, self.config.dim_condition)
        else:
            self.predict_action = nn.Sequential(*([nn.Linear(self.seq_len * self.dim_hidden, self.dim_action)] + ([nn.Tanh()] if self.action_tanh else [])))

    # def forward(self, observations, actions, attn_mask=None):
    def forward(self, data):
        batch_size, seq_len = data['observation'].shape[0], data['observation'].shape[1]

        # for consisting token as (o,a,r); not separating
        if self.config.use_reward:
            inputs = th.cat((data['observation'], data['action'], data['reward']), dim=-1)
        else:
            inputs = th.cat((data['observation'], data['action']), dim=-1)


        if self.config.use_mask_padding:
            if self.config.randomize:
                mask_goal = data['mask'][:, -1].float().reshape(-1, 1, 1)
                inputs_goal= th.cat((data['goal_state'], mask_goal), dim=-1)
                mask = th.unsqueeze(data['mask'][:, :-1].float(), dim=-1)
                inputs = th.cat((inputs, mask), dim=-1)
            else:
                mask = th.unsqueeze(data['mask'].float(), dim=-1)
                inputs = th.cat((inputs, mask), dim=-1)

        # if self.config.print_in_out:
        #     print("Input of embedding:", inputs)

        if self.config.randomize:
            input_embeddings = F.gelu(th.cat((self.embed(inputs), self.embed_goal(inputs_goal)), dim=1))
        else:
            input_embeddings = F.gelu(self.embed(inputs))

        # if self.config.print_in_out:
        #     print("Output of embedding:", input_embeddings)
        
        # select trainable/fixed positional encoding
        if self.config.train_pos_en:
            time_embeddings = self.embed_timestep(data['timestep'])
            input_embeddings = input_embeddings + time_embeddings
        else:
            input_embeddings = self.pos_embed(input_embeddings)

        # if 'mask' not in data:
        #     # attention mask for GPT: 1 if can be attended to, 0 if not
        #     attn_mask = th.ones((batch_size, seq_len), dtype=th.long)
        attn_mask = ~data['mask']

        input_embeddings = self.dropout(input_embeddings)
        input_embeddings = self.ln(input_embeddings)

        # if self.config.print_in_out:
        #     print("Input of 1-th GPT2DecoderLayer:", input_embeddings)

        dec_outputs, attn_prob = self.layers[0](input_embeddings, attn_mask)

        # if self.config.print_in_out:
        #     print("Output of 1-th GPT2DecoderLayer:", dec_outputs)

        for l, layer in enumerate(self.layers[1:]):

            # if self.config.print_in_out:
            #     print(f"Input of {l+2}-th GPT2DecoderLayer:", dec_outputs)

            dec_outputs, attn_prob = layer(dec_outputs, attn_mask)

            # if self.config.print_in_out:
            #     print(f"Output of {l+2}-th GPT2DecoderLayer:", dec_outputs)

        # dec_outputs = self.ln2(dec_outputs)

        # get predictions
        # if self.config.print_in_out:
            # print(f"Input of action predict FC:", dec_outputs)

        if self.config.model == 'CVAE' or 'ValueNet' or 'PolicyValueNet':
            # out = self.fc_condi(dec_outputs.flatten(start_dim=1))
            out = F.gelu(self.fc_condi(dec_outputs.flatten(start_dim=1)))
        else:    
            out = {}
            pred_action = self.predict_action(dec_outputs.flatten(start_dim=1))  # predict next action given state
            pred_action = th.squeeze(pred_action)
            out['action'] = pred_action

        # if self.config.print_in_out:
        #     print(f"Output of action predict FC:", pred_action)

        # if self.config.use_reward:
        #     pred_reward = self.predict_reward(dec_outputs.flatten(start_dim=1))
        #     pred_reward = th.squeeze(pred_reward)
        #     pred['reward'] = pred_reward

        return out


class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.config = config
        self.dim_observation = config.dim_observation
        self.dim_action = config.dim_action
        self.dim_reward = config.dim_reward
        self.dim_embed = config.dim_embed
        self.dim_hidden = config.dim_hidden
        self.num_layers = config.num_layers

        if self.config.use_reward:
            if self.config.use_mask_padding:
                self.embed = nn.Linear(self.dim_observation + self.dim_action + self.dim_reward + 1, self.dim_embed)
            else:
                self.embed = nn.Linear(self.dim_observation + self.dim_action + self.dim_reward, self.dim_embed)
            # self.predict_reward = nn.Linear(self.seq_len * self.dim_hidden, self.dim_reward)
        else:
            if self.config.use_mask_padding:
                self.embed = nn.Linear(self.dim_observation + self.dim_action + 1, self.dim_embed)
            else:
                self.embed = nn.Linear(self.dim_observation + self.dim_action, self.dim_embed)
        
        self.rnn = nn.RNN(input_size=self.dim_embed, hidden_size=self.dim_hidden, num_layers=self.num_layers, dropout=self.config.dropout, batch_first=True)
        self.predict_action = nn.Linear(self.dim_hidden, self.dim_action)


    def forward(self, data):
        batch_size, seq_len = data['observation'].shape[0], data['observation'].shape[1]
        
        if self.config.use_reward:
            inputs = th.cat((data['observation'], data['action'], data['reward']), dim=-1)
        else:
            inputs = th.cat((data['observation'], data['action']), dim=-1)

        if self.config.use_mask_padding:
            mask = th.unsqueeze(data['mask'].float(), dim=-1)
            inputs = th.cat((inputs, mask), dim=-1)
            
        input_embeddings = self.embed(inputs)

        if 'mask' in data:
            stacked_attention_mask = th.unsqueeze(data['mask'], dim=-1)
            stacked_attention_mask = th.repeat_interleave(~stacked_attention_mask, self.dim_hidden, dim=-1)
            input_embeddings.masked_fill_(stacked_attention_mask, 0)

        # # swithing dimension order for batch_first=False
        # input_embeddings = th.transpose(input_embeddings, 0, 1)

        h_0 = th.zeros(self.num_layers, batch_size, self.dim_hidden).to(self.config.device)
        output, h_n = self.rnn(input_embeddings, h_0)

        pred = {}
        pred_action = self.predict_action(output[:, -1, :])
        pred_action = th.squeeze(pred_action)
        pred['action'] = pred_action
        # if self.config.use_reward:
        #     pred_reward = self.predict_reward(output[:, -1, :])
        #     pred_reward = th.squeeze(pred_reward)
        #     pred['reward'] = pred_reward

        return pred


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.config = config
        self.dim_observation = config.dim_observation
        self.dim_action = config.dim_action
        self.dim_reward = config.dim_reward
        self.dim_embed = config.dim_embed
        self.dim_hidden = config.dim_hidden
        self.num_layers = config.num_layers

        if self.config.use_reward:
            if self.config.use_mask_padding:
                self.embed = nn.Linear(self.dim_observation + self.dim_action + self.dim_reward + 1, self.dim_embed)
            else:
                self.embed = nn.Linear(self.dim_observation + self.dim_action + self.dim_reward, self.dim_embed)
            # self.predict_reward = nn.Linear(self.seq_len * self.dim_hidden, self.dim_reward)
        else:
            if self.config.use_mask_padding:
                self.embed = nn.Linear(self.dim_observation + self.dim_action + 1, self.dim_embed)
            else:
                self.embed = nn.Linear(self.dim_observation + self.dim_action, self.dim_embed)
        
        self.lstm = nn.LSTM(input_size=self.dim_embed, hidden_size=self.dim_hidden, num_layers=self.num_layers, dropout=self.config.dropout, batch_first=True)
        self.predict_action = nn.Linear(self.dim_hidden, self.dim_action)


    def forward(self, data):
        batch_size, seq_len = data['observation'].shape[0], data['observation'].shape[1]
        
        if self.config.use_reward:
            inputs = th.cat((data['observation'], data['action'], data['reward']), dim=-1)
        else:
            inputs = th.cat((data['observation'], data['action']), dim=-1)

        if self.config.use_mask_padding:
            mask = th.unsqueeze(data['mask'].float(), dim=-1)
            inputs = th.cat((inputs, mask), dim=-1)
            
        input_embeddings = self.embed(inputs)

        if 'mask' in data:
            stacked_attention_mask = th.unsqueeze(data['mask'], dim=-1)
            stacked_attention_mask = th.repeat_interleave(~stacked_attention_mask, self.dim_hidden, dim=-1)
            input_embeddings.masked_fill_(stacked_attention_mask, 0)

        # # swithing dimension order for batch_first=False
        # input_embeddings = th.transpose(input_embeddings, 0, 1)

        h_0 = th.zeros(self.num_layers, batch_size, self.dim_hidden).to(self.config.device)
        c_0 = th.zeros(self.num_layers, batch_size, self.dim_hidden).to(self.config.device)
        output, h_n = self.lstm(input_embeddings, (h_0, c_0))

        pred = {}
        pred_action = self.predict_action(output[:, -1, :])
        pred_action = th.squeeze(pred_action)
        pred['action'] = pred_action
        # if self.config.use_reward:
        #     pred_reward = self.predict_reward(output[:, -1, :])
        #     pred_reward = th.squeeze(pred_reward)
        #     pred['reward'] = pred_reward

        return pred


class CVAEEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.layer_sizes = config.encoder_layer_sizes
        self.latent_size = config.latent_size
        self.dim_condition = config.dim_condition

        self.layer_sizes[0] += self.dim_condition

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(self.layer_sizes[-1], self.latent_size)
        self.linear_log_var = nn.Linear(self.layer_sizes[-1], self.latent_size)

    def forward(self, x, c):
        # |TODO| check shape after cat
        x = th.cat((x, c), dim=-1)
        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class CVAEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer_sizes = config.decoder_layer_sizes
        self.latent_size = config.latent_size
        self.dim_condition = config.dim_condition

        self.MLP = nn.Sequential()

        input_size = self.latent_size + self.dim_condition

        for i, (in_size, out_size) in enumerate(zip([input_size]+self.layer_sizes[:-1], self.layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(self.layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                if self.config.action_tanh:
                    self.MLP.add_module(name="tanh", module=nn.Tanh())

    def forward(self, z, c):
        z = th.cat((z, c), dim=-1)
        x = self.MLP(z)

        return x


class CVAE(nn.Module):
    def __init__(self, config):
        super(CVAE, self).__init__()
        
        self.config = config
        self.latent_size = config.latent_size
        self.encoder_layer_sizes = config.encoder_layer_sizes
        self.decoder_layer_sizes = config.decoder_layer_sizes
        self.dim_condition = config.dim_condition

        assert type(self.latent_size) == int
        assert type(self.encoder_layer_sizes) == list
        assert type(self.decoder_layer_sizes) == list

        self.embed = nn.Linear(self.config.dim_action, self.config.dim_embed)
        self.encoder = CVAEEncoder(config)
        self.decoder = CVAEDecoder(config)
        # |TODO| make RNN, LSTM also available
        self.condition = GPT2(config)

    def forward(self, data):
        # For training
        x = data['next_action'].squeeze()
        x = F.gelu(self.embed(x))

        c = self.condition(data).squeeze()

        mean, log_var = self.encoder(x, c)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, mean, log_var, z

    def reparameterize(self, mu, log_var):
        std = th.exp(0.5 * log_var)
        eps = th.randn_like(std)

        return mu + eps * std

    def inference(self, data):
        c = self.condition(data)
        z = th.randn([c.size(0), self.config.latent_size]).to(self.config.device)
        recon_x = self.decoder(z, c)

        return recon_x


class ValueNet(nn.Module):
    def __init__(self, config):
        super(ValueNet, self).__init__()

        self.config = config
        self.condition = GPT2(config)
        self.fc = nn.Linear(config.dim_condition, 1)
        # self.fc1 = nn.Linear(config.dim_condition, config.dim_condition)
        # self.fc2 = nn.Linear(config.dim_condition, 1)

    def forward(self, data):
        c = self.condition(data)
        accumulated_reward = self.fc(c)
        # accumulated_reward = F.relu(self.fc1(c))
        # accumulated_reward = self.fc2(accumulated_reward)

        return accumulated_reward


if __name__ == '__main__':
    from torchsummary import summary
    from run import Settings
    config = Settings()
