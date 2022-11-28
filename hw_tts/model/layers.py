import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, q, k, v, mask=None):
        # q, k, v: [ (batch_size * n_heads) x seq_len x hidden_size ]
        # attn: [ (batch_size * n_heads) x seq_len x seq_len ]
        # output: [ (batch_size * n_heads) x seq_len x hidden_size ]
        
        attn = torch.einsum('BLD,BlD->BLl', q, k)
        attn /= self.temperature

        if mask is not None:
            attn = torch.masked_fill(attn, mask, -torch.inf)

        attn = self.dropout(self.softmax(attn))
        output = torch.einsum('BLl,BlD->BLD', attn, v)

        return output, attn


class MultiHeadSelfAttention(nn.Module):
    ''' Multi-Head Self Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(
            temperature=d_k**0.5) 
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
         # normal distribution initialization better than kaiming(default in pytorch)
        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_v))) 
        
    def forward(self, x, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_x, _ = x.size()

        residual = x

        x = self.layer_norm(x)

        q = self.w_qs(x).view(sz_b, len_x, n_head, d_k)
        k = self.w_ks(x).view(sz_b, len_x, n_head, d_k)
        v = self.w_vs(x).view(sz_b, len_x, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_v)  # (n*b) x lv x dv
        
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_x, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_x, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = output + residual

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, model_config, d_in, d_hid, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in, d_hid, kernel_size=model_config['fft_conv1d_kernel'][0], padding=model_config['fft_conv1d_padding'][0])
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=model_config['fft_conv1d_kernel'][1], padding=model_config['fft_conv1d_padding'][1])

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = self.layer_norm(x).transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = output + residual

        return output


class FFTBlock(nn.Module):
    """FFT Block"""

    def __init__(self,
                 model_config,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadSelfAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            model_config, d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, mask=slf_attn_mask)
        
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn