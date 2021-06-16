# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from torch.nn.modules.batchnorm import BatchNorm1d

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

class AdaIN(nn.Module):
    def __init__(self, latent_dim, style_dim):
        super().__init__()
        self.tran = nn.Linear(style_dim, 2)
    
    def forward(self, x, t):
        """
        x: (N, latent_dim)
        t: (N, style_dim)
        """
        style = self.tran(t)
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)+1e-6
        out = (x-mean) / std * style[:, [0]] + style[:, [1]]
        return out
    
class DeformationNet(nn.Module):
    def __init__(self, ch_pos, ch_time, n_dim, n_layer):
        super().__init__()
        self.act = nn.ReLU(True)

        self.start = nn.Linear(ch_pos, n_dim)
        self.linear = [nn.Linear(n_dim, n_dim) for i in range(n_layer-1)]
        self.linear.append(nn.Linear(n_dim, 3))
        self.linear = nn.ModuleList(self.linear)

        self.adain = nn.ModuleList([AdaIN(n_dim, ch_time) for i in range(n_layer)])
        self.n_layer = n_layer//2
    
    def forward(self, x, t, s):
        """
        x: (N, latent_dim)
        t: (N, style_dim)
        out: (N, 3)
        """
        out = self.start(x)
        # t_in = torch.cat([t, s], -1)
        for i in range(self.n_layer):
            out = self.act(out)
            out = self.adain[i](out, t)
            out = self.linear[i](out)
        for i in range(self.n_layer, self.n_layer+self.n_layer):
            out = self.act(out)
            out = self.adain[i](out, s)
            out = self.linear[i](out)
        return out

class MLP(nn.Module):
    def __init__(self, ch_pos, ch_time, n_dim, n_layer):
        super().__init__()
        ch_in = ch_pos+ch_time*2
        self.net = [nn.Linear(ch_in, n_dim)]
        for i in range(n_layer-1):
            self.net += [
                nn.ReLU(True),
                # nn.BatchNorm1d(n_dim),
                nn.Linear(n_dim, n_dim, bias=True)
            ]
        self.net += [
            nn.ReLU(True),
            # nn.BatchNorm1d(n_dim),
            nn.Linear(n_dim, 4, bias=True)
        ]
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x, t, s):
        """
        x: (N, latent_dim)
        t: (N, style_dim)
        out: (N, 3)
        """
        ret, occ = self.net(torch.cat([x, t, s], -1)).split([3, 1], dim=-1)
        occ = torch.sigmoid(occ)
        return ret, occ


class DeformationModel(nn.Module):
    # Map (x, y, z, t0, t1) to (x', y', z', ...)
    
    def __init__(self, n_dim=128, n_layer=6, n_emb_pos=8, n_emb_time=4):
        super().__init__()
        self.emb_time = 32
        self.ch_pos = n_emb_pos*3*2
        # self.ch_pos = 3
        self.ch_time = n_emb_time*2
        # self.ch_time = self.emb_time
        self.deform = MLP(self.ch_pos, self.ch_time, n_dim, n_layer)
        # self.emb_time_net = nn.Sequential(
        #     nn.Linear(self.ch_time, self.emb_time),
        #     nn.ReLU(True),
        #     nn.Linear(self.emb_time, self.emb_time),
        #     nn.ReLU(True),
        # )
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                # nn.init.constant_(m.weight.data, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
        self.deform.apply(weights_init)
        self.coef_pos = self.coef_table(n_emb_pos, start=-4).cuda()
        self.coef_time = self.coef_table(n_emb_time, start=0).cuda()

        self.crit_smooth = nn.SmoothL1Loss()

    def coef_table(self, L, start=-4):
        return 2**torch.arange(start, start+L)*np.pi

    def pos_enc(self, x, coef_table):
        """ 
        x: [..., D]
        sin_table: [L]
        """
        coef = torch.repeat_interleave(coef_table, 2) # 2L: (sin, cos, sin, cos, ...)
        xx = torch.repeat_interleave(x, coef.size(0), dim=-1) * coef.repeat(x.size(-1)).reshape(1, -1) # [..., 2L*D]: (D1*2L, D2*2L)
        ret = xx.clone()
        ret[..., 0::2] = torch.sin(xx[..., 0::2])
        ret[..., 1::2] = torch.cos(xx[..., 1::2])
        return ret

    def forward(self, tar_time, src_time, x, is_loss=False):
        # TODO: DeformationModel
        ''' 
        tar_time: scalar [1]
        scr_time: [1, N_sources]
        x: [N_rays, N_samples, 3]
        input: [N_sources, N_rays, N_samples, 5(tars, srcs, x)]
        output: [N_sources, N_rays, N_samples, 3]
        '''
        original_shape = x.shape[:2]
        N_sources = src_time.size(1)
        xx = x.repeat((N_sources, 1, 1, 1)).cuda() # [N_sources, N_rays, N_samples, 3]
        out_shape = (N_sources, *original_shape) # (N_sources, N_rays, N_samples)
        tar = torch.full((*out_shape, 1), tar_time.item(), device='cuda')  # [N_sources, N_rays, N_samples, 1]
        src = torch.repeat_interleave(src_time.reshape(-1, 1).cuda(), original_shape[0]*original_shape[1], 1)
        # print(src_time.shape, src.shape, out_shape)
        # src = src.reshape(*out_shape, 1)
        

        # out
        tar = self.pos_enc(tar.reshape(-1, 1), self.coef_time)
        # tar = tar.reshape(-1, 1)
        # tar = self.emb_time_net(tar)
        src = self.pos_enc(src.reshape(-1, 1), self.coef_time)
        # src = src.reshape(-1, 1)
        # src = self.emb_time_net(src)
        dx, occ = self.deform(self.pos_enc(xx.reshape(-1, 3), self.coef_pos), tar, src)
        dx = dx.reshape(*out_shape, 3)
        occ = occ.reshape(*out_shape, 1)
        # dx = self.deform(xx.reshape(-1, 3), tar, src).reshape(*out_shape, 3)
        ret = xx+dx
        if is_loss:            
            # Indentity loss
            ret_emb = self.pos_enc(ret.reshape(-1, 3), self.coef_pos)
            # ret_emb = ret.reshape(-1, 3)
            di0, occ0 = self.deform(self.pos_enc(xx.reshape(-1, 3), self.coef_pos), tar, tar)
            # di0 = self.deform(xx.reshape(-1, 3), tar, tar)
            # di1 = torch.cat([src, src, ret_emb], -1)
            di1, occ1 = self.deform(ret_emb, src, src)
            
            loss_i = (torch.norm(di0, dim=-1)).mean() + (torch.norm(di1, dim=-1)).mean() \
                + torch.norm(occ0-1).sum() + torch.norm(occ1-1).sum()

            # bidirectional loss
            # dx_ = torch.cat([src, tar, ret_emb], -1)
            dx_, occ_ = self.deform(ret_emb, src, tar)
            dx_ = dx_.reshape(*out_shape, 3)+dx
            # occ_ = occ_.reshape(*out_shape, 1)
            loss_bi = (occ.squeeze(-1)*torch.norm(dx+dx_, p=1, dim=-1)).mean()
            
            # smooth regularization
            # smooth = torch.norm(dx, p=1, dim=-1).mean() + torch.norm(dx_, p=1, dim=-1).mean()
            # smooth = self.crit_smooth(dx, torch.zeros_like(dx, device='cuda')) + self.crit_smooth(dx_, torch.zeros_like(dx_, device='cuda'))

            # non-trivial
            non_trivial = torch.norm(occ-1, p=1).sum() + torch.norm(occ0-1, p=1).sum()
            return ret, occ, (loss_bi + loss_i + non_trivial)*0.1
            # return ret, occ, 0.

        return ret, occ

    def forward_(self, tar_time, src_time, x, is_loss=False):
        # TODO: DeformationModel
        ''' 
        tar_time: scalar [1]
        scr_time: [1, N_sources]
        x: [N_rays, N_samples, 3]

        input: [N_sources, N_rays, N_samples, 1(time)]
        output: [N_sources, N_rays, N_samples, 3]
        '''
        N_sources = src_time.size(1)
        original_shape = x.shape[:2]
        x = x.cuda().reshape(-1, 3)

        # x_tar to x_canonical
        tar = torch.full((x.size(0), 1), tar_time.item(), device='cuda')  # [N_rays, N_samples, 1]
        dx_t2can = self.deform(torch.cat([self.pos_enc(x, self.coef_pos), self.pos_enc(tar, self.coef_time)], -1).reshape(-1, self.ch_in))
        x_canonical = x + dx_t2can # [N_rays*N_samples, 3]

        # x_canonical to x_src's
        src = torch.repeat_interleave(src_time.reshape(-1, 1).cuda(), original_shape[0]*original_shape[1], 1).reshape(-1, 1)
        # [N_sources*N_rays*N_samples, 1]
        xx = x_canonical.repeat((N_sources, 1, 1)).reshape(-1, 3) # [N_sources*N_rays*N_samples, 3]
        dx_can2s = self.deform_inv(torch.cat([self.pos_enc(xx, self.coef_pos), self.pos_enc(src, self.coef_time)], -1))
        x_src = xx + dx_can2s # [N_sources*N_rays*N_samples, 3]

        ret = x_src.reshape(N_sources, *original_shape, 3) # [N_sources, N_rays, N_samples, 3]

        if is_loss:
            # smooth regularization
            smooth = torch.norm(dx_t2can, dim=-1).mean() + torch.norm(dx_can2s, dim=-1).mean()
            
            # bidirectional loss
            bi_loss1 = torch.norm(self.deform_inv(torch.cat([
                self.pos_enc(x_canonical, self.coef_pos), self.pos_enc(tar.reshape(-1, 1), self.coef_time)
                ], -1)) + dx_t2can, dim=-1).mean()
            
            bi_loss2 = torch.norm(self.deform(torch.cat([
                self.pos_enc(x_src, self.coef_pos), self.pos_enc(src, self.coef_time)
                ], -1)) + dx_can2s, dim=-1).mean()

            return ret, smooth + bi_loss1 + bi_loss2
        
        return ret


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            # attn = attn * mask

        attn = F.softmax(attn, dim=-1)
        # attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        # x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # q = self.dropout(self.fc(q))
        q = self.fc(q)
        q += residual

        q = self.layer_norm(q)

        return q, attn


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


# default tensorflow initialization of linear layers
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


# @torch.jit.script
def fused_mean_variance(x, weight):
    mean = torch.sum(x*weight, dim=2, keepdim=True)
    var = torch.sum(weight * (x - mean)**2, dim=2, keepdim=True)
    return mean, var


class IBRNet(nn.Module):
    def __init__(self, args, in_feat_ch=32, n_samples=64, **kwargs):
        super(IBRNet, self).__init__()
        self.args = args
        self.anti_alias_pooling = args.anti_alias_pooling
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)
        self.n_samples = n_samples
        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, in_feat_ch + 3),
                                        activation_func)

        self.base_fc = nn.Sequential(nn.Linear((in_feat_ch+3)*3, 64),
                                     activation_func,
                                     nn.Linear(64, 32),
                                     activation_func)

        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    activation_func,
                                    nn.Linear(32, 33),
                                    activation_func,
                                    )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     activation_func,
                                     nn.Linear(32, 1),
                                     nn.Sigmoid()
                                     )

        self.geometry_fc = nn.Sequential(nn.Linear(32*2+1, 64),
                                         activation_func,
                                         nn.Linear(64, 16),
                                         activation_func)

        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16),
                                             activation_func,
                                             nn.Linear(16, 1),
                                             nn.ReLU())

        self.rgb_fc = nn.Sequential(nn.Linear(32+1+4, 16),
                                    activation_func,
                                    nn.Linear(16, 8),
                                    activation_func,
                                    nn.Linear(8, 1))

        self.pos_encoding = self.posenc(d_hid=16, n_samples=self.n_samples)

        self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.vis_fc.apply(weights_init)
        self.geometry_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)

    def posenc(self, d_hid, n_samples):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).to("cuda:{}".format(self.args.local_rank)).float().unsqueeze(0)
        return sinusoid_table

    def forward(self, rgb_feat, ray_diff, mask, occ):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param ray_diff: ray direction difference [n_rays, n_samples, n_views, 4], first 3 channels are directions,
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''

        num_views = rgb_feat.shape[2]
        direction_feat = self.ray_dir_fc(ray_diff)
        rgb_in = rgb_feat[..., :3]
        rgb_feat = rgb_feat + direction_feat
        if self.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]) * mask
            weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8)
        else:
            weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)

        # compute mean and variance across different views for each point
        mean, var = fused_mean_variance(rgb_feat, weight)  # [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean, var], dim=-1)  # [n_rays, n_samples, 1, 2*n_feat]

        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1), rgb_feat], dim=-1)  # [n_rays, n_samples, n_views, 3*n_feat]
        x = self.base_fc(x)

        x_vis = self.vis_fc(x * weight)
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1]-1, 1], dim=-1)
        vis = F.sigmoid(vis) * mask
        x = x + x_res
        vis = self.vis_fc2(x * vis) * mask
        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

        mean, var = fused_mean_variance(x, weight)
        globalfeat = torch.cat([mean.squeeze(2), var.squeeze(2), weight.mean(dim=2)], dim=-1)  # [n_rays, n_samples, 32*2+1]
        globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, 16]
        num_valid_obs = torch.sum(mask, dim=2)
        globalfeat = globalfeat + self.pos_encoding
        globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat,
                                           mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        sigma_out = sigma.masked_fill(num_valid_obs < 1, 0.)  # set the sigma of invalid point to zero

        # rgb computation
        x = torch.cat([x, vis, ray_diff], dim=-1)
        x = self.rgb_fc(x)
        x = x.masked_fill(mask == 0, -1e9)
        # TODO: x*disoclusion
        blending_weights_valid = F.softmax(x*occ, dim=2)  # color blending
        rgb_out = torch.sum(rgb_in*blending_weights_valid, dim=2) # sum of N views' colors
        out = torch.cat([rgb_out, sigma_out], dim=-1)
        return out

if __name__ == '__main__':
    a = DeformationModel()
    a.load_state_dict(torch.load('/home/csvt32745/IBRNet/out/finetune_llff/model_260000_deform.pth'))
    b = DeformationModel()
    b.load_state_dict(torch.load('/home/csvt32745/IBRNet/out/finetune_llff/model_285000_deform.pth'))
    print(((list(a.parameters())[0]-list(b.parameters())[0])**2).sum())
    