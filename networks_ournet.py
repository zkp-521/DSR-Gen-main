from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
import copy
import math
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

class LayerNorm1(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm1, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm1(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size##paraphrase

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Embeddings_mid(nn.Module): #Construct the embeddings from patch, position embeddings.
    def __init__(self, img_size=(256, 256),hidden_size=768, in_channels=512):
        super(Embeddings_mid, self).__init__()
        img_size = _pair(img_size)
        patch_size = (img_size[0] // 8 // 16, img_size[1] // 8 // 16)
        patch_size_real = (patch_size[0] * 8, patch_size[1] * 8)
        n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
        self.dropout = Dropout(0.1)

    def forward(self, x):
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Transformer_mid(nn.Module):
    def __init__(self, config, img_size, vis,in_channels=1):
        super(Transformer_mid, self).__init__()
        self.embeddings = Embeddings_mid(img_size=img_size, hidden_size=config.hidden_size, in_channels=in_channels)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output= self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights
class channel_compression(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(channel_compression, self).__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
          self.skip = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))
        else:
          self.skip = None

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.block(x)
        out += (x if self.skip is None else self.skip(x))
        out = F.relu(out)
        return out
class ART_block_mid(nn.Module):
    def __init__(self, hidden_size,ngf=64,transformer=None):
        super(ART_block_mid, self).__init__()
        self.transformer = transformer
        use_bias = False
        norm_layer = nn.BatchNorm2d
        model = [nn.ConvTranspose2d(hidden_size, ngf,
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        setattr(self, 'upsample', nn.Sequential(*model))
        self.cc = channel_compression(ngf * 2, ngf)

    def forward(self, x):

        # feed to transformer
        transformer_out, attn_weights = self.transformer(x)
        B, n_patch, hidden = transformer_out.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        transformer_out = transformer_out.permute(0, 2, 1)
        transformer_out = transformer_out.contiguous().view(B, hidden, h, w)
        transformer_out=self.upsample(transformer_out)

        # concat transformer output and resnet output
        x = torch.cat([transformer_out, x], dim=1)
        # channel compression
        x = self.cc(x)
        return x


class Decoder_zijiandu_addall_cha2_weight_share(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim,config, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder_zijiandu_addall_cha2_weight_share, self).__init__()

        self.n_upsample = n_upsample
        self.model_rec = []
        self.up_rec = []
        self.model_tr = []
        self.up_tr = []
        self.config = config

        self.Transformer = Transformer_mid(config=self.config, img_size=(256, 256), vis=False, in_channels=dim)
        self.mid_tform = ART_block_mid(hidden_size=self.config.hidden_size,ngf=dim,transformer=self.Transformer)
        self.model_rec += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        self.model_tr += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # self.model_rec_tr += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model_rec += [Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            self.up_rec += [nn.Upsample(scale_factor=2)]
            self.model_tr += [Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            self.up_tr += [nn.Upsample(scale_factor=2)]
            dim //= 2

        # use reflection padding in the last conv layer
        # self.model += []
        self.output_rec = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)
        self.output_tr = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)
        self.model_rec = nn.Sequential(*self.model_rec)
        self.up_rec = nn.Sequential(*self.up_rec)
        self.model_tr = nn.Sequential(*self.model_tr)
        self.up_tr = nn.Sequential(*self.up_tr)

        #self.model_rec_tr = nn.Sequential(*self.model_rec_tr)

    def forward(self, skips):
        lres_input = skips[-1]
        mid_fre=self.mid_tform(lres_input)
        x_rec = self.model_rec[0](mid_fre)
        x_tr = self.model_tr[0](mid_fre)
        #x_tr_rec = self.model_rec[0](lres_input)
        x_tr_rec=x_rec

        for i in range(self.n_upsample):
            # print(x_rec.shape)
            x_rec = x_rec + skips[-(i + 2)]
            x_tr_rec = x_tr_rec + x_tr + skips[-(i + 2)]
            x_rec = self.up_rec[i](x_rec)
            x_rec = self.model_rec[i + 1](x_rec)
            x_tr_rec = self.up_rec[i](x_tr_rec)
            x_tr_rec = self.model_rec[i + 1](x_tr_rec)

            x_tr = self.up_tr[i](x_tr+skips[-(i + 2)])
            x_tr = self.model_tr[i + 1](x_tr)

        x_rec = self.output_rec(x_rec)
        x_tr_rec = self.output_rec(x_tr + x_tr_rec)
        x_tr = self.output_tr(x_tr)

        return x_rec, x_tr, x_tr_rec
class Decoder_zijiandu_xiaorong_resvit(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim,config, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder_zijiandu_xiaorong_resvit, self).__init__()

        self.n_upsample = n_upsample
        self.model_rec = []
        self.up_rec = []
        self.model_tr = []
        self.up_tr = []
        self.config = config

        self.Transformer = Transformer_mid(config=self.config, img_size=(256, 256), vis=False, in_channels=dim)
        self.mid_tform = ART_block_mid(hidden_size=self.config.hidden_size,ngf=dim,transformer=self.Transformer)
        self.model_rec += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        #self.model_tr += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # self.model_rec_tr += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model_rec += [Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            self.up_rec += [nn.Upsample(scale_factor=2)]
            #self.model_tr += [Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            #self.up_tr += [nn.Upsample(scale_factor=2)]
            dim //= 2

        # use reflection padding in the last conv layer
        # self.model += []
        self.output_rec = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)
        #self.output_tr = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)
        self.model_rec = nn.Sequential(*self.model_rec)
        self.up_rec = nn.Sequential(*self.up_rec)
        #self.model_tr = nn.Sequential(*self.model_tr)
        #self.up_tr = nn.Sequential(*self.up_tr)

        #self.model_rec_tr = nn.Sequential(*self.model_rec_tr)

    def forward(self, skips):
        lres_input = skips[-1]
        mid_fre=self.mid_tform(lres_input)
        x_rec = self.model_rec[0](mid_fre)
        #x_tr = self.model_tr[0](mid_fre)
        #x_tr_rec = self.model_rec[0](lres_input)
        #x_tr_rec=x_rec

        for i in range(self.n_upsample):
            # print(x_rec.shape)
            x_rec = x_rec + skips[-(i + 2)]
            #x_tr_rec = x_tr_rec + x_tr + skips[-(i + 2)]
            x_rec = self.up_rec[i](x_rec)
            x_rec = self.model_rec[i + 1](x_rec)
            #x_tr_rec = self.up_rec[i](x_tr_rec)
            #x_tr_rec = self.model_rec[i + 1](x_tr_rec)

            #x_tr = self.up_tr[i](x_tr+skips[-(i + 2)])
            #x_tr = self.model_tr[i + 1](x_tr)

        x_rec = self.output_rec(x_rec)
        #x_tr_rec = self.output_rec(x_tr + x_tr_rec)
        #x_tr = self.output_tr(x_tr)

        return x_rec
class Decoder_zijiandu_xiaorong_resvit_shuangfenzhi(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim,config, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder_zijiandu_xiaorong_resvit_shuangfenzhi, self).__init__()

        self.n_upsample = n_upsample
        self.model_rec = []
        self.up_rec = []
        self.model_tr = []
        self.up_tr = []
        self.config = config

        self.Transformer = Transformer_mid(config=self.config, img_size=(256, 256), vis=False, in_channels=dim)
        self.mid_tform = ART_block_mid(hidden_size=self.config.hidden_size,ngf=dim,transformer=self.Transformer)
        self.model_rec += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        self.model_tr += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # self.model_rec_tr += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model_rec += [Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            self.up_rec += [nn.Upsample(scale_factor=2)]
            self.model_tr += [Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            self.up_tr += [nn.Upsample(scale_factor=2)]
            dim //= 2

        # use reflection padding in the last conv layer
        # self.model += []
        self.output_rec = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)
        self.output_tr = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)
        self.model_rec = nn.Sequential(*self.model_rec)
        self.up_rec = nn.Sequential(*self.up_rec)
        self.model_tr = nn.Sequential(*self.model_tr)
        self.up_tr = nn.Sequential(*self.up_tr)

        #self.model_rec_tr = nn.Sequential(*self.model_rec_tr)

    def forward(self, skips):
        lres_input = skips[-1]
        mid_fre=self.mid_tform(lres_input)
        x_rec = self.model_rec[0](mid_fre)
        x_tr = self.model_tr[0](mid_fre)
        #x_tr_rec = self.model_rec[0](lres_input)
        x_tr_rec=x_rec

        for i in range(self.n_upsample):
            # print(x_rec.shape)
            #x_rec = x_rec + skips[-(i + 2)]
            x_tr_rec = x_tr_rec + x_tr + skips[-(i + 2)]
            #x_rec = self.up_rec[i](x_rec)
            #x_rec = self.model_rec[i + 1](x_rec)
            x_tr_rec = self.up_rec[i](x_tr_rec)
            x_tr_rec = self.model_rec[i + 1](x_tr_rec)

            x_tr = self.up_tr[i](x_tr+skips[-(i + 2)])
            x_tr = self.model_tr[i + 1](x_tr)

        #x_rec = self.output_rec(x_rec)
        x_tr_rec = self.output_rec(x_tr + x_tr_rec)
        x_tr = self.output_tr(x_tr)

        return x_tr,x_tr_rec
class Decoder_zijiandu_xiaorong_cnn_shuangfenzhi(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim,config, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder_zijiandu_xiaorong_cnn_shuangfenzhi, self).__init__()

        self.n_upsample = n_upsample
        self.model_rec = []
        self.up_rec = []
        self.model_tr = []
        self.up_tr = []
        self.config = config

        #self.Transformer = Transformer_mid(config=self.config, img_size=(256, 256), vis=False, in_channels=dim)
        #self.mid_tform = ART_block_mid(hidden_size=self.config.hidden_size,ngf=dim,transformer=self.Transformer)
        self.model_rec += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        self.model_tr += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # self.model_rec_tr += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model_rec += [Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            self.up_rec += [nn.Upsample(scale_factor=2)]
            self.model_tr += [Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            self.up_tr += [nn.Upsample(scale_factor=2)]
            dim //= 2

        # use reflection padding in the last conv layer
        # self.model += []
        self.output_rec = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)
        self.output_tr = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)
        self.model_rec = nn.Sequential(*self.model_rec)
        self.up_rec = nn.Sequential(*self.up_rec)
        self.model_tr = nn.Sequential(*self.model_tr)
        self.up_tr = nn.Sequential(*self.up_tr)

        #self.model_rec_tr = nn.Sequential(*self.model_rec_tr)

    def forward(self, skips):
        lres_input = skips[-1]
        #mid_fre=self.mid_tform(lres_input)
        x_rec = self.model_rec[0](lres_input)
        x_tr = self.model_tr[0](lres_input)
        #x_tr_rec = self.model_rec[0](lres_input)
        x_tr_rec=x_rec

        for i in range(self.n_upsample):
            # print(x_rec.shape)
            #x_rec = x_rec + skips[-(i + 2)]
            x_tr_rec = x_tr_rec + x_tr + skips[-(i + 2)]
            #x_rec = self.up_rec[i](x_rec)
            #x_rec = self.model_rec[i + 1](x_rec)
            x_tr_rec = self.up_rec[i](x_tr_rec)
            x_tr_rec = self.model_rec[i + 1](x_tr_rec)

            x_tr = self.up_tr[i](x_tr+skips[-(i + 2)])
            x_tr = self.model_tr[i + 1](x_tr)

        #x_rec = self.output_rec(x_rec)
        x_tr_rec = self.output_rec(x_tr + x_tr_rec)
        x_tr = self.output_tr(x_tr)

        return x_tr,x_tr_rec
class Decoder_zijiandu_addallcnn_weight_share(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder_zijiandu_addallcnn_weight_share, self).__init__()

        self.n_upsample = n_upsample
        self.model_rec = []
        self.up_rec = []
        self.model_tr = []
        self.up_tr = []


        self.model_rec += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        self.model_tr += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # self.model_rec_tr += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model_rec += [Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            self.up_rec += [nn.Upsample(scale_factor=2)]
            self.model_tr += [Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            self.up_tr += [nn.Upsample(scale_factor=2)]
            dim //= 2

        # use reflection padding in the last conv layer
        # self.model += []
        self.output_rec = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)
        self.output_tr = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)
        self.output_rec_tr = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)
        self.model_rec = nn.Sequential(*self.model_rec)
        self.up_rec = nn.Sequential(*self.up_rec)
        self.model_tr = nn.Sequential(*self.model_tr)
        self.up_tr = nn.Sequential(*self.up_tr)

        #self.model_rec_tr = nn.Sequential(*self.model_rec_tr)

    def forward(self, skips):
        lres_input = skips[-1]
        x_rec = self.model_rec[0](lres_input)
        x_tr = self.model_tr[0](lres_input)
        #x_tr_rec = self.model_rec[0](lres_input)
        x_tr_rec=x_rec

        for i in range(self.n_upsample):
            # print(x_rec.shape)
            x_rec = x_rec + skips[-(i + 2)]
            x_tr_rec = x_tr_rec + x_tr + skips[-(i + 2)]
            x_rec = self.up_rec[i](x_rec)
            x_rec = self.model_rec[i + 1](x_rec)
            x_tr_rec = self.up_rec[i](x_tr_rec)
            x_tr_rec = self.model_rec[i + 1](x_tr_rec)

            x_tr = self.up_tr[i](x_tr+skips[-(i + 2)])
            x_tr = self.model_tr[i + 1](x_tr)

        x_rec = self.output_rec(x_rec)
        x_tr_rec = self.output_rec_tr(x_tr + x_tr_rec)
        x_tr = self.output_tr(x_tr)

        return x_rec, x_tr, x_tr_rec

class Decoder_zijiandu_addallcnn_x(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder_zijiandu_addallcnn_x, self).__init__()

        self.n_upsample = n_upsample
        self.model_rec = []
        self.up_rec = []
        self.model_tr = []
        self.up_tr = []

        self.model_rec += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        self.model_tr += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # self.model_rec_tr += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model_rec += [Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            self.up_rec += [nn.Upsample(scale_factor=2)]
            self.model_tr += [Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            self.up_tr += [nn.Upsample(scale_factor=2)]
            dim //= 2


        #self.output_rec = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)
        self.output_tr = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)
        self.output_rec_tr = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)
        self.model_rec = nn.Sequential(*self.model_rec)
        self.up_rec = nn.Sequential(*self.up_rec)
        self.model_tr = nn.Sequential(*self.model_tr)
        self.up_tr = nn.Sequential(*self.up_tr)


    def forward(self, skips):
        lres_input = skips[-1]
        x_tr_rec = self.model_rec[0](lres_input)
        x_tr = self.model_tr[0](lres_input)
        #x_tr_rec = self.model_rec[0](lres_input)


        for i in range(self.n_upsample):
            # print(x_rec.shape)

            x_tr_rec = x_tr_rec + x_tr + skips[-(i + 2)]
            x_tr_rec = self.up_rec[i](x_tr_rec)
            x_tr_rec = self.model_rec[i + 1](x_tr_rec)

            x_tr = self.up_tr[i](x_tr+skips[-(i + 2)])
            x_tr = self.model_tr[i + 1](x_tr)

        x_tr_rec = self.output_rec_tr(x_tr + x_tr_rec)
        x_tr = self.output_tr(x_tr)
        return x_tr, x_tr_rec

class ContentEncoder_zijiandu(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder_zijiandu, self).__init__()
        self.n_downsample=n_downsample
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim


    def forward(self, x):
        ret = []
        for s in range(self.n_downsample+2):
            # print('x.shape', x.shape, s)
            x = self.model[s](x)
            ret.append(x)
        return ret

class Model(nn.Module):#our_model
    def __init__(self, config,input_dim=1, dim=64,n_downsample=3,n_res=3,activ='relu',pad_type='reflect'):
        super(Model, self).__init__()
        self.enc_content = ContentEncoder_zijiandu(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)

        self.dec =Decoder_zijiandu_addall_cha2_weight_share(n_downsample, n_res, self.enc_content.output_dim, input_dim, config=config,
                                                   res_norm='in', activ=activ, pad_type=pad_type)
    def encode(self, images):
        content = self.enc_content(images)
        return content

    def decode(self, content):
        rec,tr,rectr= self.dec(content)
        return rec,tr,rectr

class Model_cnn(nn.Module):#Attention mechanisms are not integrated into the proposed model
    def __init__(self,input_dim=1, dim=64,n_downsample=3,n_res=3,activ='relu',pad_type='reflect'):
        super(Model_cnn, self).__init__()
        self.enc_content = ContentEncoder_zijiandu(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)

        self.dec =Decoder_zijiandu_addallcnn_weight_share(n_downsample, n_res, self.enc_content.output_dim, input_dim,
                                                   res_norm='in', activ=activ, pad_type=pad_type)
    def encode(self, images):
        content = self.enc_content(images)
        return content

    def decode(self, content):
        rec,tr,rectr= self.dec(content)
        return rec,tr,rectr

class Model_cnn_X(nn.Module): #Our model does not adopt weight sharing and Attention mechanisms
    def __init__(self,input_dim=1, dim=64,n_downsample=3,n_res=3,activ='relu',pad_type='reflect'):
        super(Model_cnn_X, self).__init__()
        self.enc_content = ContentEncoder_zijiandu(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec =Decoder_zijiandu_addallcnn_x(n_downsample, n_res, self.enc_content.output_dim, input_dim,
                                                   res_norm='in', activ=activ, pad_type=pad_type)
    def encode(self, images):
        content = self.enc_content(images)
        return content

    def decode(self, content):
        tr,rectr= self.dec(content)
        return tr,rectr


class Model_xiaorong_resvit(nn.Module): #Ablation Variant 1
    def __init__(self, config,input_dim=1, dim=64,n_downsample=3,n_res=3,activ='relu',pad_type='reflect'):
        super(Model_xiaorong_resvit, self).__init__()
        self.enc_content = ContentEncoder_zijiandu(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)

        self.dec =Decoder_zijiandu_xiaorong_resvit(n_downsample, n_res, self.enc_content.output_dim, input_dim, config=config,
                                                   res_norm='in', activ=activ, pad_type=pad_type)
    def encode(self, images):
        content = self.enc_content(images)
        return content

    def decode(self, content):
        rec= self.dec(content)
        return rec

class Model_xiaorong_resvit_shuangfenzhi(nn.Module): #Ablation Variant 2
    def __init__(self, config,input_dim=1, dim=64,n_downsample=3,n_res=3,activ='relu',pad_type='reflect'):
        super(Model_xiaorong_resvit_shuangfenzhi, self).__init__()
        self.enc_content = ContentEncoder_zijiandu(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec =Decoder_zijiandu_xiaorong_resvit_shuangfenzhi(n_downsample, n_res, self.enc_content.output_dim, input_dim, config=config,
                                                   res_norm='in', activ=activ, pad_type=pad_type)
    def encode(self, images):
        content = self.enc_content(images)
        return content

    def decode(self, content):
        x_tr,x_tr_rec= self.dec(content)
        return x_tr,x_tr_rec
