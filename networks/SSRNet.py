import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from torch import nn
from .segformer import *
from .SSRModule import *
from .merit_lib.networks import MaxViT4Out_Small
from timm.models.layers import DropPath, to_2tuple
import math


class DWConvLKA(nn.Module):
    def __init__(self, dim=768):
        super(DWConvLKA, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConvLKA(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class LKABlock(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 linear=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, linear=linear)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        y = x.permute(0, 2, 3, 1)
        y = self.norm1(y)
        y = y.permute(0, 3, 1, 2)
        y = self.attn(y)
        y = self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * y
        y = self.drop_path(y)
        x = x + y

        y = x.permute(0, 2, 3, 1)
        y = self.norm2(y)
        y = y.permute(0, 3, 1, 2)
        y = self.mlp(y)
        y = self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * y
        y = self.drop_path(y)
        x = x + y
        return x


##########################################
#
#         General Decoder Blocks
#
##########################################
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)

        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(
            x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2)
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())

        return x


##########################################
#
#         SSR-Net Decoder Blocks
#
##########################################

class MyDecoderLayerLKA(nn.Module):
    def __init__(
            self, input_size, in_out_chan, head_count, token_mlp_mode, reduction_ratio, n_class=9,
            norm_layer=nn.LayerNorm, is_last=False
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        reduction_ratio = reduction_ratio
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.ag_attn = SSRModule(dims)
            self.ag_attn_norm = nn.LayerNorm(out_dim)
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.ag_attn = SSRModule(dims)
            self.ag_attn_norm = nn.LayerNorm(out_dim)
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        self.layer_lka_1 = LKABlock(dim=out_dim)
        self.layer_lka_2 = LKABlock(dim=out_dim)
        self.before_similarity_values = []
        self.after_similarity_values = []

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def check_and_output_average(self, values_list, name):
        if len(values_list) > 10000:
            avg = torch.mean(torch.tensor(values_list)).item()
            print(f"Average {name} value: {avg}")
            values_list.clear()

    def correlation_coefficient(self, x1, x2):
        mean_x1 = torch.mean(x1, dim=1, keepdim=True)
        mean_x2 = torch.mean(x2, dim=1, keepdim=True)
        std_x1 = torch.std(x1, dim=1, keepdim=True)
        std_x2 = torch.std(x2, dim=1, keepdim=True)

        correlation = torch.mean((x1 - mean_x1) * (x2 - mean_x2) / (std_x1 * std_x2), dim=1)

        return correlation

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exists
            x2 = x2.contiguous()
            b2, h2, w2, c2 = x2.shape
            x2 = x2.view(b2, -1, c2)

            x1_expand = self.x1_linear(x1)
            x2_new = x2.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2)
            x1_expand = x1_expand.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2)

            before_attn_similarity = self.correlation_coefficient(x1_expand, x2_new)
            self.before_similarity_values.append(before_attn_similarity.mean().item())

            self.check_and_output_average(self.before_similarity_values, "before_attention")
            attn_gate = self.ag_attn(x2_new)

            cat_linear_x = x1_expand + attn_gate  # B C H W
            cat_linear_x = cat_linear_x.permute(0, 2, 3, 1)  # B H W C
            cat_linear_x = self.ag_attn_norm(cat_linear_x)  # B H W C

            cat_linear_x = cat_linear_x.permute(0, 3, 1, 2).contiguous()  # B C H W

            tran_layer_1 = self.layer_lka_1(cat_linear_x)
            tran_layer_2 = self.layer_lka_2(tran_layer_1)
            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2),
                                             tran_layer_2.size(1))
            if self.last_layer:
                out = self.last_layer(
                    self.layer_up(tran_layer_2).view(b2, 4 * h2, 4 * w2, -1).permute(0, 3, 1, 2))  # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2)
        else:
            out = self.layer_up(x1)
        return out


class MyDecoderLayer34(nn.Module):
    def __init__(
            self, input_size, in_out_chan, head_count, token_mlp_mode, reduction_ratio, n_class=9,
            norm_layer=nn.LayerNorm, is_last=False
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        reduction_ratio = reduction_ratio
        head_count = head_count
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)
            self.ag_attn = SSRModule(dims)
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)
            self.ag_attn = SSRModule(dims)
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exist
            x2 = x2.contiguous()
            b2, h2, w2, c2 = x2.shape  # B C H W --> B H W C
            x2 = x2.view(b2, -1, c2)  # B C H W --> B (HW) C
            x1_expand = self.x1_linear(x1)  # B N C
            x2_new = x2.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2)  # B (HW) C --> B C H W
            x1_expand = x1_expand.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2)  # B N C --> B C H W
            attn_gate = self.ag_attn(x2_new)  # B C H W
            cat_linear_x = x1_expand + attn_gate  # B C H W
            cat_linear_x = cat_linear_x.permute(0, 2, 3, 1)  # B H W C
            cat_linear_x = self.ag_attn_norm(cat_linear_x)  # B H W C
            cat_linear_x = cat_linear_x.view(b2, -1, c2)  # B H W C --> B (HW) C
            tran_layer_1 = cat_linear_x
            tran_layer_2 = cat_linear_x

            if self.last_layer:

                out = self.last_layer(
                    self.layer_up(tran_layer_2).view(b2, 4 * h2, 4 * w2, -1).permute(0, 3, 1, 2))  # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2)  # 1 3136 160
        else:
            out = self.layer_up(x1)
        return out


import torch
import torch.nn as nn
from thop import profile


class SSRNet(nn.Module):
    def __init__(self, num_classes=9, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder
        self.backbone = MaxViT4Out_Small(n_class=num_classes, img_size=224)

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [96, 96, 96, 96, 96],
            [192, 192, 192, 192, 192],
            [384, 384, 384, 384, 384],
            [768, 768, 768, 768, 768],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]
        reduction_ratio = [16, 8, 6, 2]
        head_count = [32, 16, 1, 1]

        self.decoder_3 = MyDecoderLayer34(
            (d_base_feat_size, d_base_feat_size),
            in_out_chan[3],
            head_count[0],
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[0])

        self.decoder_2 = MyDecoderLayer34(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            head_count[1],
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[1])
        self.decoder_1 = MyDecoderLayerLKA(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            head_count[2],
            token_mlp_mode,
            n_class=num_classes,
            reduction_ratio=reduction_ratio[2])
        self.decoder_0 = MyDecoderLayerLKA(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            head_count[3],
            token_mlp_mode,
            n_class=num_classes,
            is_last=True,
            reduction_ratio=reduction_ratio[3])


    def forward(self, x):
        # ---------------Encoder-------------------------

        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc_3, output_enc_2, output_enc_1, output_enc_0 = self.backbone(x)

        b, c, _, _ = output_enc_3.shape
        # ---------------Decoder-------------------------
        tmp_3 = self.decoder_3(output_enc_3.permute(0, 2, 3, 1).view(b, -1, c))
        tmp_2 = self.decoder_2(tmp_3, output_enc_2.permute(0, 2, 3, 1))
        tmp_1 = self.decoder_1(tmp_2, output_enc_1.permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc_0.permute(0, 2, 3, 1))

        return tmp_0