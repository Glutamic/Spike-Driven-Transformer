import torch.nn as nn
from timm.models.layers import DropPath
import torch
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)


class Erode(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )

    def forward(self, x):
        return self.pool(x)
    


class MyConv2D(nn.Module):
    """
    2D-convolutional layer that can be reparameterized into skip (see Eq. 6 of paper).

    Args:
        nf (int): The number of output channels.
        nx (int): The number of input channels.
        resid_gain (float): Residual weight.
        skip_gain (float): Skip weight, if None then defaults to standard Conv2d layer.
        trainable_gains (bool): Whether or not gains are trainable.
        init_type (one of ["orth", "id", "normal"]): Type of weight initialization.
        bias (bool): Whether or not to use bias parameters.
    """

    def __init__(
        self,
        nf,
        nx,
        resid_gain=None,
        skip_gain=None,
        trainable_gains=False,
        init_type="normal",
        bias=True,
    ):
        super().__init__()
        self.nf = nf

        if bias:
            self.bias = nn.Parameter(torch.zeros(nf))
        else:
            self.bias = nn.Parameter(torch.zeros(nf), requires_grad=False)

        if skip_gain is None:
            # Standard convolutional layer
            self.weight = nn.Parameter(torch.empty(nf, nx, 1, 1))
            if init_type == "orth":
                nn.init.orthogonal_(self.weight.view(nf, nx))
            elif init_type == "id":
                self.weight.data = torch.eye(nx).view(nf, nx, 1, 1)
            elif init_type == "normal":
                nn.init.normal_(self.weight, std=0.02)
            else:
                raise NotImplementedError
            self.skip = False

        elif skip_gain is not None:
            # Reparameterized convolutional layer
            assert nx == nf
            self.resid_gain = nn.Parameter(
                torch.Tensor([resid_gain]), requires_grad=trainable_gains
            )
            self.skip_gain = nn.Parameter(
                torch.Tensor([skip_gain]),
                requires_grad=trainable_gains,
            )

            self.weight = nn.Parameter(torch.zeros(nf, nx, 1, 1))
            if init_type == "orth":
                self.id = nn.init.orthogonal_(torch.empty(nx, nx)).cuda().view(nf, nx, 1, 1)
            elif init_type == "id":
                self.id = torch.eye(nx).cuda().view(nf, nx, 1, 1)
            elif init_type == "normal":
                self.id = nn.init.normal_(
                    torch.empty(nx, nx), std=1 / nx
                ).cuda().view(nf, nx, 1, 1)
            else:
                raise NotImplementedError
            self.skip = True
            self.init_type = init_type

    def forward(self, x):
        size_out = x.size()[:-2] + (self.nf, 1, 1)
        if self.skip:
            if self.resid_gain == 0 and self.init_type == "id":
                x = torch.add(self.bias.view(1, -1, 1, 1), x * self.skip_gain.view(1, -1, 1, 1))
            else:
                x = F.conv2d(
                    x, self.resid_gain * self.weight + self.skip_gain * self.id, self.bias, stride=(1, 1), padding=(0, 0)
                )
        else:
            x = F.conv2d(x, self.weight, self.bias, stride=(1, 1), padding=(0, 0))
        x = x.view(size_out)

        return x



class MS_MLP_Conv(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        spike_mode="lif",
        layer=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.res = in_features == hidden_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        if spike_mode == "lif":
            self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.fc1_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        self.fc2_conv = nn.Conv2d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm2d(out_features)
        if spike_mode == "lif":
            self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.fc2_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        self.c_hidden = hidden_features
        self.c_output = out_features
        self.layer = layer

    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        identity = x

        x = self.fc1_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_fc1_lif"] = x.detach()
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()
        if self.res:
            x = identity + x
            identity = x
        x = self.fc2_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_fc2_lif"] = x.detach()
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

        x = x + identity  # TODO: 可能要删除
        return x, hook


class MS_SSA_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
        simplified=False,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.dvs = dvs
        self.num_heads = num_heads
        self.simplified = simplified
        if dvs:
            self.pool = Erode()
        self.scale = 0.125

        if self.simplified:
            print("--------------------using simplified model--------------------------")
            self.centre_attn = True  # 是否使用中心注意力机制
            self.attn_mat_skip_gain = 1
            self.attn_mat_resid_gain = 1
            self.centre_attn_gain = self.attn_mat_resid_gain
            self.trainable_attn_mat_gains = True
            uniform_causal_attn_mat = torch.ones(
                (dim // num_heads, dim // num_heads), dtype=torch.float32
            ) / torch.arange(1, dim // num_heads + 1).view(-1, 1)
            self.register_buffer(
                "uniform_causal_attn_mat",
                torch.tril(
                    uniform_causal_attn_mat,
                ).view(1, 1, 1, dim // num_heads, dim // num_heads),  # 中心化矩阵是一个下三角矩阵，确保自注意力模型只能看到当前位置及之前的位置，用view变为4维张量。
                persistent=False,
            )
            self.register_buffer(
                "diag",
                torch.eye(dim // num_heads).view(1, 1, 1, dim // num_heads, dim // num_heads),
                persistent=False,
            )
            self.attn_mat_resid_gain = nn.Parameter(
                self.attn_mat_resid_gain * torch.ones((1, self.num_heads, 1, 1)),
                requires_grad=self.trainable_attn_mat_gains,
            )
            self.attn_mat_skip_gain = nn.Parameter(
                self.attn_mat_skip_gain * torch.ones((1, self.num_heads, 1, 1)),  # attn_mat_skip_gain默认为1
                requires_grad=self.trainable_attn_mat_gains,
            )
            self.centre_attn_gain = nn.Parameter(
                self.centre_attn_gain * torch.ones((1, self.num_heads, 1, 1)),  # center_attn_gain默认设置为1，即attn_mat_resid_gain
                requires_grad=self.trainable_attn_mat_gains
                and self.centre_attn_gain != 0,
            )

        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim)
        if spike_mode == "lif":
            self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.q_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm2d(dim)
        if spike_mode == "lif":
            self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.k_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        if not self.simplified:  # 是否使用简化后的模块
            self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        else:
            self.v_conv = nn.Identity()
        self.v_bn = nn.BatchNorm2d(dim)
        if spike_mode == "lif":
            self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.v_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        if spike_mode == "lif":
            self.attn_lif = MultiStepLIFNode(
                tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.attn_lif = MultiStepParametricLIFNode(
                init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )

        self.talking_heads = nn.Conv1d(
            num_heads, num_heads, kernel_size=1, stride=1, bias=False
        )
        if spike_mode == "lif":
            self.talking_heads_lif = MultiStepLIFNode(
                tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.talking_heads_lif = MultiStepParametricLIFNode(
                init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )

        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm2d(dim)

        if spike_mode == "lif":
            self.shortcut_lif = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.shortcut_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        self.mode = mode
        self.layer = layer

    def _myattn(self, q, k, v, x, hook=None):
        query_length, key_length = q.size(-2), k.size(-2)
        dots = q.mul(k)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_qk_before"] = dots
        if self.dvs:
            dots = self.pool(dots)
        attn = dots.sum(dim=-2, keepdim=True)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_qk"] = attn.detach()
        if self.centre_attn:
            post_sm_bias_matrix = (
                self.attn_mat_skip_gain * self.diag[:, :, :, :key_length, :key_length]
            ) - self.centre_attn_gain * (
                self.uniform_causal_attn_mat[
                    :, :, :, key_length - query_length : key_length, :key_length
                ]
            )
            print(post_sm_bias_matrix.shape)
            new_attn_weights = attn + post_sm_bias_matrix
        sn_attn = self.talking_heads_lif(new_attn_weights)
        print(v.shape)
        print(sn_attn.shape)
        x = v.mul(sn_attn)
        if self.dvs:
            x = self.pool(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_x_after_qkv"] = x.detach()
        return x

    
    def _attn(self,q, k, v, hook):
        kv = k.mul(v)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_kv_before"] = kv
        if self.dvs:
            kv = self.pool(kv)
        kv = kv.sum(dim=-2, keepdim=True)
        kv = self.talking_heads_lif(kv)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_kv"] = kv.detach()
        x = q.mul(kv)
        if self.dvs:
            x = self.pool(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_x_after_qkv"] = x.detach()
        return x
    
    def forward(self, x, hook=None):  # TODO: 在forward函数中更改注意力机制
        T, B, C, H, W = x.shape  # (4, 2, 256, 8 ,8) notice that x has extra dim T, and B, C, H, W is same as origin ViT model
        identity = x
        N = H * W  # 在ViT中，图片每一个patch宽高之积就是token的数量N，保持不变，不可分割
        x = self.shortcut_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_first_lif"] = x.detach()

        x_for_qkv = x.flatten(0, 1)
        # print("x_for_qkv: ",x_for_qkv.shape)
        q_conv_out = self.q_conv(x_for_qkv)
        # print("q_conv_out: ",q_conv_out.shape)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, H, W).contiguous()
        # print("q_conv_out: ",q_conv_out.shape)
        q_conv_out = self.q_lif(q_conv_out)

        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_q_lif"] = q_conv_out.detach()
        q = (
            q_conv_out.flatten(3)  # T B C H W -> T B C N (4, 2, 256, 64)
            .transpose(-1, -2)  # T B C N -> T B N C (4, 2, 64, 256)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)  # T B N Head C//h (4, 2, 64, 8, 32)
            .permute(0, 1, 3, 2, 4)  # T B Head N C//h -> T B Head N C//h (4, 2, 8, 64, 32)
            .contiguous()
        )
        # print("q: ", q.shape)  # T B Head N C//h (4, 2, 8, 64, 32)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, H, W).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        if self.dvs:
            k_conv_out = self.pool(k_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_k_lif"] = k_conv_out.detach()
        k = (
            k_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        # print("k: ", k.shape)

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, H, W).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        if self.dvs:
            v_conv_out = self.pool(v_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_v_lif"] = v_conv_out.detach()
        v = (
            v_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )  # T B head N C//h
        # print("v: ", v.shape)

        if not self.simplified:
            x = self._attn(q, k, v, hook)
        else:
            x = self._myattn(q, k, v, hook)
        
        x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous()
        x = (
            self.proj_bn(self.proj_conv(x.flatten(0, 1)))
            .reshape(T, B, C, H, W)
            .contiguous()
        )

        x = x + identity
        return x, v, hook


class MS_Block_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        attn_mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
        simplified=False,
    ):
        super().__init__()
        # 调用了MS_SSA_Conv
        self.attn = MS_SSA_Conv(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            mode=attn_mode,
            spike_mode=spike_mode,
            dvs=dvs,
            layer=layer,
            simplified=simplified,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP_Conv(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
            spike_mode=spike_mode,
            layer=layer,
        )

    def forward(self, x, hook=None):
        # 这里的attn是调用的MS_SSA_Conv的forward函数的输出
        x_attn, attn, hook = self.attn(x, hook=hook)
        x, hook = self.mlp(x_attn, hook=hook)
        return x, attn, hook
