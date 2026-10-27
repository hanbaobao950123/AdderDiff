import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from spiking_layer import LIFSpike, ExpandTime, RateEncoding, singleLIF, LIFAct, LIFAct_nolearned_th


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch, Timestep):
        super().__init__()
        # self.main = nn.AvgPool2d(kernel_size=2, stride=2)
        self.act = LIFAct(T=Timestep)
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(in_ch)
        self.initialize()

    def initialize(self):
        pass

    def forward(self, x, temb):
        # print(x.shape,1111)
        x = self.act(x)
        x = self.main(x)
        x = self.bn(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch, Timestep):
        super().__init__()
        self.act = LIFAct(T=Timestep)
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(in_ch)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        # print(x.shape, 2222)
        x = self.act(x)
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        x = self.bn(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch, Timestep):
        super().__init__()
        # self.group_norm = nn.BatchNorm2d(in_ch)
        # self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        # self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        # self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()
        self.acth = LIFAct(T=Timestep)
        self.actq = LIFAct(T=Timestep)
        self.actk = LIFAct(T=Timestep)
        # self.actv = LIFAct(T=Timestep)
        self.actq_sum = LIFAct(T=Timestep)
        # self.act_h = LIFAct(T=Timestep)
        self.qbn = nn.BatchNorm2d(in_ch)
        self.kbn = nn.BatchNorm2d(in_ch)
        # self.vbn = nn.BatchNorm2d(in_ch)
        # self.bn = nn.BatchNorm2d(in_ch)
        # self.bn = nn.GroupNorm(32, in_ch)
    def initialize(self):
        for module in [self.proj_q, self.proj_k]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        # init.xavier_uniform_(self.proj.weight, gain=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.acth(x)
        q = self.proj_q(h)
        k = self.proj_k(h)

        q = self.qbn(q)
        k = self.kbn(k)


        q = self.actq(q)



        q = torch.sum(q, dim=1, keepdim=True)
        attn = self.actq_sum(q)
        h = torch.mul(attn, k) #.view(B, C, H, W)
        # h = self.act_h(h)
        # h = self.proj(h)
        # h = self.bn(h)

        return x + h


# class AttnBlock(nn.Module):
#     def __init__(self, in_ch, Timestep):
#         super().__init__()
#         # self.group_norm = nn.BatchNorm2d(in_ch)
#         self.group_norm = nn.GroupNorm(32, in_ch)
#         self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
#         self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
#         self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
#         self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
#         self.initialize()
#         self.acth = LIFAct(T=Timestep)
#         self.actq = LIFAct(T=Timestep)
#         self.actk = LIFAct(T=Timestep)
#         self.actv = LIFAct(T=Timestep)
#         self.actlast = LIFAct(T=Timestep)
#         self.qbn = nn.BatchNorm2d(in_ch)
#         self.kbn = nn.BatchNorm2d(in_ch)
#         self.vbn = nn.BatchNorm2d(in_ch)
#         # self.bn = nn.BatchNorm2d(in_ch)
#         self.bn = nn.GroupNorm(32, in_ch)
#
#     def initialize(self):
#         for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
#             init.xavier_uniform_(module.weight)
#             init.zeros_(module.bias)
#         init.xavier_uniform_(self.proj.weight, gain=1e-5)
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.group_norm(x)
#         h = self.acth(x)
#         q = self.proj_q(h)
#         k = self.proj_k(h)
#         v = self.proj_v(h)
#         q = self.qbn(q)
#         k = self.kbn(k)
#         v = self.vbn(v)
#
#         q = self.actq(q)
#         q = q.permute(0, 2, 3, 1).view(B, H * W, C)
#
#         # k = self.actk(k)
#         k = k.view(B, C, H * W)
#         # w = torch.bmm(q, k)
#         w = torch.bmm(q, k) * (int(C) ** (-0.5))
#         assert list(w.shape) == [B, H * W, H * W]
#         w = F.softmax(w, dim=-1)
#         # w = F.relu(w)
#
#         v = self.actv(v)
#         v = v.permute(0, 2, 3, 1).view(B, H * W, C)
#         h = torch.bmm(w, v)
#         assert list(h.shape) == [B, H * W, C]
#
#         h = h.view(B, H, W, C).permute(0, 3, 1, 2)
#         h = self.actlast(h)
#         h = self.proj(h)
#         h = self.bn(h)
#
#         return x + h


class Shortcut(nn.Module):
    def __init__(self, in_ch, out_ch, Timestep):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        # 直接使用恒等映射（当输入输出通道相等时）
        if in_ch == out_ch:
            # print("===")
            self.shortcut = nn.Identity()

        # 当输出通道是输入通道2倍时（in_ch = out_ch/2）
        elif out_ch / in_ch == 2:
            # print("duplicate")
            self.shortcut = self._duplicate_channels

        # 当输入通道是输出通道2倍时（in_ch = 2 * out_ch）
        elif in_ch / out_ch == 2:
            # print("halve")
            self.shortcut = self._halve_channels

        # 其他比例使用1x1卷积调整通道
        else:
            # print("conv", in_ch, out_ch)
            # self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
            self.shortcut = nn.Sequential(
                LIFAct(T=Timestep),
                nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
            )

    def _duplicate_channels(self, x):
        """通过复制输入tensor实现通道翻倍"""
        return torch.cat([x, x], dim=1)  # 沿通道维度复制拼接

    def _halve_channels(self, x):
        """将输入通道分为两半后相加实现通道减半"""
        half_ch = self.out_ch
        part1 = x[:, :half_ch, :, :]  # 前半部分
        part2 = x[:, half_ch:, :, :]  # 后半部分
        return 1 / 2 * (part1 + part2)  # 对应通道元素相加

    def forward(self, x):
        return self.shortcut(x)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, Timestep, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            LIFAct(T=Timestep),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            LIFAct(T=Timestep),
            # nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        if in_ch != out_ch:
            self.shortcut = Shortcut(in_ch, out_ch, Timestep)#
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch, Timestep)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-2].weight, gain=1e-5)

    def forward(self, x, temb):

        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class MembraneOutputLayer(nn.Module):
    """
    outputs the last time membrane potential of the LIF neuron with V_th=infty
    """

    def __init__(self, timestep=4) -> None:
        super().__init__()
        self.n_steps = timestep

    def forward(self, x):
        """
        x : (T,N,C,H,W)
        """

        arr = torch.arange(self.n_steps - 1, -1, -1)
        coef = torch.pow(0.8, arr).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(x.device)
        out = torch.sum(x * coef, dim=0)
        return out


class UNet(nn.Module):
    def __init__(self, T, Timestep, img_ch, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.Timestep = Timestep
        self.expand_time = ExpandTime(T=Timestep)

        self.head = nn.Conv2d(img_ch, ch, kernel_size=3, stride=1, padding=1)
        self.headbn = nn.BatchNorm2d(ch)

        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, Timestep=Timestep, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch, Timestep))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, Timestep=Timestep, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, Timestep=Timestep, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, Timestep=Timestep, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch, Timestep))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, img_ch, 3, stride=1, padding=1)
        )
        # self.tail_bn = nn.BatchNorm2d(now_ch)
        # self.tail_swish = Swish()
        # self.tail_conv = nn.Conv2d(now_ch, img_ch, kernel_size=3, stride=1, padding=1)
        # self.T_output_layer = nn.Conv3d(img_ch, img_ch, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0))
        # self.last_bn = nn.BatchNorm2d(img_ch)
        # self.swish = Swish()
        self.membrane_output_layer = MembraneOutputLayer(timestep=self.Timestep)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        x = self.expand_time(x)
        # Timestep embedding
        temb = self.time_embedding(t)
        temb = temb[None, :, :]
        temb = temb.repeat(self.Timestep, 1, 1)
        temb = temb.view(-1, *temb.shape[2:])
        # Downsampling
        h = self.head(x)
        # h = self.headbn(h)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)

        h = self.tail(h)
        # h = self.tail_bn(h)
        # h = self.tail_swish(h)
        # h = self.tail_conv(h)
        # h_temp = h.view(self.Timestep, -1, *h.shape[1:]).permute(1, 2, 3, 4, 0)
        # h_temp = self.T_output_layer(h_temp).permute(4, 0, 1, 2, 3)  # [ T, B, C, H, W]
        # h_temp = self.last_bn(h_temp.flatten(0, 1))#.reshape(T, B, -1, H, W).contiguous()
        # h = self.swish(h_temp) + h  # [ T*B, C, H, W]
        h = h.view(self.Timestep, -1, *h.shape[1:])

        h = self.membrane_output_layer(h)
        # h = h.mean(0)
        assert len(hs) == 0
        return h


if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size,))
    y = model(x, t)
