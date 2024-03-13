import torch
import torch.nn as nn
from model.swin import SwinTransformerBlock


class ConvBNSiLU(nn.Module):
    """ Conv+BN+SiLU """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, res=True):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.silu = nn.SiLU()
        self.res = res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.res:
            return x + self.silu(self.bn(self.conv(x)))
        else:
            return self.silu(self.bn(self.conv(x)))


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, (x_HL, x_LH, x_HH)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IDWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class LIW(nn.Module):
    def __init__(self, input_resolution=256, in_channel=64, middle_channel=32, level=2, num_heads=4, window_size=4):
        super().__init__()
        assert in_channel >= 4, "LIW in_channel error!"
        self.in_C = in_channel
        self.in_R = input_resolution
        self.level = level
        self.num_head = num_heads
        self.window_size = window_size
        self.middle_C = middle_channel

        self.dwt = DWT()
        self.idwt = IDWT()

        swin_list = []
        for i in range(self.level):
            swin_list.append(
                SwinTransformerBlock(self.in_C if self.level == 1 else self.middle_C,
                                     (self.in_R // (2 ** (i + 1)), self.in_R // (2 ** (i + 1))),
                                     num_heads=self.num_head, window_size=self.window_size))
        self.swin_blocks = nn.ModuleList(swin_list)
        self.conv = ConvBNSiLU(self.in_C, self.middle_C, 1, 1, False)

        up_conv_list = []
        for i in range(self.level - 2):
            up_conv_list.append(ConvBNSiLU(self.middle_C, self.middle_C))
        up_conv_list.append(ConvBNSiLU(self.in_C if self.level == 1 else self.middle_C, self.in_C, 3, 1, False))
        self.up_conv_blocks = nn.ModuleList(up_conv_list)
        self.conv_out = ConvBNSiLU(self.in_C, self.in_C, 1, 1, False)

    def forward(self, x):
        H_list = []
        LL, (HL, LH, HH) = self.dwt(x)
        H_list.append((HL, LH, HH))

        y = self.conv(LL) if self.level > 1 else LL
        # y = self.conv(LL)
        for i, m in enumerate(self.swin_blocks):
            B, C, H, W = y.shape  # [B, C', H, W]
            y = y.view(B, -1, C)  # [B, H*W, C']
            y = m(y).view(B, C, H, W)  # [B, C', H, W]

            if i != (len(self.swin_blocks) - 1):
                y, (HL, LH, HH) = self.dwt(y)
                H_list.append((HL, LH, HH))

        h = H_list.pop()
        y = self.idwt(torch.cat((y, *h), dim=1))

        for i, m in enumerate(self.up_conv_blocks):
            y = m(y)
            if self.level > 1:
                h = H_list.pop()
                y = self.idwt(torch.cat((y, *h), dim=1))

        y = self.conv_out(y)

        return x + y
