import math
import numpy as np
import torch.nn as nn
from .resnet_model import BasicBlock, BottleNeck, ResNet

def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum
    return fn

# BatchNorm momentum scheduler
class BNMomentumScheduler(object):

    def __init__(self, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(type(model).__name__))

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

# ResNet feature extractor
class Backbone(nn.Module):

    # mode = ResNet version
    def __init__(self, in_dim, planes=16, mode='34'):
        super().__init__()
        if mode == '18': # 18 layers
            self.net = ResNet(BasicBlock, [2, 2, 2, 2],
                              in_dim=in_dim,
                              planes=planes)
        elif mode == '34': # 34 layers
            self.net = ResNet(BasicBlock, [3, 4, 6, 3],
                              in_dim=in_dim,
                              planes=planes)
        elif mode == '50': # 50 layers
            self.net = ResNet(BottleNeck, [3, 4, 6, 3],
                              in_dim=in_dim,
                              planes=planes // 4) # div 4 for same channels
        else:
            raise RuntimeError(f'Backbone mode {mode} not found!')

    def forward(self, x):
        x = self.net(x)
        return x

# Convolution + BatchNorm + Activation
class convolution(nn.Module):

    def __init__(self, in_dim, out_dim, kernel, bias=False, stride=1, act=nn.ReLU):
        super(convolution, self).__init__()
        pad = (kernel - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel, stride, pad, bias=bias),
            nn.BatchNorm2d(out_dim),
            act(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# Transposed-Convolution + BatchNorm + Activation
class trconvolution(nn.Module):

    def __init__(self, in_dim, out_dim, padding=1, output_padding=0, act=nn.ReLU):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_dim,
                               out_dim,
                               kernel_size=4,
                               stride=2,
                               padding=padding,
                               output_padding=output_padding,
                               bias=False),
            nn.BatchNorm2d(out_dim),
            act(inplace=True)
        )

    def forward(self, x):
        x = self.deconv(x)
        return x

# Upsampling + Convolution + BatchNorm + Activation
class upsampleconvolution(nn.Module):

    def __init__(self, in_dim, out_dim, output_size=None, act=nn.ReLU):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2) if output_size is None else nn.UpsamplingBilinear2d(size=output_size),
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(out_dim),
            act(inplace=True)
        )

    def forward(self, x):
        x = self.deconv(x)
        return x

# Convolution + Reshape
def conv_with_dim_reshape(in_dim, mid_dim, out_dim, bias=True):
    return nn.Sequential(
        convolution(in_dim, mid_dim, 5, bias=bias),
        nn.Conv2d(mid_dim, out_dim, 1, bias=bias)
    )

# AnchorNet (GHM)
class AnchorGraspNet(nn.Module):

    def __init__(self, feature_dim=128, ratio=1, anchor_k=6, mid_dim=32, use_upsampling=False):
        super(AnchorGraspNet, self).__init__()

        self.ratio = ratio
        self.trconv = nn.ModuleList()

        # stacked transposed-convolutions or upsampling-convolutions (decoder)
        self.depth = 4 # channels: [64, 32, 16, 8, 8]
        channels = [max(8, feature_dim // (2**(i + 1))) for i in range(self.depth + 1)]
        cur_dim = feature_dim
        output_sizes = [(40, 23), (80, 45)] # only for upsampling
        for i, dim in enumerate(channels):
            if use_upsampling:
                if i < min(5 - np.log2(ratio), 2):
                    self.trconv.append(upsampleconvolution(cur_dim, dim, output_sizes[i]))
                else:
                    self.trconv.append(upsampleconvolution(cur_dim, dim))
            else:
                if i < min(5 - np.log2(ratio), 2): # padding is hard-coded (!)
                    self.trconv.append(trconvolution(cur_dim, dim, padding=1, output_padding=0))
                else:
                    self.trconv.append(trconvolution(cur_dim, dim))
            cur_dim = dim

        # heatmap predictor
        cur_dim = channels[self.depth - int(np.log2(self.ratio))]
        self.hmap = conv_with_dim_reshape(channels[-1], mid_dim, 1)
        self.cls_mask_conv = conv_with_dim_reshape(cur_dim, mid_dim, anchor_k)
        self.theta_offset_conv = conv_with_dim_reshape(cur_dim, mid_dim, anchor_k)
        self.width_offset_conv = conv_with_dim_reshape(cur_dim, mid_dim, anchor_k)
        self.depth_offset_conv = conv_with_dim_reshape(cur_dim, mid_dim, anchor_k)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, xs):
        # use trconv or upsampling + conv to get perpoint features
        x = xs[-1]
        for i, layer in enumerate(self.trconv):
            # skip connection (add ResNet features)
            x = layer(x + xs[self.depth - i])

            # down sample classification mask
            if x.shape[1] == 32:
                features = x.detach()
                # print(f"extracted features in layer {i} with shape {features.shape}")
            if int(np.log2(self.ratio)) == self.depth - i:
                cls_mask = self.cls_mask_conv(x)
                theta_offset = self.theta_offset_conv(x)
                width_offset = self.width_offset_conv(x)
                depth_offset = self.depth_offset_conv(x)

        # full scale location map
        loc_map = self.hmap(x)
        return (loc_map, cls_mask, theta_offset, depth_offset, width_offset), features
