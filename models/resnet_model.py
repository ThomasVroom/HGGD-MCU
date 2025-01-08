"""
resnet in pytorch.
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition
https://arxiv.org/abs/1512.03385v1
"""
import torch.nn as nn

# BasicBlock for ResNet18 and ResNet34
class BasicBlock(nn.Module):
    # BasicBlock and BottleNeck have different output sizes
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels * BasicBlock.expansion,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut (not the same output dimension as residual function, so use 1*1 convolution)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * BasicBlock.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        
        # forward activation function
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.act(self.residual_function(x) + self.shortcut(x))

# BottleNeck for ResNet over 50 layers
class BottleNeck(nn.Module):
    # BasicBlock and BottleNeck have different output sizes
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels,
                      stride=stride,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels * BottleNeck.expansion,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        # shortcut (not the same output dimension as residual function, so use 1*1 convolution)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * BottleNeck.expansion,
                          stride=stride,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
        )
            
        # forward activation function
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.act(self.residual_function(x) + self.shortcut(x))

# ResNet model
class ResNet(nn.Module):

    def __init__(self, block, num_blocks, in_dim=3, planes=64):
        super().__init__()
        print('planes:', planes)

        # x4 for BottleNeck to get the same dimensions as BasicBlock
        self.in_channels = planes * 4 if block is BottleNeck else planes

        # initial convolution + batchnorm + activation layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim,
                      out_channels=self.in_channels,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.LeakyReLU(inplace=True)
        )

        # version-specific layers
        self.conv2_x = self._make_layer(block, planes * 2, num_blocks[0], 2)
        self.conv3_x = self._make_layer(block, planes * 4, num_blocks[1], 2)
        self.conv4_x = self._make_layer(block, planes * 8, num_blocks[2], 2)
        self.conv5_x = self._make_layer(block, planes * 16, num_blocks[3], 2)
        embeddings = planes * 16 * block.expansion  # *4 for ResNet with more than 50 layers
        print('embeddings:', embeddings)

    # create a custom ResNet layer:
    def _make_layer(self, block, out_channels, num_block, stride):
        # the stride of the first block can be 1 or 2, other blocks will always be 1
        strides = [stride] + [1] * (num_block - 1)

        # create the ResNet layer
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        o1 = self.conv1(x)
        o2 = self.conv2_x(o1)
        o3 = self.conv3_x(o2)
        o4 = self.conv4_x(o3)
        o5 = self.conv5_x(o4)
        return [o1, o2, o3, o4, o5]

# predefined models
def resnet18(**kwargs):
    """return a ResNet 18 object."""
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
def resnet34(**kwargs):
    """return a ResNet 34 object."""
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
def resnet50(**kwargs):
    """return a ResNet 50 object."""
    return ResNet(BottleNeck, [3, 4, 6, 3], **kwargs)
def resnet101(**kwargs):
    """return a ResNet 101 object."""
    return ResNet(BottleNeck, [3, 4, 23, 3], **kwargs)
def resnet152(**kwargs):
    """return a ResNet 152 object."""
    return ResNet(BottleNeck, [3, 8, 36, 3], **kwargs)
