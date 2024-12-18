from torch import nn
import torch.nn.functional as F

class DownBlock2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=4, pool=False):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size)
        self.pool = pool
    def forward(self, x):
        out = x
        out = self.conv(out)
        out = F.leaky_relu(out, 0.2)
        if self.pool:
            out = F.avg_pool2d(out, (2, 2))
        return out


class Discriminator(nn.Module):
    """
    Discriminator for GAN loss
    """
    def __init__(self, num_channels, block_expansion=64, num_blocks=4, max_features=512):
        super(Discriminator, self).__init__()
        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(num_channels  if i == 0 else min(max_features, block_expansion * (2 ** i)),
                            min(max_features, block_expansion * (2 ** (i + 1))),
                            kernel_size=4, pool=(i != num_blocks - 1)))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv2d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)
    def forward(self, x):
        feature_maps = []
        out = x
        for down_block in self.down_blocks:
            feature_maps.append(down_block(out))
            out = feature_maps[-1]
        out = self.conv(out)
        return feature_maps, out
