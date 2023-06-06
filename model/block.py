import mindspore
from mindspore import nn
from mindspore import dtype as mstype
from mindspore import Tensor
from mindspore.common import initializer



class VggBlock(nn.Cell):
    def __init__(self, input_channels, output_channels):
        super(VggBlock, self).__init__()
        self.block = nn.SequentialCell(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def construct(self, x):
        return self.block(x)


if __name__ == '__main__':
    x = Tensor(shape=(1, 3, 256, 256), dtype=mstype.float32, init=initializer.Normal())
    block = VggBlock(3, 64)
    blocks = nn.SequentialCell(
            VggBlock(3, 64),
            VggBlock(64, 128),
            VggBlock(128, 256),
            VggBlock(256, 512),
            VggBlock(512, 512),
        )
    print(block(x).shape)
    print(blocks(x).shape)
