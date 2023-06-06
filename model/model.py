import mindspore.nn
from mindspore import nn, ops
from mindspore.common.initializer import Normal
from mindspore.common.tensor import Tensor
from mindspore.common import initializer
from mindspore import dtype as mstype
from mindspore.common.initializer import TruncatedNormal
from model.block import VggBlock


class LeNet5(nn.Cell):
    """LeNet5"""
    def __init__(self, num_classes=10, num_channel=3, include_top=True):
        super(LeNet5, self).__init__()
        self.include_top = include_top

        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
            self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
            self.fc3 = nn.Dense(84, num_classes, weight_init=Normal(0.02))

    def construct(self, x):
        """
        LeNet5 construct.
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if self.include_top:
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
        return x


class AlexNet(nn.Cell):
    def __init__(self, num_classes=10, num_channels=3):
        super(AlexNet, self).__init__()

        self.features = nn.SequentialCell(
            nn.Conv2d(num_channels, 64, kernel_size=11, stride=4, pad_mode='valid'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, pad_mode='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, pad_mode='same'),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, pad_mode='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, pad_mode='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.SequentialCell(
            nn.Dense(256 * 6 * 6, 4096, weight_init=TruncatedNormal(0.02)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Dense(4096, 4096, weight_init=TruncatedNormal(0.02)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Dense(4096, num_classes, weight_init=TruncatedNormal(0.02))
        )

    def construct(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class VGG16(nn.Cell):
    def __init__(self, num_classes=10, num_channels=3):
        super(VGG16, self).__init__()
        self.features = nn.SequentialCell(
            VggBlock(3, 64),
            VggBlock(64, 128),
            VggBlock(128, 256),
            VggBlock(256, 512),
            VggBlock(512, 512),
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.SequentialCell(
            nn.Dense(512 * 8 * 8, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Dense(4096, num_classes)
        )

    def construct(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def conv3x3(in_channels, out_channels, stride=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=padding, pad_mode='pad', has_bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                     pad_mode='pad', has_bias=False)

class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad', has_bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(512 * block.expansion, num_classes, weight_init=init.normal(0.02))

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.SequentialCell([
                conv1x1(self.in_channels, channels * block.expansion, stride),
                nn.BatchNorm2d(channels * block.expansion)
            ])

        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


def get_model(name):
    if name == "lenet":
        return LeNet5()
    if name == "alexnet":
        return AlexNet()
    if name == "vgg16":
        return VGG16()


if __name__ == '__main__':
    x = Tensor(shape=(1, 3, 256, 256), dtype=mstype.float32, init=initializer.Normal())
    le_net = LeNet5()
    alex_net = AlexNet()
    vgg = VGG16()
    # print(le_net)
    # print(le_net(x).shape)
    print(alex_net)
    print(alex_net(x).shape)
    print(vgg(x).shape)

