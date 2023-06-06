import argparse
import os.path

import mindspore
from mindspore import train
from mindspore import nn
from mindspore.train import Model, LossMonitor, CheckpointConfig, ModelCheckpoint

from model.model import get_model
from utils.data import get_CIFA_datasets


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="alexnet", help="训练的模型，alexnet,vgg,resnet", required=False)
    parser.add_argument("--epochs", type=int, default=50, help="训练的轮次", required=False)
    parser.add_argument("--batch", type=int, default=32, help="数据集的批大小", required=False)
    args = parser.parse_args()
    if not os.path.exists(os.path.join("runs", args.model)):
        os.makedirs(os.path.join("runs", args.model))
    return args


if __name__ == '__main__':
    args = get_parser()
    network = get_model(args.model)
    dataset_dir = "datasets/cifar10"
    dataset_train, dataset_val = get_CIFA_datasets(image_size=256, batch_size=args.batch, dataset_dir=dataset_dir)
    dataset_size = dataset_train.get_dataset_size()

    net_loss = nn.CrossEntropyLoss()
    net_opt = nn.Momentum(network.trainable_params(), learning_rate=0.01, momentum=0.9)
    config_ck = CheckpointConfig(save_checkpoint_steps=dataset_size, keep_checkpoint_max=10)
    ckpoint = ModelCheckpoint(prefix=os.path.join("runs", args.model, args.model), config=config_ck)
    model = Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={'accuracy'})
    model.train(1, dataset_train, callbacks=[ckpoint, LossMonitor(dataset_size)])
    acc = model.eval(dataset_val)
    print("{}".format(acc))
