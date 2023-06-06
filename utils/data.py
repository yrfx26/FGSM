import os.path

# import download
from mindspore import dataset as ds
import mindspore.dataset.vision as transforms
from mindspore.dataset.vision import Inter


def download_CIFA_10(path="./datasets"):
    url = 'https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/datasets/cifar10.zip'
    path = download(url, "./datasets", kind="zip", replace=True)
    return path


def get_CIFA_dataset_train(image_size=32, batch_size=32, dataset_dir="./datasets", transform=None):
    ds_train_path = os.path.join(dataset_dir, "train")
    dataset_train = ds.Cifar10Dataset(ds_train_path, num_parallel_workers=8, shuffle=True)

    if transform is None:
        transform = [
            transforms.Resize(size=256, interpolation=Inter.LINEAR),
            transforms.Rescale(1.0 / 255.0, 0.0),
            transforms.Rescale(1 / 0.3081, -1 * 0.1307 / 0.3081),
            transforms.HWC2CHW(),
        ]

    dataset_train = dataset_train.map(operations=transform, input_columns=["image"])
    dataset_train = dataset_train.map(operations=lambda x: x.astype("int32"), input_columns=["label"])
    dataset_train = dataset_train.batch(batch_size=batch_size, drop_remainder=True)

    return dataset_train


def get_CIFA_dataset_val(image_size=32, batch_size=32, dataset_dir="./datasets", transform=None):
    dataset_val_path = os.path.join(dataset_dir, "test")
    dataset_val = ds.Cifar10Dataset(dataset_val_path, num_parallel_workers=8, shuffle=False)

    if transform is None:
        transform = [
            transforms.Resize(size=image_size, interpolation=Inter.LINEAR),
            transforms.Rescale(1.0 / 255.0, 0.0),
            transforms.Rescale(1 / 0.3081, -1 * 0.1307 / 0.3081),
            transforms.HWC2CHW(),
        ]

    dataset_val = dataset_val.map(operations=transform, input_columns=["image"])
    dataset_val = dataset_val.map(operations=lambda x: x.astype("int32"), input_columns=["label"])
    dataset_val = dataset_val.batch(batch_size=batch_size, drop_remainder=True)
    return dataset_val


def get_CIFA_datasets(image_size=32, batch_size=32, dataset_dir="./datasets", transform=None):
    dataset_train = get_CIFA_dataset_train(image_size, batch_size=batch_size, dataset_dir=dataset_dir, transform=transform)
    dataset_val = get_CIFA_dataset_val(image_size, batch_size=batch_size, dataset_dir=dataset_dir, transform=transform)
    return dataset_train, dataset_val


if __name__ == '__main__':
    def get_example(dataset):
        # print(len(list(dataset)))
        print(dataset.get_dataset_size())
        for x, y in dataset:
            print("x:", type(x), x.shape)
            print("y:", type(y), y.shape)
            print("y data:", y)
            break

    dataset_train, dataset_val = get_CIFA_datasets(dataset_dir=r"../datasets/cifar10", batch_size=1)
    get_example(dataset_train)
    get_example(dataset_val)
