import os
import pickle
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# データセットの情報
DATA_DIR = "/dataset/cifar10"
META_DATA = "batches.meta"
TRAIN_BATCHES = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
TEST_BATCH = "test_batch"


def get_meta():
    meta_file = os.path.join(DATA_DIR, META_DATA)
    with open(meta_file, "rb") as f:
        meta_data = pickle.load(f, encoding="latin-1")
    return meta_data


def make_train_val_loader(
    train_transform, val_transform, batch_size,
    val_ratio=0.2, num_workers=4, random_state=42
):
    """学習セットと検証セットを作成する関数
    Parameters
    ----------
        train_transform : torchvision.transforms.Compose
            学習用データセットの画像変換クラスのリスト
        val_transform : torchvision.transforms.Compose
            検証用データセットの画像変換クラスのリスト
        batch_size : int
            バッチサイズ (学習用と検証用で共通とする)
        train_ratio : float
            学習用データセットの割合
        num_workers : int
            データローダのプロセス数
        random_state : int
            データセットのシャッフルを制御する変数

    Returns
    -------
        train_dataloader : torch.utils.data.Dataloader
            学習用データセットのデータローダ
        val_dataloader : torch.utils.data.Dataloader
            検証用データセットのデータローダ
    """

    # データの読み出し
    images, labels = [], []
    for batch in TRAIN_BATCHES:
        batch_file = os.path.join(DATA_DIR, batch)
        with open(batch_file, "rb") as f:
            batch_data = pickle.load(f, encoding="latin-1")
        images += batch_data["data"]
        labels += batch_data["labels"]

    # 学習用データセットと検証用データセットに分割し、各データセットのインスタンスを作成
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=val_ratio, random_state=random_state
    )
    train_dataset = CIFAR10Dataset(train_images, train_labels, transform=train_transform)
    val_dataset = CIFAR10Dataset(val_images, val_labels, transform=val_transform)

    # データローダを生成して返す
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers    
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers 
    )
    return train_dataloader, val_dataloader


def make_test_loader(test_transform, batch_size, num_workers=4):
    """学習セットと検証セットを作成する関数
    Parameters
    ----------
        test_transform : torchvision.transforms.Compose
            テスト用データセットの画像変換クラスのリスト
        batch_size : int
            バッチサイズ
        num_workers : int
            データローダのプロセス数

    Returns
    -------
        test_dataloader : torch.utils.data.Dataloader
            テスト用データセットのデータローダ
    """

    # データの読み出し
    batch_file = os.path.join(DATA_DIR, TEST_BATCH)
    with open(batch_file, "rb") as f:
        batch_data = pickle.load(f, encoding="latin-1")
    images = batch_data["data"]
    labels = batch_data["labels"]

    # テスト用データセットのインスタンスを生成
    test_dataset = CIFAR10Dataset(images, labels, transform=test_transform)

    # データローダを生成して返す
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers    
    )
    return test_dataloader


class CIFAR10Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    
    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        # 画像データをを整形するして、PIL.Image オブジェクトに変換
        image = self.images[index].reshape((3, 32, 32))
        image = Image.fromarray(np.transpose(image, (1, 2, 0)))

        if self.transform is not None:
            # transform が None 出なければ画像を変換
            image = self.transform(image)
        else:
            # Image オブジェクト (H, W, C) から torch.Tensor (C, H, W) に変換し、
            # [0, 255] (uint8) から [0, 1] (float32) に正規化する
            image = transforms.ToTensor()(image)

        # ラベルの取得
        label = self.labels[index]


        return image, label
