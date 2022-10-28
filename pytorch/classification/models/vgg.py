import torch
import torch.nn as nn


VGG_TYPES = [
    "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"
]

VGG_CFG = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
}


class VGG(nn.Module):
    def __init__(self, vgg_type="vgg16", num_classes=1000):
        """
        Args:
            vgg_type (string): VGG のネットワークの構成を VGG_TYPES の中から選択
            num_classes (int): 分類するクラス数、すなわち出力層の次元
        """

        # VGG_TYPES にない値が渡された時にはエラーとする
        if vgg_type not in VGG_TYPES:
            msg = "{} は無効な VGG タイプです。[{}] から選択してください".format(vgg_type, ", ".join(VGG_TYPES))
            raise Exception(msg)

        # 特徴抽出を行うネットワークを定義する
        vgg_cfg = VGG_CFG[vgg_type.split("_")[0]]
        batch_norm = ("bn" in vgg_type)
        self.feats = _make_layers(vgg_cfg, batch_norm)

        # 分類の前に adaptive average pooling を適用する
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        # クラス分類を行う全結合層を定義する
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )


    def forward(self, x):
        x = self.feats(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)

        

def _make_layers(vgg_cfg, batch_norm):
    """
    特徴抽出を行うレイヤを定義する関数

    Args:
        vgg_cfg (list):     VGG のレイヤ情報を格納したリスト
        batch_norm (bool):  Batch Normalization を使うか

    Returns:
        nn.Sequential 型の変数
    """
    
    in_channels = 3
    layers = nn.Sequential()
    for v in vgg_cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        else:
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=v,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            )
            if batch_norm:
                layers.append(conv)
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU())
            else:
                layers.append(conv)
                layers.append(nn.ReLU())
        in_channels = v

    return layers