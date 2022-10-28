import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """
    オリジナルからの変更点
        (プーリング) => Max Pooling
        (活性化関数) => ReLU
        (入力チャネル数) => 3
    """
    def __init__(self):
        super(LeNet, self).__init__()
        # 第1層：入力3チャネル、出力6チャネル、5x5の畳み込み
        self.cn1 = nn.Conv2d(3, 6, 5)
        # 第2層：入力6チャネル、出力16チャネル、5x5の畳み込み
        self.cn2 = nn.Conv2d(6, 16, 5)
        # 出力がそれぞれ120次元、84次元、10次元の全結合層
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 畳み込み層の出力は 5x5
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        # 第1層の畳み込みとプーリング
        x = F.relu(self.cn1(x))
        x = F.max_pool2d(x, (2, 2))
        # 第1層の畳み込みとプーリング
        x = F.relu(self.cn2(x))
        x = F.max_pool2d(x, (2, 2))
        # 1次元のベクトルに変換する
        x = x.view(-1, self.flattened_features(x))
        # 全結合層
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def flattened_features(self, x):
        # 特徴量の層数を計算する
        size = x.size()[1:]
        num_feats = 1
        for s in size:
            num_feats *= s
        return num_feats