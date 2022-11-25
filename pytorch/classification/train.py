import copy
import time
import torch


def train(model, dataloaders, dataset_sizes, device, optim, loss_func, epochs, save_path):
    start_time = time.time()

    accuracy = 0.0
    best_weights = None
    for epoch in range(epochs):
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        for phase in ["train", "val"]:
            if phase == "train":
                # モデルを学習モードに設定する
                model.train()
            else:
                # モデルを評価モードに設定する
                model.eval()

            loss = 0.0
            successes = 0
            for imgs, tgts in dataloaders[phase]:
                # データを device で指定されたデバイスに送る
                imgs = imgs.to(device)
                tgts = tgts.to(device)

                optim.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # モデルの順伝播を実行して推論結果を取得
                    outputs = model(imgs)
                    _, preds = torch.max(outputs, 1)
                    loss_curt = loss_func(outputs, tgts)
                    if phase == "train":
                        # 学習時には逆伝播する
                        loss_curt.backward()
                        optim.step()
                    
                # ロスの累積と正解数のカウント
                loss += loss_curt.item() * imgs.size(0)
                successes += torch.sum(preds == tgts.data)

            loss_epoch = loss / dataset_sizes[phase]
            accuracy_epoch = successes / dataset_sizes[phase]
            print("\t{} loss: {}, accuracy: {}".format(phase, loss_epoch, accuracy_epoch))

            if (phase == "val") and (accuracy_epoch > accuracy):
                # 検証時の精度がこれまでで一番高ければ、一時的に best_weights に保持する
                accuracy = accuracy_epoch
                best_weights = copy.deepcopy(model.state_dict())

    # モデルを保存
    torch.save(best_weights, save_path)

    duration_time = time.time() - start_time
    hours = duration_time // 3600
    minutes = (duration_time - hours * 3600) // 60
    seconds = int(duration_time % 60)
    print("Training has been done!! ({} h {} m {} s)".format(hours, minutes, seconds))