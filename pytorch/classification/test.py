import copy
import time
import torch


def test(model, dataloader, dataset_size, device, loss_func):
    start = time.time()

    model.eval()
    successes = 0
    with torch.no_grad():
        for imgs, tgts in dataloader:
            # データを device で指定されたデバイスに送る
            imgs = imgs.to(device)
            tgts = tgts.to(device)

            # モデルの順伝播を実行して推論結果を取得
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            successes += torch.sum(preds == tgts.data)
    
    duration_time = time.time() - start_time
    hours = duration_time // 3600
    minutes = (duration_time - hours * 3600) // 60
    seconds = int(duration_time % 60)
    print("Test accuracy: {} % ({} h {} m {} s)".format(successes / dataset_size, hours, minutes, seconds))
            