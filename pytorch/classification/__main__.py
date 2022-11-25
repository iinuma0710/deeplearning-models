import argparse
from classification import train, test, inference


def args_parser():
    parser = argparse.ArgumentParser()
    # 
    parser.add_argument("phase", type=str, choices=["train", "test", "inference"])
    parser.add_argument("-c", "--cfg", type=str)
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-i", "--input", type=str)
    return parser.parse_args()


def load_model(cfg):
    pass


def main():
    # 引数をパースして設定ファイルを読み込む
    args = args_parser()


    if args.phase == "train":
        pass

    
    print(args.phase, args.cfg, args.dataset, args.input)


if __name__ == "__main__":
    main()