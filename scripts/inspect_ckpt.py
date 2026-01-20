import argparse
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', required=True)
    args = ap.parse_args()
    ckpt = torch.load(args.path, map_location='cpu')
    print('Loaded:', args.path)
    print('Keys:', list(ckpt.keys()))
    print('Epoch:', ckpt.get('epoch'))
    print('Best IoU:', ckpt.get('best_iou'))
    args_dict = ckpt.get('args')
    if isinstance(args_dict, dict):
        print('Args:')
        for k, v in args_dict.items():
            print(f'  {k}: {v}')

if __name__ == '__main__':
    main()

