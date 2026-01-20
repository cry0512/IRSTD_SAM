import os
import sys
import pathlib
import argparse
from typing import Optional

from PIL import Image

# ensure repository root on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sirst_dataset import SIRSTDataset


def pick_txt(path_dir: str, prefer_keywords) -> Optional[str]:
    if not os.path.isdir(path_dir):
        return None
    txts = [f for f in os.listdir(path_dir) if f.lower().endswith('.txt')]
    if not txts:
        return None
    for kw in prefer_keywords:
        for f in txts:
            if kw in f.lower():
                return os.path.join(path_dir, f)
    return os.path.join(path_dir, txts[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True, type=str)
    ap.add_argument('--train_txt', type=str, default=None)
    ap.add_argument('--val_txt', type=str, default=None)
    ap.add_argument('--split_dir', type=str, default='50_50', help='subdir that contains txt files')
    ap.add_argument('--limit', type=int, default=5)
    ap.add_argument('--size', type=int, default=1024)
    args = ap.parse_args()

    # Auto-pick txt if not provided
    split_dir_abs = os.path.join(args.data_root, args.split_dir)
    if args.train_txt is None:
        cand = pick_txt(split_dir_abs, ['train'])
        args.train_txt = cand if cand is not None else None
    elif not os.path.isabs(args.train_txt):
        args.train_txt = os.path.join(args.data_root, args.train_txt)

    if args.val_txt is None:
        cand = pick_txt(split_dir_abs, ['test', 'val'])
        args.val_txt = cand if cand is not None else None
    elif not os.path.isabs(args.val_txt):
        args.val_txt = os.path.join(args.data_root, args.val_txt)

    print('Data root:', args.data_root)
    print('Train txt:', args.train_txt)
    print('Val   txt:', args.val_txt)

    def check(split_txt):
        if split_txt is None or not os.path.exists(split_txt):
            print('  [!] txt not found:', split_txt)
            return
        try:
            ds = SIRSTDataset(
                root=args.data_root,
                split_txt=split_txt,
                size=args.size,
                augment=False,
                keep_ratio_pad=False,
            )
        except Exception as e:
            print('  [!] Failed to build dataset:', e)
            return
        print('  samples:', len(ds))
        lim = min(args.limit, len(ds))
        for i in range(lim):
            img_path, mask_path = ds.samples[i]
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                with Image.open(mask_path) as mm:
                    mw, mh = mm.size
                print(f'   - {i}:')
                print(f'     img:  {img_path}  ({w}x{h})')
                print(f'     mask: {mask_path}  ({mw}x{mh})')
            except Exception as e:
                print(f'   - {i}: failed to open ->', e)

    print('[Train split]')
    check(args.train_txt)
    print('[Val/Test split]')
    check(args.val_txt)


if __name__ == '__main__':
    main()
