import argparse
import os
from pathlib import Path
import torch
from torchvision import datasets
import lmdb
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed  # Import as_completed here


def read_image(image_path):
    with open(image_path, 'rb') as f:
        data = f.read()
    return image_path, data

def convert_to_lmdb(root, allocated_bytes, max_workers=16, delete_original=False):
    pt_path = os.path.join(root + '_faster_imagefolder.lmdb.pt')
    lmdb_path = os.path.join(root + '_faster_imagefolder.lmdb')

    if os.path.isfile(pt_path) and os.path.isdir(lmdb_path):
        print(f'[Dataset] LMDB dataset already exists at {lmdb_path}')
        return
    else:
        data_set = datasets.ImageFolder(root)
        torch.save(data_set, pt_path, pickle_protocol=4)
        print(f'[Dataset] Saving pt to {pt_path}')
        print(f'[Dataset] Building lmdb to {lmdb_path}')

        env = lmdb.open(lmdb_path, map_size=allocated_bytes)
        with env.begin(write=True) as txn, ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(read_image, _path): _path for _path, _ in data_set.imgs}
            for future in tqdm(as_completed(future_to_path), total=len(future_to_path)):  # Use as_completed directly
                image_path, data = future.result()
                txn.put(image_path.encode('ascii'), data)

    if delete_original:
        print(f'[Dataset] Deleting original image folder {root}')
        for _root, dirs, files in os.walk(root, topdown=False):
            for name in files:
                os.remove(os.path.join(_root, name))
            for name in dirs:
                os.rmdir(os.path.join(_root, name))

def main():
    parser = argparse.ArgumentParser(description='Convert Image Folder to LMDB Dataset')
    parser.add_argument("--dataset-dir", type=Path, default="/dataset", help="path to dataset")
    parser.add_argument('--delete_original', action='store_true', help='Delete the original image folder after conversion')
    parser.add_argument('--server', type=str, default='hpc', help='server - determines gbs for lmdb dataset size allocation')

    args = parser.parse_args()

    if args.server == 'hpc':
        allocate_ram_memory_in_gbs = 200
    else:
        allocate_ram_memory_in_gbs = 20
        
    allocated_bytes = allocate_ram_memory_in_gbs * (1024 ** 3)

    val_dir = os.path.join(args.dataset_dir, 'val')
    train_dir = os.path.join(args.dataset_dir, 'train')

    convert_to_lmdb(val_dir, allocated_bytes, delete_original=args.delete_original)
    convert_to_lmdb(train_dir, allocated_bytes, delete_original=args.delete_original)

if __name__ == "__main__":
    main()
