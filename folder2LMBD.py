import os
import os.path as osp
import os, sys
import os.path as osp
from PIL import Image
import six
import string

import lmdb
import pickle  # Import pickle for serialization
import tqdm
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

class SingleFolderDataset(data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform

        # List all image files
        self.image_files = [f for f in os.listdir(folder_path) if osp.isfile(osp.join(folder_path, f))]

        # Extract and sort unique class names
        class_names = sorted(set(self._extract_class_from_filename(f) for f in self.image_files))

        # Create a mapping from class names to integers
        self.class_to_idx = {class_names[i]: i for i in range(len(class_names))}

        # Map filenames to labels
        self.labels = [self.class_to_idx[self._extract_class_from_filename(filename)] for filename in self.image_files]

    def __getitem__(self, index):
        # Load image as raw binary data
        img_path = osp.join(self.folder_path, self.image_files[index])
        with open(img_path, 'rb') as f: #raw_reader
            img = f.read()

        # Get label
        label = self.labels[index]

        return img, label

    def __len__(self):
        return len(self.image_files)

    def _extract_class_from_filename(self, filename):
        """
        Extract the class label from the filename.
        Assumes class label is the part of the filename before the first underscore.
        """
        return filename.split('_')[0]

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.folder_path + ')'
    
class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))  # Deserialize using pickle
            self.keys = pickle.loads(txn.get(b'__keys__'))  # Deserialize using pickle

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pickle.loads(byteflow)  # Deserialize using pickle
        imgbuf = unpacked[0][b'data']
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

# Modify the dumps_pyarrow function to use pickle
def dumps_pyarrow(obj):
    image, label = obj
    return pickle.dumps((image, label))  # Serialize using pickle

# Modify the folder2lmdb function
def folder2lmdb(dpath, name="train", write_frequency=5000, num_workers=16):
    directory = osp.expanduser(osp.join(dpath, name))
    print("Loading dataset from %s" % directory)
    dataset = SingleFolderDataset(directory)
    data_loader = DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x)

    lmdb_path = osp.join(dpath, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)
    
    print(len(dataset), len(data_loader))
    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        #print(type(data), data)
        image, label = data[0]
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys))  # Serialize using pickle
        txn.put(b'__len__', pickle.dumps(len(keys)))  # Serialize using pickle

    print("Flushing database ...")
    db.sync()
    db.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str)
    parser.add_argument('-s', '--split', type=str, default="val")
    parser.add_argument('--out', type=str, default=".")
    parser.add_argument('-p', '--procs', type=int, default=20)

    args = parser.parse_args()

    folder2lmdb(args.folder, num_workers=args.procs, name=args.split)
