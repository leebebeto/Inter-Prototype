import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import Dataset
import mxnet as mx
import PIL.Image
import numbers
import argparse
import shutil
from pathlib import Path
import gdown
from zipfile import ZipFile
import sys

# Create a dataset class for reading from the .rec and .idx files
class MXFaceDataset(Dataset):
    def __init__(self, root_dir):
        super(MXFaceDataset, self).__init__()
        path_imgrec = f"{root_dir}/train.rec";
        path_imgidx = f"{root_dir}/train.idx";
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        sample = mx.image.imdecode(img).asnumpy()
        img = PIL.Image.fromarray(sample)
        label = header.label
        if not isinstance(label, numbers.Number): label = label[0]
        return img, int(label)
    
    def __len__(self):
        return len(self.imgidx)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to download training data from insightface repo")
    parser.add_argument('--dataset_type', type=str, help='which dataset to download from insightface repository')
    args = parser.parse_args()

    # Download file from the internet (Gdrive links as provided by insightface in their implementation here https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)
    if args.dataset_type.lower() == "casia_webface":
        dataset_id = "1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l"
        output = "casia"
    elif args.dataset_type.lower() == "ms1mv2":
        dataset_id = "1SXS4-Am3bsKSK615qbYdbA_FMVh3sAvR"
        output = "ms1mv2"
    elif args.dataset_type.lower() == "ms1mv3":
        dataset_id = "1JgmzL9OLTqDAZE86pBgETtSQL4USKTFy"
        output = "ms1mv3"
    else:
        print("Not a valid dataset. Please select one of the following\ncasia_webface, ms1mv2, ms1mv3")
        sys.exit()
    
    # Download the dataset from google drive using gdown library
    gdown.download(id = dataset_id, output = f"{output}.zip", quiet=False)

    # Create a downloads folder in dataset temporarily to hold the file
    # If it already exists, overwrite the download folder
    download_path = Path(f"dataset/downloads/")
    if download_path.exists(): shutil.rmtree(download_path)
    os.makedirs(str(download_path))

    # Unzip the extracted file 
    with ZipFile(f"{output}.zip", 'r') as f:
        f.extractall(str(download_path))
    
    # Each of the insightface datasets has a train.idx and train.rec file which is used
    # to read the images present in the dataset. The following code extracts the path of the same file
    for element in download_path.glob("**/*"):
        if element.name == "train.rec":
            root_dir = element.parent
            break

    # Extract the train data in dataset folder
    # Create a directory for storing the training data, overwrite the training data if there is already a train folder
    dest_folder = Path(f"dataset/train/{output}")
    if dest_folder.exists(): shutil.rmtree(dest_folder)
    os.makedirs(str(dest_folder))

    # root_dir = "/home/vinayak.n/Downloads/faces_webface_112x112"
    dataset = MXFaceDataset(str(root_dir))

    # Extract each image one by one
    for element in tqdm(dataset.imgidx, total = len(dataset.imgidx)):
        img, label = dataset[element]
        dest_path = dest_folder/f"{int(label)}"
        if not dest_path.exists(): dest_path.mkdir(exist_ok=True)
        dest_path = dest_path/f"{element}.png"
        img.save(dest_path, format="png")
        
    # Delete the unwanted downloads i.e. the downloaded zip file
    # And the mxnet files containing the train.rec and train.idx files
    shutil.rmtree(download_path)
    Path(f"{output}.zip").unlink()