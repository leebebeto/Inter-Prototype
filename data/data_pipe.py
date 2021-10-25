from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import glob
import os
import bcolz


def de_preprocess(tensor):
    return tensor*0.5 + 0.5

def get_train_loader(args):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    data_root = f'./dataset/train/{args.data_mode}/'
    if args.data_mode == 'casia':
        ds = CASIADataset(data_root, train_transforms=train_transform,  args=args)
        class_num = ds.class_num
        child_identity = ds.child_identity

    elif args.data_mode  == 'ms1m':
        ds = MS1MDataset(data_root, train_transforms=train_transform,  args=args)
        class_num = ds.class_num
        child_identity = ds.child_identity
    else:
        print('Wrong dataset name')
        raise NotImplementedError

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    return loader, class_num, child_identity

def get_val_pair(path, name):
    '''
    Returns image pairs with labels
        carray: numpy-like array with image pairs
        issame: boolean list of image pair validity
    '''
    carray = bcolz.carray(rootdir = os.path.join(path, name), mode='r')
    issame = np.load(os.path.join(path, '{}_list.npy'.format(name)))    
    return carray, issame

def get_val_data(data_path, name='cfp_fp'):
    data, data_issame = get_val_pair(data_path, name)
    return data, data_issame

class CASIADataset(Dataset):
    def __init__(self, imgs_folder, train_transforms, args):
        self.args = args
        self.root_dir = imgs_folder
        self.transform = train_transforms
        self.class_num = len(os.listdir(imgs_folder))
        self.age_file = open('./dataset/train/age-label/casia-webface.txt').readlines()
        self.id2age = { os.path.join(str(int(line.split(' ')[1].split('/')[1])), str(int(line.split(' ')[1].split('/')[2][:-4]))) : float(line.split(' ')[2]) for line in self.age_file}
        self.child_image2age = { os.path.join(str(int(line.split(' ')[1].split('/')[1])), str(int(line.split(' ')[1].split('/')[2][:-4]))) : float(line.split(' ')[2]) for line in self.age_file if float(line.split(' ')[2]) < 13}
        self.child_image2freq = {id.split('/')[0]: 0 for id in self.child_image2age.keys()}
        for k, v in self.child_image2age.items():
            self.child_image2freq[k.split('/')[0]] += 1

        # sorted in ascending order
        self.child_identity_freq = {int(k): v for k, v in sorted(self.child_image2freq.items(), key=lambda item: item[1]) if v >= args.child_filter}
        self.child_identity = list(self.child_identity_freq.keys())
        print(f'child number: {len(self.child_identity)}')

        total_list = glob.glob(self.root_dir + '/*/*')

        self.total_imgs = len(total_list)
        self.total_list = total_list
        print(f'{imgs_folder} length: {self.total_imgs}')

    def __len__(self):
        return self.total_imgs

    def __getitem__(self, index):
        img_path = self.total_list[index]
        id_img = '/'.join((img_path.split('/')[-2], img_path.split('/')[-1].split('_')[0]))
        if 'jpg' in id_img:
            id_img = id_img[:-4]
        try:
            age = self.id2age[id_img]
        except:
            age = 30 # for images which do not have age labels

        img = Image.open(img_path)
        label = int(img_path.split('/')[-2])

        if self.transform is not None:
            img = self.transform(img)

        else:
            age= 0 if age< 13 else 1

        return img, label, age


class MS1MDataset(Dataset):
    def __init__(self, imgs_folder, train_transforms, args):
        self.args = args
        self.root_dir = imgs_folder
        self.transform = train_transforms
        self.class_num = len(os.listdir(imgs_folder))
        self.age_file = open('./dataset/train/age-label/ms1m.txt').readlines()
        self.id2age = {os.path.join(str(int(line.split(' ')[1].split('/')[1])), str(int(line.split(' ')[1].split('/')[2][:-4]))) : float(line.split(' ')[2]) for line in self.age_file}
        self.child_image2age = { os.path.join(str(int(line.split(' ')[1].split('/')[1])), str(int(line.split(' ')[1].split('/')[2][:-4]))) : float(line.split(' ')[2]) for line in self.age_file if float(line.split(' ')[2]) < 13}
        self.child_image2freq = {id.split('/')[0]: 0 for id in self.child_image2age.keys()}
        for k, v in self.child_image2age.items():
            self.child_image2freq[k.split('/')[0]] += 1

        # sorted in ascending order
        self.child_identity_freq = {int(k): v for k, v in sorted(self.child_image2freq.items(), key=lambda item: item[1]) if v >= args.child_filter}
        self.child_identity = list(self.child_identity_freq.keys())
        print(f'child number: {len(self.child_identity)}')

        total_list = glob.glob(self.root_dir + '/*/*.jpg')
        self.total_imgs = len(total_list)

        self.total_list = total_list
        print(f'{imgs_folder} length: {self.total_imgs}')
    def __len__(self):
        return self.total_imgs

    def __getitem__(self, index):
        img_path = self.total_list[index]
        id_img = '/'.join((img_path.split('/')[-2], img_path.split('/')[-1].split('_')[0]))
        if 'jpg' in id_img:
            id_img = id_img[:-4]
        try:
            age = self.id2age[id_img]
        except:
            age = 30 # for images which do not have age labels

        img = Image.open(img_path)
        label = int(img_path.split('/')[-2])

        if self.transform is not None:
            img = self.transform(img)

        else:
            age= 0 if age< 13 else 1

        return img, label, age