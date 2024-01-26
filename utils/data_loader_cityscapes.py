import torch.utils.data as data
from PIL import Image
import torch.nn.functional as F
import numpy as np
import os
from glob import glob
import random
from torchvision.transforms import Compose, ToTensor, Normalize
from random import randrange


def prepare_cityscapes(datapath):
    print("process cityscapes!")
    clean_filenames = []
    noisy_filenames = []
    ids = []
    clean_filenames.extend(glob(os.path.join(datapath, 'gt', '*.png')))
    noisy_filenames.extend(glob(os.path.join(datapath, 'input', '*.png')))
    for f in noisy_filenames:
        filename = os.path.basename(f)
        ids.append(filename)
    return noisy_filenames, clean_filenames, ids

def prepare_cityscapes_val(datapath):
    print("process cityscapes!")
    clean_filenames = []
    noisy_filenames = []
    ids = []
    clean_filenames.extend(glob(os.path.join(datapath, 'gt', '*.png')))
    noisy_filenames.extend(glob(os.path.join(datapath, 'input', '*.png')))
    for f in noisy_filenames:
        filename = os.path.basename(f)
        ids.append(filename)
    return noisy_filenames, clean_filenames, ids

def prepare_cityscapes_test(datapath):
    print("process cityscapes!")
    clean_filenames = []
    noisy_filenames = []
    ids = []
    clean_filenames.extend(glob(os.path.join(datapath, 'gt', '*.png')))
    noisy_filenames.extend(glob(os.path.join(datapath, 'input', '*.png')))
    for f in noisy_filenames:
        filename = os.path.basename(f)
        ids.append(filename)
    return noisy_filenames, clean_filenames, ids

class DataLoaderTrain(data.Dataset):
    def __init__(self, opt):
        super(DataLoaderTrain, self).__init__()
        self.opt = opt
        if opt.train_data_path.find('Cityscapes') != -1:
            imgs, gts, ids = prepare_cityscapes(opt.train_data_path)
        else:
            raise (RuntimeError('Cannot find dataset!'))

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + opt.base_dir + "\n"))
        self.imgs = imgs
        self.gts = gts
        self.ids = ids
        self.sizex = len(self.imgs)
        self.count = 0
        self.crop_size = opt.crop_size

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        inp_path = self.imgs[index_]
        tar_path = self.gts[index_]
        imgid = self.ids[index_]
        input_img = Image.open(inp_path).convert('RGB')
        gt_img = Image.open(tar_path).convert('RGB')

        crop_width = self.crop_size
        crop_height = self.crop_size


        if len(gt_img.split()) != 3:
            gt_img = np.delete(gt_img, 3, axis=2)
            gt_img = Image.fromarray(gt_img)

        width, height = input_img.size

        if width < crop_width and height < crop_height:
            input_img = input_img.resize((crop_width, crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width, crop_height), Image.ANTIALIAS)
        elif width < crop_width:
            input_img = input_img.resize((crop_width, height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width, height), Image.ANTIALIAS)
        elif height < crop_height:
            input_img = input_img.resize((width, crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((width, crop_height), Image.ANTIALIAS)

        width, height = input_img.size

        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        input_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        # --- Transform to tensor --- #
        # transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_input = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_crop_img)
        gt = transform_gt(gt_crop_img)

        # Data Augmentations
        aug = random.randint(0, 3)
        if aug == 1:
            input_im = input_im.flip(1)
            gt = gt.flip(1)
        elif aug == 2:
            input_im = input_im.flip(2)
            gt = gt.flip(2)

        return input_im, gt, imgid


class DataLoaderTest(data.Dataset):
    def __init__(self, opt):
        super(DataLoaderTest, self).__init__()
        self.opt = opt
        if opt.test_data_path.find('Cityscapes') != -1:
            imgs, gts, ids = prepare_cityscapes_test(opt.test_data_path)
        else:
            raise (RuntimeError('Cannot find dataset!'))

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + opt.base_dir + "\n"))
        self.imgs = imgs
        self.gts = gts
        self.ids = ids
        self.sizex = len(self.imgs)
        self.count = 0
        # self.crop_size = opt.crop_size

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        inp_path = self.imgs[index_]
        tar_path = self.gts[index_]
        imgid = self.ids[index_]
        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        # --- Transform to tensor --- #
        # transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_input = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])
        inp_im = transform_input(inp_img)
        tar_im = transform_gt(tar_img)

        # 用于训练过程中的测试，将图片的高宽crop成特定大小以提高训练速度
        # 若需要在测试过程不改变输入图像的大小进行测试，请注释掉以下代码
        # ===================================================================
        wd, ht = inp_img.size
        # wd_new = int(8 * np.floor(wd / 8.0))
        # ht_new = int(8 * np.floor(ht / 8.0))
        wd_new = 256
        ht_new = 256
        pd = (0, wd_new - wd, 0, ht_new - ht)
        inp_im = F.pad(inp_im, pd, "constant", 0)
        tar_im = F.pad(tar_im, pd, "constant", 0)
        # ===================================================================

        return inp_im, tar_im, imgid

class DataLoaderVal(data.Dataset):
    def __init__(self, opt):
        super(DataLoaderVal, self).__init__()
        self.opt = opt
        if opt.test_data_path.find('Cityscapes') != -1:
            imgs, gts, ids = prepare_cityscapes_val(opt.val_data_path)
        else:
            raise (RuntimeError('Cannot find dataset!'))

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + opt.base_dir + "\n"))
        self.imgs = imgs
        self.gts = gts
        self.ids = ids
        self.sizex = len(self.imgs)
        self.count = 0
        # self.crop_size = opt.crop_size

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        inp_path = self.imgs[index_]
        tar_path = self.gts[index_]
        imgid = self.ids[index_]
        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        # --- Transform to tensor --- #
        # transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_input = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])
        inp_im = transform_input(inp_img)
        tar_im = transform_gt(tar_img)

        # 用于训练过程中的测试，将图片的高宽crop成特定大小以提高训练速度
        # 若需要在测试过程不改变输入图像的大小进行测试，请注释掉以下代码
        # ===================================================================
        wd, ht = inp_img.size
        # wd_new = int(8 * np.floor(wd / 8.0))
        # ht_new = int(8 * np.floor(ht / 8.0))
        wd_new = 256
        ht_new = 256
        pd = (0, wd_new - wd, 0, ht_new - ht)
        inp_im = F.pad(inp_im, pd, "constant", 0)
        tar_im = F.pad(tar_im, pd, "constant", 0)
        # ===================================================================

        return inp_im, tar_im, imgid




def getloader(opt):
    dataset = DataLoaderTrain(opt)
    print("Dataset Size:%d" %(len(dataset)))
    trainloader = data.DataLoader(dataset,
            batch_size=opt.train_batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=False)
    return trainloader

def getloader_test(opt):
    dataset = DataLoaderTest(opt)
    print("Dataset Size:%d" %(len(dataset)))
    testloader = data.DataLoader(dataset,
            batch_size=opt.val_batch_size,
            shuffle=False,
            num_workers=8)
    return testloader

def getloader_val(opt):
    dataset = DataLoaderVal(opt)
    print("Dataset Size:%d" %(len(dataset)))
    testloader = data.DataLoader(dataset,
            batch_size=opt.val_batch_size,
            shuffle=False,
            num_workers=8)
    return testloader




