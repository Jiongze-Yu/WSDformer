import torch.utils.data as data
from PIL import Image
import torch.nn.functional as F
import numpy as np
import os
from glob import glob
import random
from torchvision.transforms import Compose, ToTensor, Normalize
from random import randrange


def prepare_Rain200H(datapath):
    print("process Rain200H!")
    input_path = os.path.join(datapath, 'input')
    target_path = os.path.join(datapath, 'gt')
    imgs = []
    gts = []
    ids = []
    for i in range(1800):
        target_file = "norain-%d.png" % (i + 1)
        input_file = "norain-%dx2.png" %(i + 1)
        imgs.append(os.path.join(input_path, input_file))
        # imgs.append(os.path.join(target_path, target_file))
        gts.append(os.path.join(target_path, target_file))
        ids.append(target_file)
    return imgs, gts, ids

def prepare_Rain200H_test(datapath):
    print("process Rain200H!")
    input_path = os.path.join(datapath, 'input')
    target_path = os.path.join(datapath, 'gt')
    imgs = []
    gts = []
    ids = []
    for i in range(200):
        target_file = "norain-%d.png" % (i + 1)
        input_file = "norain-%dx2.png" %(i + 1)
        imgs.append(os.path.join(input_path, input_file))
        gts.append(os.path.join(target_path, target_file))
        ids.append(target_file)
    return imgs, gts, ids

def prepare_Rain200L(datapath):
    print("process Rain200L!")
    input_path = os.path.join(datapath, 'input')
    target_path = os.path.join(datapath, 'gt')
    imgs = []
    gts = []
    ids = []
    for i in range(1800):
        target_file = "norain-%d.png" % (i + 1)
        input_file = "norain-%dx2.png" %(i + 1)
        imgs.append(os.path.join(input_path, input_file))
        gts.append(os.path.join(target_path, target_file))
        ids.append(target_file)
    return imgs, gts, ids

def prepare_Rain200L_test(datapath):
    print("process Rain200L!")
    input_path = os.path.join(datapath, 'input')
    target_path = os.path.join(datapath, 'gt')
    imgs = []
    gts = []
    ids = []
    for i in range(200):
        target_file = "norain-%d.png" % (i + 1)
        input_file = "norain-%dx2.png" %(i + 1)
        imgs.append(os.path.join(input_path, input_file))
        gts.append(os.path.join(target_path, target_file))
        ids.append(target_file)
    return imgs, gts, ids


def prepare_DDN(datapath):
    print("process DDN!")
    input_path = os.path.join(datapath, 'input')
    target_path = os.path.join(datapath, 'gt')
    imgs = []
    gts = []
    ids = []
    for i in range(900):
        target_file = "%d.jpg" % (i + 1)
        for j in range(14):
            input_file = "%d_%d.jpg" % (i + 1, j + 1)
            imgs.append(os.path.join(input_path, input_file))
            gts.append(os.path.join(target_path, target_file))
            ids.append(input_file)
    return imgs, gts, ids

def prepare_DDN_test(datapath):
    print("process DDN!")
    input_path = os.path.join(datapath, 'input')
    target_path = os.path.join(datapath, 'gt')
    imgs = []
    gts = []
    ids = []
    for i in range(100):
        target_file = "%d.jpg" % (i + 901)
        for j in range(14):
            input_file = "%d_%d.jpg" % (i + 901, j + 1)
            imgs.append(os.path.join(input_path, input_file))
            gts.append(os.path.join(target_path, target_file))
            ids.append(input_file)
    return imgs, gts, ids

def prepare_DID(datapath):
    print("process DID!")
    imgs = []
    gts = []
    ids = []
    inputpath = os.path.join(datapath, 'input')
    gtpath = os.path.join(datapath, 'gt')
    for i in range(12000):
        target_file = "%d.jpg" % (i + 1)
        input_file = "%d.jpg" % (i + 1)
        imgs.append(os.path.join(inputpath, input_file))
        gts.append(os.path.join(gtpath, target_file))
        ids.append(target_file)
    return imgs, gts, ids

def prepare_DID_test(datapath):
    print("process DID!")
    imgs = []
    gts = []
    ids = []
    inputpath = os.path.join(datapath, 'input')
    gtpath = os.path.join(datapath, 'gt')
    for i in range(1200):
        target_file = "%d.jpg" % (i + 1)
        input_file = "%d.jpg" % (i + 1)
        imgs.append(os.path.join(inputpath, input_file))
        gts.append(os.path.join(gtpath, target_file))
        ids.append(target_file)
    return imgs, gts, ids

def prepare_SPA(datapath):
    print("process SPA!")
    txtpath = os.path.join(datapath, 'real_world.txt')
    mat_files = open(txtpath, 'r').readlines()
    file_num = len(mat_files)
    imgs = []
    gts = []
    ids = []
    for i in range(file_num):
        file_name = mat_files[i]
        input_file = file_name.split(' ')[0]
        target_file = file_name.split(' ')[1][:-1]
        imgs.append(datapath + input_file)
        gts.append(datapath + target_file)
        ids.append(input_file.split('/')[-1])
    return imgs, gts, ids

def prepare_SPA_test(datapath):
    print("process SPA!")
    txtpath = os.path.join(datapath, 'real_test_1000.txt')
    mat_files = open(txtpath, 'r').readlines()
    file_num = len(mat_files)
    imgs = []
    gts = []
    ids = []
    for i in range(file_num):
        file_name = mat_files[i]
        input_file = file_name.split(' ')[0]
        target_file = file_name.split(' ')[1][:-1]
        imgs.append(datapath+input_file)
        gts.append(datapath+target_file)
        ids.append(input_file.split('/')[-1])
    return imgs, gts, ids

def prepare_Real_test(datapath):
    print("process Real!")
    txtpath = os.path.join(datapath, 'Real_Internet.txt')
    mat_files = open(txtpath, 'r').readlines()
    file_num = len(mat_files)
    imgs = []
    gts = []
    ids = []
    for i in range(file_num):
        file_name = mat_files[i]
        input_file = file_name.split(' ')[0]
        target_file = file_name.split(' ')[1][:-1]
        imgs.append(datapath+input_file)
        gts.append(datapath+target_file)
        ids.append(input_file.split('/')[-1])
    return imgs, gts, ids

def prepare_Outdoor_rain(datapath):
    print("process Outdoor_rain!")
    clean_filenames = []
    noisy_filenames = []
    ids = []
    clean_filenames.extend(glob(os.path.join(datapath, 'gt', '*.png')))
    noisy_filenames.extend(glob(os.path.join(datapath, 'input', '*.png')))
    for f in noisy_filenames:
        filename = os.path.basename(f)
        ids.append(filename)
    return noisy_filenames, clean_filenames, ids

def prepare_Outdoor_rain_test(datapath):
    print("process Outdoor_rain!")
    clean_filenames = []
    noisy_filenames = []
    ids = []
    clean_filenames.extend(glob(os.path.join(datapath, 'gt', '*.png')))
    noisy_filenames.extend(glob(os.path.join(datapath, 'input', '*.png')))
    for f in noisy_filenames:
        filename = os.path.basename(f)
        ids.append(filename)
    return noisy_filenames, clean_filenames, ids

def prepare_rain100L(datapath):
    print("process Rain100L!")
    clean_filenames = []
    noisy_filenames = []
    ids = []
    clean_filenames.extend(glob(os.path.join(datapath, 'gt', '*.png')))
    noisy_filenames.extend(glob(os.path.join(datapath, 'input', '*.png')))
    for f in noisy_filenames:
        filename = os.path.basename(f)
        ids.append(filename)
    return noisy_filenames, clean_filenames, ids

def prepare_rain100L_test(datapath):
    print("process Rain100L!")
    clean_filenames = []
    noisy_filenames = []
    ids = []
    clean_filenames.extend(glob(os.path.join(datapath, 'gt', '*.png')))
    noisy_filenames.extend(glob(os.path.join(datapath, 'input', '*.png')))
    for f in noisy_filenames:
        filename = os.path.basename(f)
        ids.append(filename)
    return noisy_filenames, clean_filenames, ids

def prepare_rain100H(datapath):
    print("process Rain100H!")
    clean_filenames = []
    noisy_filenames = []
    ids = []
    clean_filenames.extend(glob(os.path.join(datapath, 'gt', '*.png')))
    noisy_filenames.extend(glob(os.path.join(datapath, 'input', '*.png')))
    for f in noisy_filenames:
        filename = os.path.basename(f)
        ids.append(filename)
    return noisy_filenames, clean_filenames, ids

def prepare_rain100H_test(datapath):
    print("process Rain100H!")
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
        if opt.train_data_path.find('Rain200H') != -1:
            imgs, gts, ids= prepare_Rain200H(opt.train_data_path)
        elif opt.train_data_path.find('Rain200L') != -1:
            imgs, gts, ids = prepare_Rain200L(opt.train_data_path)
        elif opt.train_data_path.find('DDN-Data') != -1:
            imgs, gts, ids = prepare_DDN(opt.train_data_path)
        elif opt.train_data_path.find('DID-Data') != -1:
            imgs, gts, ids = prepare_DID(opt.train_data_path)
        elif opt.train_data_path.find('SPA-Data') != -1:
            imgs, gts, ids = prepare_SPA(opt.train_data_path)
        elif opt.train_data_path.find('Outdoor-rain') != -1:
            imgs, gts, ids = prepare_Outdoor_rain(opt.train_data_path)
        elif opt.train_data_path.find('Rain100L') != -1:
            imgs, gts, ids = prepare_rain100L(opt.train_data_path)
        elif opt.train_data_path.find('Rain100H') != -1:
            imgs, gts, ids = prepare_rain100H(opt.train_data_path)
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
        if opt.test_data_path.find('Rain200H') != -1:
            imgs, gts, ids = prepare_Rain200H_test(opt.test_data_path)
        elif opt.test_data_path.find('Rain200L') != -1:
            imgs, gts, ids = prepare_Rain200L_test(opt.test_data_path)
        elif opt.test_data_path.find('DDN-Data') != -1:
            imgs, gts, ids = prepare_DDN_test(opt.test_data_path)
        elif opt.test_data_path.find('DID-Data') != -1:
            imgs, gts, ids = prepare_DID_test(opt.test_data_path)
        elif opt.test_data_path.find('SPA-Data') != -1:
            imgs, gts, ids = prepare_SPA_test(opt.test_data_path)
        elif opt.test_data_path.find('Real_Internet') != -1:
            imgs, gts, ids = prepare_Real_test(opt.test_data_path)
        elif opt.test_data_path.find('Outdoor-rain') != -1:
            imgs, gts, ids = prepare_Outdoor_rain_test(opt.test_data_path)
        elif opt.test_data_path.find('Rain100L') != -1:
            imgs, gts, ids = prepare_rain100L_test(opt.test_data_path)
        elif opt.test_data_path.find('Rain100H') != -1:
            imgs, gts, ids = prepare_rain100H_test(opt.test_data_path)
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
        transform_input = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])
        inp_im = transform_input(inp_img)
        tar_im = transform_gt(tar_img)

        # 用于训练过程中的测试，将图片的高宽crop成特定大小以提高训练速度
        # 若需要在测试过程不改变输入图像的大小进行测试，请注释掉以下代码
        # ===================================================================
        wd, ht = inp_img.size
        wd_new = int(8 * np.floor(wd / 8.0))
        ht_new = int(8 * np.floor(ht / 8.0))
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




