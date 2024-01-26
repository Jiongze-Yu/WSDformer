import torch
import time
import os
from skimage import img_as_ubyte
import utils
import argparse
import cv2
import math
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from utils.image_utils import save_img


def splitimage(imgtensor, crop_size=128, overlap_size=20):
    _, C, H, W = imgtensor.shape
    hstarts = [x for x in range(0, H, crop_size - overlap_size)]
    while hstarts[-1] + crop_size >= H:
        hstarts.pop()
    hstarts.append(H - crop_size)
    wstarts = [x for x in range(0, W, crop_size - overlap_size)]
    while wstarts[-1] + crop_size >= W:
        wstarts.pop()
    wstarts.append(W - crop_size)
    starts = []
    split_data = []
    for hs in hstarts:
        for ws in wstarts:
            cimgdata = imgtensor[:, :, hs:hs + crop_size, ws:ws + crop_size]
            starts.append((hs, ws))
            split_data.append(cimgdata)
    return split_data, starts


def get_scoremap(H, W, C, B=1, is_mean=True):
    center_h = H / 2
    center_w = W / 2

    score = torch.ones((B, C, H, W))
    if not is_mean:
        for h in range(H):
            for w in range(W):
                score[:, :, h, w] = 1.0 / (math.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + 1e-3))
    return score


def mergeimage(split_data, starts, crop_size = 128, resolution=(1, 3, 128, 128)):
    B, C, H, W = resolution[0], resolution[1], resolution[2], resolution[3]
    tot_score = torch.zeros((B, C, H, W))
    merge_img = torch.zeros((B, C, H, W))
    scoremap = get_scoremap(crop_size, crop_size, C, B=B, is_mean=False)
    for simg, cstart in zip(split_data, starts):
        hs, ws = cstart
        merge_img[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap * simg
        tot_score[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap
    merge_img = merge_img / tot_score
    return merge_img

def calc_psnr(im1, im2):
    im1 = im1[0].view(im1.shape[2],im1.shape[3],3).detach().cpu().numpy()
    im2 = im2[0].view(im2.shape[2],im2.shape[3],3).detach().cpu().numpy()

    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]

    ans = [compare_psnr(im1_y, im2_y)]

    return ans

def calc_ssim(im1, im2):
    im1 = im1[0].view(im1.shape[2],im1.shape[3],3).detach().cpu().numpy()
    im2 = im2[0].view(im2.shape[2],im2.shape[3],3).detach().cpu().numpy()

    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    ans = [compare_ssim(im1_y, im2_y)]

    return ans

def print_log(epoch, num_epochs, one_epoch_time, val_psnr, val_ssim, best_psnr, exp_name):
    print('({0:.0f}s) Epoch [{1}/{2}] '
          'Val_PSNR:{3:.2f}, Val_SSIM:{4:.4f}, Best_PSNR:{5:.2f}'
          .format(one_epoch_time, epoch, num_epochs, val_psnr, val_ssim, best_psnr))

    # --- Write the training log --- #
    with open('./training_log/{}_log.txt'.format(exp_name), 'a') as f:
        print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}],'
              'Val_PSNR: {4:.2f}, Val_SSIM: {5:.4f}, Best_PSNR:{6:.2f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, val_psnr, val_ssim, best_psnr), file=f)


def val_full_size(eval_loader, model_restoration, result_dir, save_tag=False):
    psnr_list = []
    ssim_list = []
    os.makedirs(result_dir, exist_ok=True)
    with torch.no_grad():
        for batch_id, train_data in enumerate(eval_loader):
            input, gt, imgid = train_data
            input = input.cuda()
            B, C, H, W = input.shape
            split_data, starts = splitimage(input)
            for i, data in enumerate(split_data):
                split_data[i] = model_restoration(data).cpu()
            restored = mergeimage(split_data, starts, resolution=(B, C, H, W))
            restored = torch.clamp(restored, 0, 1)
            # restored = torch.clamp(restored, 0, 1).permute(0, 2, 3, 1).numpy()

            # print(gt.shape,restored.shape)
            psnr_list.extend(calc_psnr(gt, restored))
            print(calc_psnr(gt, restored))
            # psnr_list.extend(to_psnr(pred_image, gt))

            # --- Calculate the average SSIM --- #
            ssim_list.extend(calc_ssim(gt, restored))

            restored = restored.permute(0, 2, 3, 1).numpy()
            if save_tag:
                for j in range(B):
                    fname = imgid[j]
                    cleanname = fname
                    save_file = os.path.join(result_dir, cleanname)
                    save_img(save_file, img_as_ubyte(restored[j]))
                    print(fname)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)

    psnr_tensor = torch.as_tensor(avr_psnr)
    ssim_tensor = torch.as_tensor(avr_ssim)

    return psnr_tensor, ssim_tensor

def validation(net, val_data_loader):

    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():
            input_im, gt, imgid = val_data
            input_im = input_im.cuda()
            gt = gt.cuda()
            pred_image = net(input_im)


# --- Calculate the average PSNR --- #
        psnr_list.extend(calc_psnr(gt, pred_image))
        # psnr_list.extend(to_psnr(pred_image, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(calc_ssim(gt, pred_image))
        # ssim_list.extend(to_ssim_skimage(pred_image, gt))


    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)

    psnr_tensor = torch.as_tensor(avr_psnr)
    ssim_tensor = torch.as_tensor(avr_ssim)

    return psnr_tensor, ssim_tensor