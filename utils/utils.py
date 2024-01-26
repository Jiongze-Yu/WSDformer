import time
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from math import log10
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np

def calc_psnr(im1, im2):
    im1 = im1[0].contiguous().view(im1.shape[2],im1.shape[3],3).detach().cpu().numpy()
    im2 = im2[0].contiguous().view(im2.shape[2],im2.shape[3],3).detach().cpu().numpy()

    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]

    ans = [compare_psnr(im1_y, im2_y)]
    # print(ans)
    return ans

def calc_ssim(im1, im2):
    im1 = im1[0].contiguous().view(im1.shape[2],im1.shape[3],3).detach().cpu().numpy()
    im2 = im2[0].contiguous().view(im2.shape[2],im2.shape[3],3).detach().cpu().numpy()

    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    ans = [compare_ssim(im1_y, im2_y)]

    ssim_numpy = np.array(ans)
    ssim_tensor = torch.from_numpy(ssim_numpy)

    return ssim_tensor

def to_psnr(pred_image, gt):
    mse = F.mse_loss(pred_image, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]

    psnr_numpy = np.array(psnr_list)
    psnr_tensor = torch.from_numpy(psnr_numpy)
    return psnr_tensor


def to_ssim_skimage(pred_image, gt):
    pred_image_list = torch.split(pred_image, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    pred_image_list_np = [pred_image_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
    ssim_list = [compare_ssim(pred_image_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(pred_image_list))]

    ssim_numpy = np.array(ssim_list)
    ssim_tensor = torch.from_numpy(ssim_numpy)
    return ssim_tensor


def validation(net, val_data_loader, device, exp_name, category, save_tag=False):

    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():
            input_im, gt, imgid = val_data
            input_im = input_im.to(device)
            gt = gt.to(device)
            pred_image = net(input_im)

        # --- Calculate the average PSNR --- #
        psnr_list.extend(calc_psnr(gt, pred_image))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(calc_ssim(gt, pred_image))

        # --- Save image --- #
        if save_tag:
            save_image(pred_image, imgid, exp_name, category)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)

    psnr_tensor = torch.as_tensor(avr_psnr)
    ssim_tensor = torch.as_tensor(avr_ssim)

    return psnr_tensor, ssim_tensor


def validation_val(net, val_data_loader, device, exp_name, category, save_tag=False):

    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():
            input_im, gt, imgid = val_data
            # print(input_im.shape)
            input_im = input_im.to(device)
            gt = gt.to(device)
            pred_image = net(input_im)

        # --- Calculate the average PSNR --- #
        psnr_list.extend(calc_psnr(pred_image, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(calc_ssim(pred_image, gt))

        # --- Save image --- #
        if save_tag:
            save_image(pred_image, imgid, exp_name,category)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim

def save_image(pred_image, image_name, exp_name, category):
    pred_image_images = torch.split(pred_image, 1, dim=0)  # 将batch_size维度分开,得到的是一个turple,每个元素的batch_size=1
    batch_num = len(pred_image_images)
    
    for ind in range(batch_num):
        image_name_1 = image_name[ind].split('/')[-1]  # 将最后一块切出来
        print(image_name_1)
        # print(pred_image[ind].shape)
        utils.save_image(pred_image_images[ind], './results/{}/{}/{}'.format(category,exp_name,image_name_1))


def save_image_real(pred_image, image_name, exp_name, category):
    pred_image_images = torch.split(pred_image, 1, dim=0)  # 将batch_size维度分开,得到的是一个turple,每个元素的batch_size=1
    batch_num = len(pred_image_images)

    for ind in range(batch_num):
        image_name_1 = image_name[ind].split('/')[-1]  # 将最后一块切出来
        print(image_name_1)
        utils.save_image(pred_image_images[ind], './results_real/{}/{}/{}'.format(category, exp_name, image_name_1))

def print_log(epoch, num_epochs, one_epoch_time, train_psnr, train_ssim, val_psnr, val_ssim,
              old_val_psnr1, exp_name, lr):
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Train_SSIM:{4:.4f}, '
          'Val_PSNR:{5:.2f}, Val_SSIM:{6:.4f}, Best_PSNR:{7:.2F}\n '
          'learning rate sets to {8:.10f}\n '
          .format(one_epoch_time, epoch, num_epochs, train_psnr, train_ssim, val_psnr, val_ssim,
                  old_val_psnr1, lr))

    # --- Write the training log --- #
    with open('./training_log/{}_log.txt'.format(exp_name), 'a') as f:
        print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Train_SSIM:{5:.4f}, '
              'Val_PSNR: {6:.2f}, Val_SSIM: {7:.4f}, Best_PSNR:{8:.2F}, '
              'learning rate sets to {9:.10f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, train_psnr, train_ssim, val_psnr, val_ssim,
                      old_val_psnr1, lr), file=f)

def adjust_learning_rate(optimizer, epoch,  lr_decay=0.5):
    # --- Decay learning rate --- #
    step = 50

    if not epoch % step and epoch > 50:  # 当整除时epoch % step = 0; 前面加上not就为1
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))
