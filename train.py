import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
import random
from timm.scheduler import CosineLRScheduler
from utils.scheduler import CosineAnnealingLRWarmup
from torchvision.models import vgg16
from utils.utils import to_psnr, calc_ssim, print_log, validation, adjust_learning_rate
from utils.perceptual import TotalLoss, LossNetwork
from utils.ssim_loss import SSIM
from utils.data_loader import getloader, getloader_test
# from utils_merge.val_function import val_full_size

from model.WSDformer_model import WSDformer

plt.switch_backend('agg')

# 设置超参数
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-learning_rate', help='Set the learning rate', default=4e-4, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=128, nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=16, type=int)
parser.add_argument('-epoch_start', help='Starting epoch number of the training', default=0, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.2, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', default='WSDformer_rain200h', type=str)
parser.add_argument('-seed', help='set random seed', default=14, type=int)
parser.add_argument('-num_epochs', help='number of epochs', default=400, type=int)
parser.add_argument('-resume', help='Use checkpoint or not', default=False, type=bool)
parser.add_argument('-dis_freq', type=int, default=40)

parser.add_argument('--train_data_path', type=str, default="./datasets/Rain200H/train")
parser.add_argument('--test_data_path', type=str, default="./datasets/Rain200H/test")

args = parser.parse_args()

learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
epoch_start = args.epoch_start
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
exp_name = args.exp_name
num_epochs = args.num_epochs

if __name__ == '__main__':
    # 设置随机种子
    seed = args.seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        print('Seed:\t{}'.format(seed))

    # 打印超参数
    print('--- Hyper-parameters for training ---')
    print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nlambda_loss: {}'.format(learning_rate, crop_size,
          train_batch_size, val_batch_size, lambda_loss))

    # 定义gpu
    device_ids = [0, 1, 2, 3]
    device = torch.device("cuda:0")

    # 定义网络
    net = WSDformer()

    # 多GPU训练
    net = net.to(device)
    net = nn.DataParallel(net, device_ids=device_ids)

    # 导入网络预训练权重
    if os.path.exists('./weights/{}/'.format(exp_name))==False:
        os.mkdir('./weights/{}/'.format(exp_name))
    try:
        net.load_state_dict(torch.load('./weights/{}/best.pth'.format(exp_name), map_location='cuda:0'))
        print('--- weight loaded ---')
    except:
        print('--- no weight loaded ---')

    # 打印网络结构
    # print(net)

    # 计算网络参数数量
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    with open('./training_log/{}_log.txt'.format(exp_name), 'a') as f:
        print("Total_params: {}".format(pytorch_total_params), file=f)

    # 定义损失函数
    loss_network = SSIM().to(device)
    L1 = nn.L1Loss()
    L_mse = nn.MSELoss()

    # 导入训练数据/测试数据

    ## 导入训练数据
    lbl_train_data_loader = getloader(args)


    ## 在训练过程中测试数据
    val_data_loader1 = getloader_test(args)

    # 学习率更新（按iter）
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
    num_steps = len(lbl_train_data_loader) * num_epochs
    warmup_epochs = 5
    warmup_steps = len(lbl_train_data_loader) * warmup_epochs
    lr_scheduler_warmup = CosineAnnealingLRWarmup(optimizer,
                                                  T_max=num_steps,
                                                  eta_min=2.0e-5,
                                                  last_epoch=-1,
                                                  warmup_steps=warmup_steps,
                                                  warmup_start_lr=1.0e-6)

    # 计算训练前网络处理图像的PSNR和SSIM
    net.eval()
    category = 'test1'
    result_dir = './results/{}/{}/'.format(category, exp_name)

    # old_val_psnr1 = 0.
    # old_val_ssim1 = 0.
    # old_val_psnr1, old_val_ssim1 = val_full_size(val_data_loader1, net, result_dir)
    old_val_psnr1, old_val_ssim1 = validation(net, val_data_loader1, device, exp_name, category)
    print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr1, old_val_ssim1))
    with open('./training_log/{}_log.txt'.format(exp_name), 'a') as f:
        print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr1, old_val_ssim1), file=f)

    # 中断恢复
    log_dir = './weights/{}/checkpoint_log.pth'.format(exp_name)
    if args.resume:
            checkpoint = torch.load(log_dir, map_location='cuda:0')
            net.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch_start = checkpoint['epoch']
            lr_scheduler_warmup.load_state_dict(checkpoint['scheduler'])
            print('加载中断处模型成功，继续训练!')
    else:
            epoch_start = 0
            print('无保存模型，从头开始训练! ')

    # 开始训练
    net.train()
    run_loss = 0.0
    global_step = 0
    for epoch in range(epoch_start,num_epochs):
        print("####### Epoch:%d starts #######" % (epoch + 1))
        with open('./training_log/{}_log.txt'.format(exp_name), 'a') as f:
            print("####### Epoch:%d starts #######" % (epoch + 1), file=f)
        psnr_list = []
        ssim_list = []
        start_time = time.time()
        Dis_start = time.time()
        for batch_id, train_data in enumerate(lbl_train_data_loader):
            global_step += 1
            input_image, gt, imgid = train_data  # 从dataloader中获取输入图像，输出图像及图像名称
            input_image = input_image.to(device)
            gt = gt.to(device)

            # --- Zero the parameter gradients --- #
            optimizer.zero_grad()

            # --- Forward + Backward + Optimize --- #
            net.train()
            pred_image = net(input_image)

            loss = (-1 * loss_network(pred_image, gt) + 1) + L1(pred_image, gt) + L_mse(pred_image, gt)

            loss.backward()

            optimizer.step()

            # 学习率更新
            lr_scheduler_warmup.step()

            # 计算平均PSNR和SSIM
            psnr_list.extend(to_psnr(pred_image, gt))   # to_psnr为utils里定义的函数
            ssim_list.extend(calc_ssim(pred_image, gt))  # calc_ssim为utils里定义的函数

            run_loss += loss.item()
            if not (global_step % args.dis_freq):
                Dis_end = time.time()
                print('Epoch: {0}, Iteration: {1}， Loss: {2:.6f}, LR: {3:.10f}, Time_cost:{4:.3f}s'
                      .format(epoch+1, global_step, run_loss/args.dis_freq, lr_scheduler_warmup.get_last_lr()[0], Dis_end - Dis_start))
                with open('./training_log/{}_log.txt'.format(exp_name), 'a') as f:
                    print('Epoch: {0}, Iteration: {1}， Loss: {2:.6f}, LR: {3:.10f},  Time_cost:{4:.3f}s'.
                          format(epoch+1, global_step, run_loss / args.dis_freq, lr_scheduler_warmup.get_last_lr()[0], Dis_end - Dis_start), file=f)
                run_loss = 0.0
                Dis_start = Dis_end

        # 计算1个epoch后的PSNR, SSIM和LOSS
        train_psnr = sum(psnr_list) / len(psnr_list)
        train_ssim = sum(ssim_list) / len(ssim_list)

        # 保存网络的权重参数
        torch.save(net.state_dict(), './weights/{}/saved_model.pth'.format(exp_name))

        # 保存网络训练过程中的参数（用于中断恢复）
        state = {'model': net.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch,
                 'scheduler': lr_scheduler_warmup.state_dict()}
        torch.save(state, log_dir)


        # 每个epoch后计算测试数据集的PSNR和SSIM变化
        if (epoch % 10==0 or epoch<10 or epoch>350):
            net.eval()

            val_psnr1, val_ssim1 = validation(net, val_data_loader1, device, exp_name, category)
            one_epoch_time = time.time() - start_time

            print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, train_ssim, val_psnr1, val_ssim1,
                      old_val_psnr1, exp_name, optimizer.param_groups[0]['lr'])

            # 保留最好的结果
            if val_psnr1 >= old_val_psnr1:
                torch.save(net.state_dict(), './weights/{}/best.pth'.format(exp_name))
                print('model saved')
                old_val_psnr1 = val_psnr1