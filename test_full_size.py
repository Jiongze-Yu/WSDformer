import time
import torch
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from utils.data_loader import getloader_test
from utils.val_function import val_full_size

from model.WSDformer_model import WSDformer

plt.switch_backend('agg')

# 设置超参数
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', type=str, default="WSDformer_Rain200H")
parser.add_argument('--test_data_path', type=str, default="./datasets/Rain200H/test")

args = parser.parse_args()

val_batch_size = args.val_batch_size
exp_name = args.exp_name

if __name__ == '__main__':
    # 定义gpu
    device_ids = [0]
    device = torch.device("cuda:0")

    # 定义网络
    net = WSDformer()

    # 多GPU训练
    net = net.to(device)
    net = nn.DataParallel(net, device_ids=device_ids)

    # 导入网络预训练权重
    try:
        # net.load_state_dict(torch.load('./weights/{}/best.pth'.format(exp_name)))
        net.load_state_dict(torch.load('./weights/{}.pth'.format(exp_name), map_location='cuda:0'))
        print('--- weight loaded ---')
    except:
        print('--- no weight loaded ---')

    # 导入测试数据
    val_data_loader1 = getloader_test(args)

    net.eval()

    # save_path = './result/' + exp_name
    # category = 'test_merge'
    # print('./results/{}/{}/'.format(category, exp_name))
    if os.path.exists('./results/{}/'.format(exp_name))==False:
        os.mkdir('./results/{}/'.format(exp_name))

    result_dir = './results/{}/'.format(exp_name)
    val_psnr1, val_ssim1 = val_full_size(val_data_loader1, net, result_dir, save_tag=True)

    # val_psnr1, val_ssim1 = validation(net, val_data_loader1, device, exp_name, category, save_tag=True)

    print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr1, val_ssim1))




