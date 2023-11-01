import cv2
import argparse
import yaml
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels,random_add_gaussian_noise_pt,random_add_poisson_noise_pt
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from torch.utils import data as data
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from collections import OrderedDict
from torch.nn import functional as F 
import  scipy.io as sio


def ordered_yaml():
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper



def yaml_load(f):
    if os.path.isfile(f):
        with open(f, 'r') as f:
            return yaml.load(f, Loader=ordered_yaml()[0])
    else:
        return yaml.load(f, Loader=ordered_yaml()[0])
    

# 解析yml文件
def parse_options(root_path):
    #parser = argparse.ArgumentParser()
    #parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
#
    #args = parser.parse_args()
#
    #opt = yaml_load(args.opt)
    opt = yaml_load(root_path)

    return opt

def augment_img(data, opt):
    gt = data['gt'].cuda()
    gt = torch.unsqueeze(gt, 0)
    usm_sharper = USMSharp().cuda()
    jpeger = DiffJPEG(differentiable=False).cuda()
    print(gt.shape)
    gt_usm = usm_sharper(gt)
    kernel1 = data['kernel1'].cuda()
    kernel2 = data['kernel2'].cuda()
    sinc_kernel = data['sinc_kernel'].cuda()
    save_path = data['gt_path'].split('/')[-1]

    ori_h, ori_w = gt_usm.size()[2:4]


    # blur
    out = filter2D(gt_usm, kernel1)
    # random_resize
    updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob'])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, opt['resize_range'][1])
    elif updown_type == 'down':
        scale = np.random.uniform(opt['resize_range'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, scale_factor=scale, mode=mode)
    # add noise
    gray_noise_prob = opt['gray_noise_prob']
    if np.random.uniform() < opt['gaussian_noise_prob']:
        out = random_add_gaussian_noise_pt(
            out, sigma_range=opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob
        )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=opt['poisson_scale_range'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False
        )
    # JPEG compression
    #jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range'])
    #out = torch.clamp(out, 0, 1)
    #out = jpeger(out, quality=jpeg_p)
    # ----------------------- The second degradation process ----------------------- #
    # blur
    if np.random.uniform() < opt['second_blur_prob']:
        out = filter2D(out, kernel2)
    # random resize
    updown_type = random.choices(['up', 'down','keep'], opt['resize_prob2'])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, opt['resize_range2'][1])
    elif updown_type == 'down':
        scale = np.random.uniform(opt['resize_range2'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(
        out, size=(int(ori_h / opt['scale'] * scale), int(ori_w / opt['scale'] * scale)), mode=mode
    )
    # add noise
    gray_noise_prob = opt['gray_noise_prob2']
    if np.random.uniform() < opt['gaussian_noise_prob2']:
        out = random_add_gaussian_noise_pt(
            out, sigma_range=opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob
        )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=opt['poisson_scale_range2'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False
        )
    if np.random.uniform() < 0.5:
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode=mode) # 整除
        out = filter2D(out, sinc_kernel)
        # JPEG compression
        #jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
        #out = torch.clamp(out, 0, 1)
        #out = self.jpeger(out, quality=jpeg_p)
    else:
        # JPEG compression
        #jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
        #out = torch.clamp(out, 0, 1)
        #out = self.jpeger(out, quality=jpeg_p)
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode=mode)
        out = filter2D(out, sinc_kernel)
    out_1 = F.interpolate(
        out,
        size=(512,512),
        mode='bicubic',
    )
    
    out_1 = torch.clamp((out_1 * 255.0).round(), 0, 255) / 255.
    output_img = out_1.data.squeeze().float().cpu().clamp_(0,1).numpy()
    output_img = np.transpose(output_img[[2,1,0], :,:], (1,2,0))
    output = (output_img * 255.0).round().astype(np.uint8)
    cv2.imwrite('./CFW_trainingdata/inputs/' + save_path.split('.')[0] + '.png', output)

    #out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    #output_img = out.data.squeeze().float().cpu().clamp_(0,1).numpy()
    #output_img = np.transpose(output_img[[2,1,0], :,:], (1,2,0))
    #output = (output_img * 255.0).round().astype(np.uint8)
    #cv2.imwrite('./inputs_128/' + save_path.split('.')[0] + '.png', output)

    #output_img = torch.clamp((gt_usm * 255.0).round(), 0, 255) / 255.
    #output_img = output_img.data.squeeze().float().cpu().clamp_(0,1).numpy()
    #output_img = np.transpose(output_img[[2,1,0], :,:], (1,2,0))
    #output = (output_img * 255.0).round().astype(np.uint8)
    #cv2.imwrite('./gts/' + save_path, output)



def prepare_img(opt, pic_nums):
    
    for phase, dataset_opt in opt['datasets'].items():
        print(dataset_opt)
        io_backend_opt = dataset_opt['io_backend']

        with open(dataset_opt['meta_info']) as fin:
            paths = [line.strip().split(' ')[0] for line in fin]
            h_paths = [os.path.join(dataset_opt['dataroot_gt'], v) for v in paths]



        blur_kernel_size = dataset_opt['blur_kernel_size']
        kernel_list = dataset_opt['kernel_list']
        kernel_prob = dataset_opt['kernel_prob']  # a list for each kernel probability
        blur_sigma = dataset_opt['blur_sigma']
        betag_range = dataset_opt['betag_range']  # betag used in generalized Gaussian blur kernels
        betap_range = dataset_opt['betap_range']  # betap used in plateau blur kernels
        sinc_prob = dataset_opt['sinc_prob']  # the probability for sinc filters

        blur_kernel_size2 = dataset_opt['blur_kernel_size2']
        kernel_list2 = dataset_opt['kernel_list2']
        kernel_prob2 = dataset_opt['kernel_prob2']
        blur_sigma2 = dataset_opt['blur_sigma2']
        betag_range2 = dataset_opt['betag_range2']
        betap_range2 = dataset_opt['betap_range2']
        sinc_prob2 = dataset_opt['sinc_prob2']

        final_sinc_prob = dataset_opt['final_sinc_prob']
        kernel_range = [2 * v + 1 for v in range(3, 11)] 
        pulse_tensor = torch.zeros(21, 21).float()
        pulse_tensor[10, 10] = 1
        file_client = FileClient(io_backend_opt.pop('type'), **io_backend_opt)
        for i in range(pic_nums):
        
            gt_path = h_paths[i]
            print(gt_path)
            im_bytes = file_client.get(gt_path, 'gt')

            img_gt = imfrombytes(im_bytes, float32=True)

            img_gt = augment(img_gt, dataset_opt['use_hflip'], dataset_opt['use_rot'])

            kernel_size = random.choice(kernel_range)
            if np.random.uniform() < dataset_opt['sinc_prob']:
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel = random_mixed_kernels(
                    kernel_list,
                    kernel_prob,
                    kernel_size,
                    blur_sigma,
                    blur_sigma, [-math.pi, math.pi],
                    betag_range,
                    betap_range,
                    noise_range=None
                )

            pad_size = (21 - kernel_size) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

            kernel_size = random.choice(kernel_range)
            if np.random.uniform() < dataset_opt['sinc_prob2']:
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)

            else:
                kernel2 = random_mixed_kernels(
                    kernel_list2,
                    kernel_prob2,
                    kernel_size,
                    blur_sigma2,
                    blur_sigma2, [-math.pi, math.pi],
                    betag_range2,
                    betap_range2,
                    noise_range=None
                )

            pad_size = (21 - kernel_size) // 2
            kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

            if np.random.uniform() < dataset_opt['final_sinc_prob']:
                kernel_size = random.choice(kernel_range)
                omega_c = np.random.uniform(np.pi / 3, np.pi)
                sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
                sinc_kernel = torch.FloatTensor(sinc_kernel)
            else:
                sinc_kernel = pulse_tensor

            # BGR to RGB, HWC to CHW
            img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
            kernel = torch.FloatTensor(kernel)
            kernel2 = torch.FloatTensor(kernel2)

            return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'gt_path': gt_path}
            augment_img(return_d, opt)





if __name__ == '__main__':
    src_path = '/home/sr/Real-ESRGAN-master/options/train_realesrgan_x4plus.yml'
    opt = parse_options(src_path)
    prepare_img(opt, 1000) # 



