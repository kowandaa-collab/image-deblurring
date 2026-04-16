# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch
from torchvision.utils import save_image
import os
import sys
import tqdm
import argparse
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from dataloader import Test_Loader
from MIMO_UNet.models.MIMOUNetBlurDM import build_MIMOUnet_net
from utils.utils import same_seed, count_parameters, judge_and_remove_module_dict
from MIMO_UNet.models.LatentBlurDM import LatentExposureDiffusion

@torch.no_grad()
def predict(model, model_le, args, device):
    model.eval()
    model_le.eval()
    if args.dataset == 'GoPro+HIDE':
        dataset_name = ['GoPro', 'HIDE']
    else:
        dataset_name = [args.dataset]

    for val_dataset_name in dataset_name:
        dataset_path = os.path.join(args.data_path, val_dataset_name)

        dataset = Test_Loader(data_path=dataset_path,
                                crop_size=args.crop_size,
                                ZeroToOne=False)
        save_dir = os.path.join(args.dir_path, f'{val_dataset_name}')
        os.makedirs(save_dir, exist_ok=True)
        dataset_len = len(dataset)
        tq = tqdm.tqdm(range(dataset_len))
        tq.set_description(f'Predict {val_dataset_name}')

        for idx in tq:
            sample = dataset[idx]
            input = sample['blur'].unsqueeze(0).to(device)
            label = sample['sharp'].unsqueeze(0).to(device)

            b, c, h, w = input.shape
            factor=8
            h_n = (factor - h % factor) % factor
            w_n = (factor - w % factor) % factor
            input = torch.nn.functional.pad(input, (0, w_n, 0, h_n), mode='reflect')
            
            z_pred = model_le(input)
            output = model(input, z_pred)
            output = output[2][:, :, :h, :w]
            output = output.clamp(-0.5, 0.5)

            image_name = os.path.split(dataset.get_path(idx=idx)['blur_path'])[-1]
            save_img_path = os.path.join(save_dir, image_name)

            save_image(output.squeeze(0).cpu() + 0.5, save_img_path)



if __name__ == "__main__":
    # hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--data_path", default='./dataset/test', type=str)
    parser.add_argument("--dir_path", default='./results/MIMO_UNet/GoPro/', type=str)
    parser.add_argument("--model_path", default='./weights/MIMO_UNet/GoPro/MIMO_UNet_GoPro_stage3.pth', type=str)
    parser.add_argument("--model_dm_path", default='./weights/MIMO_UNet/GoPro/MIMO_UNet_GoPro_stage3.pth', type=str)
    parser.add_argument("--model", default='MIMOUNetBlurDM', type=str, choices=['MIMO-UNet', 'MIMO-UNetPlus'])
    parser.add_argument("--dataset", default='GoPro', type=str, choices=['GoPro+HIDE', 'GoPro', 'HIDE', 'Realblur_J', 'RealBlur_R', 'RWBI'])
    parser.add_argument("--crop_size", default=None, type=int)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device)
    # load_model_state = torch.load(args.model_path)

    if not os.path.isdir(args.dir_path):
        os.makedirs(args.dir_path)

    # Model and optimizer
    net = build_MIMOUnet_net(args.model)
    net_dm = LatentExposureDiffusion()
    
    load_model_state = torch.load(args.model_path)
    load_le_model_state = torch.load(args.model_dm_path)

    if 'model_state' in load_model_state.keys():
        load_model_state["model_state"] = judge_and_remove_module_dict(load_model_state["model_state"])
        net.load_state_dict(load_model_state["model_state"])
    elif 'model' in load_model_state.keys():
        load_model_state["model"] = judge_and_remove_module_dict(load_model_state["model"])
        net.load_state_dict(load_model_state["model"])
    else:
        load_model_state = judge_and_remove_module_dict(load_model_state)
        net.load_state_dict(load_model_state)

    # if 'model_dm_state' in load_model_state.keys():
    load_le_model_state["model_dm_state"] = judge_and_remove_module_dict(load_le_model_state["model_dm_state"])
    net_dm.load_state_dict(load_le_model_state["model_dm_state"])

    net.to(device)
    
    net_dm.to(device)

    print("device:", device)
    print(f'args: {args}')
    # print(f'model: {net}')
    print(f'model parameters: {count_parameters(net)}')

    same_seed(2023)
    predict(net, net_dm, args=args, device=device)





