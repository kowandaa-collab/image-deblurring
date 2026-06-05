# --- Standard Library and Third-Party Imports ---
import random
import sys
import os
import argparse
import logging
import numpy as np
import tqdm
import cv2

# --- PyTorch Core and Distributed Training Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# --- Add Parent Directory to Path to load custom modules ---
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# --- Custom Module Imports ---
from dataloader import Multi_GoPro_Loader, RealBlur_Loader
from MIMO_UNet.models.MIMOUNetBlurDM import build_MIMOUnet_net # Main Deblurring Network
from MIMO_UNet.models.LatentBlurDM import LatentExposureDiffusion # Latent Predictor
from MIMO_UNet.models.losses import CharbonnierLoss, VGGPerceptualLoss, L1andPerceptualLoss
from utils.utils import calc_psnr, same_seed, count_parameters, tensor2cv, AverageMeter, judge_and_remove_module_dict

# --- Image Quality Assessment and Logging ---
import pyiqa
from tensorboardX import SummaryWriter

# --- Environment and CUDNN Configuration ---
# Prevent OpenCV from utilizing multiple threads, which can cause deadlocks in PyTorch DataLoaders
cv2.setNumThreads(0)
# Enable cuDNN optimizations for faster training on GPUs
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# --- FFT Compatibility Polyfill ---
# Provides a backward-compatible rfft function for frequency-domain loss calculations
try:
    from torch import rfft
except ImportError:
    def rfft(x, d):
        t = torch.fft.fft(x, dim=(-d))
        r = torch.stack((t.real, t.imag), -1)
        return r

class Trainer():
    """
    Main Trainer class handling Stage 3: Joint Fine-Tuning.
    This stage takes the pre-trained Diffusion Model (model_dm) and the pre-trained 
    MIMO-UNet (model), and trains them together end-to-end to optimize final image quality.
    """
    def __init__(self, dataloader_train, dataloader_val, model, model_dm, optimizer, scheduler, args, writer) -> None:
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.model = model       # MIMO-UNet Deblurring Model
        self.model_dm = model_dm # Latent Diffusion Model (Predictor)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.writer = writer
        self.epoch = 0
        self.device = self.args.device
        
        # Image quality metrics for validation
        self.psnr_func = pyiqa.create_metric('psnr', device=device)
        self.lpips_func = pyiqa.create_metric('lpips', device=device)
        self.best_psnr = args.best_psnr if hasattr(args, 'best_psnr') else 0
        self.grad_clip = 1

        self.scheduler.T_max = self.args.end_epoch
        
        # Select the criterion (loss function) for image reconstruction
        if args.criterion == "l1":
            self.criterion = CharbonnierLoss()
        elif args.criterion == "perceptual":
            self.criterion = VGGPerceptualLoss().to(device)
        elif args.criterion == "l1perceptual":
             self.criterion = L1andPerceptualLoss(gamma=args.gamma).to(device)
        else:
            raise ValueError(f"criterion not supported {args.criterion}")
        
    def train(self):
        """Main training loop iterating over epochs."""
        if dist.get_rank() == 0:
            print('###########################################')
            print('Start_Epoch:', self.args.start_epoch)
            print('End_Epoch:', self.args.end_epoch)
            print('Model:', self.args.model_name)
            print(f"Optimizer:{self.optimizer.__class__.__name__}")
            print(f"Scheduler:{self.scheduler.__class__.__name__ if self.scheduler else None}")
            print(f"Train Data length:{len(dataloader_train.dataset)}")
            print("start train !!")
            print('###########################################')

        for epoch in range(args.start_epoch, args.end_epoch + 1):
            self.epoch = epoch
            self._train_epoch()

            # Rank 0 handles validation, image saving, and checkpoint saving
            if dist.get_rank() == 0:
                if (epoch % self.args.validation_epoch) == 0 or epoch == self.args.end_epoch:
                    self.valid()

                if(self.args.val_save_epochs > 0 and epoch % self.args.val_save_epochs == 0 or epoch == self.args.end_epoch):
                    self.val_save_image(dir_path=self.args.dir_path, dataset=self.dataloader_val.dataset)

                self.save_model()
    
    def _train_epoch(self):
        """Handles a single epoch of joint training."""
        train_sampler.set_epoch(self.epoch)
        tq = tqdm.tqdm(self.dataloader_train, total=len(self.dataloader_train))
        tq.set_description(f'Epoch [{self.epoch}/{self.args.end_epoch}] training')
        
        total_train_loss = AverageMeter()
        total_train_psnr = AverageMeter()
        total_train_lpips = AverageMeter()
        
        for idx, sample in enumerate(tq):
            # BOTH models are set to train mode for joint optimization
            self.model.train()
            self.model_dm.train()
            self.optimizer.zero_grad()
            
            blur, sharp = sample['blur'].to(device), sample['sharp'].to(device)
            
            # Step 1: Diffusion model infers the latent vector ONLY from the blurred image
            z_pred = self.model_dm(blur)
            
            # Step 2: MIMO-UNet restores the image using the blurred image and predicted latent
            outputs = self.model(blur, z_pred)
            outputs =  [output.clamp(-0.5, 0.5) for output in outputs]   # [B, C, H, W]
            
            # Multi-scale Ground Truth preparation
            gt_img2 = F.interpolate(sharp, scale_factor=0.5, mode='bilinear')
            gt_img4 = F.interpolate(sharp, scale_factor=0.25, mode='bilinear')
            
            # Multi-scale content loss (comparing outputs to ground truth)
            l1 = self.criterion(outputs[0], gt_img4)
            l2 = self.criterion(outputs[1], gt_img2)
            l3 = self.criterion(outputs[2], sharp)
            loss_content = l1+l2+l3

            # Multi-scale frequency (FFT) loss
            label_fft1 = rfft(gt_img4, 2)
            pred_fft1 = rfft(outputs[0], 2)
            label_fft2 = rfft(gt_img2, 2)
            pred_fft2 = rfft(outputs[1], 2)
            label_fft3 = rfft(sharp, 2)
            pred_fft3 = rfft(outputs[2], 2)

            f1 = self.criterion(pred_fft1, label_fft1)
            f2 = self.criterion(pred_fft2, label_fft2)
            f3 = self.criterion(pred_fft3, label_fft3)
            loss_fft = f1+f2+f3

            # Total loss backpropagates through BOTH MIMO-UNet and the Diffusion model
            loss = loss_content + 0.1 * loss_fft
            loss.backward()

            self.optimizer.step()

            # Update metrics
            total_train_loss.update(loss.detach().item())
            psnr = calc_psnr(outputs[2].detach(), sharp.detach())
            total_train_psnr.update(psnr)

            tq.set_postfix({'loss': total_train_loss.avg, 'psnr': total_train_psnr.avg, 'lpips': total_train_lpips.avg,'lr': optimizer.param_groups[0]['lr']})

        if self.scheduler:
            self.scheduler.step()
            
        if self.writer and dist.get_rank() == 0:
            self.writer.add_scalar('Loss/Train_loss', total_train_loss.avg, self.epoch)
            self.writer.add_scalar('Loss/Train_psnr', total_train_psnr.avg, self.epoch)
            self.writer.add_scalar('Loss/Train_lpips', total_train_lpips.avg, self.epoch)
            logging.info(
                f'Epoch [{self.epoch}/{args.end_epoch}]: Train_loss: {total_train_loss.avg:.4f} Train_psnr:{total_train_psnr.avg:.4f} Train_lpips:{total_train_lpips.avg:.4f}')
    
    @torch.no_grad()
    def _valid(self, blur, sharp):
        """Processes a single validation batch and returns full pipeline metrics."""
        self.model.eval()
        self.model_dm.eval()
        
        # End-to-end inference
        z_pred = self.model_dm(blur)
        outputs = self.model(blur, z_pred)
        outputs =  [output.clamp(-0.5, 0.5) for output in outputs]
        
        # Loss calculation setup (Multi-scale)
        gt_img2 = F.interpolate(sharp, scale_factor=0.5, mode='bilinear')
        gt_img4 = F.interpolate(sharp, scale_factor=0.25, mode='bilinear')
        
        l1 = self.criterion(outputs[0], gt_img4)
        l2 = self.criterion(outputs[1], gt_img2)
        l3 = self.criterion(outputs[2], sharp)
        loss_content = l1+l2+l3

        label_fft1 = rfft(gt_img4, 2)
        pred_fft1 = rfft(outputs[0], 2)
        label_fft2 = rfft(gt_img2, 2)
        pred_fft2 = rfft(outputs[1], 2)
        label_fft3 = rfft(sharp, 2)
        pred_fft3 = rfft(outputs[2], 2)

        f1 = self.criterion(pred_fft1, label_fft1)
        f2 = self.criterion(pred_fft2, label_fft2)
        f3 = self.criterion(pred_fft3, label_fft3)
        loss_fft = f1+f2+f3

        loss = loss_content + 0.1 * loss_fft
        
        # Shift back to [0, 1] for metric calculation
        psnr = torch.mean(self.psnr_func(outputs[2].detach()+0.5, sharp.detach()+0.5)).item()
        lpips = torch.mean(self.lpips_func(outputs[2].detach()+0.5, sharp.detach()+0.5)).item()
        return psnr, lpips, loss.item()
    
    @torch.no_grad()
    def valid(self):
        """Loops through the validation dataset and logs final performance."""
        self.model.eval()
        self.model_dm.eval()
        total_val_psnr = AverageMeter()
        total_val_lpips = AverageMeter()
        total_val_loss = AverageMeter()
        
        tq = tqdm.tqdm(self.dataloader_val, total=len(self.dataloader_val))
        tq.set_description(f'Epoch [{self.epoch}/{self.args.end_epoch}] Validation')
        for idx, sample in enumerate(tq):
            blur, sharp = sample['blur'].to(device), sample['sharp'].to(device)
            psnr, lpips, loss = self._valid(blur, sharp)
            total_val_psnr.update(psnr)
            total_val_lpips.update(lpips)
            total_val_loss.update(loss)
            tq.set_postfix(LPIPS=total_val_lpips.avg, PSNR=total_val_psnr.avg, Loss=total_val_loss.avg)

        self.writer.add_scalar('Val/Test_lpips', total_val_lpips.avg, self.epoch)
        self.writer.add_scalar('Val/Test_psnr', total_val_psnr.avg, self.epoch)
        self.writer.add_scalar('Val/Test_loss', total_val_loss.avg, self.epoch)
        logging.info(
            f'Crop Validation Epoch [{self.epoch}/{args.end_epoch}]: Test Loss: {total_val_loss.avg:.4f} Test lpips: {total_val_lpips.avg:.4f} Test psnr:{total_val_psnr.avg:.4f}')
        
        # Check if the joint pipeline achieved a new best PSNR
        if self.best_psnr < total_val_psnr.avg:
            self.best_psnr = total_val_psnr.avg
            args.best_psnr = self.best_psnr
            
            # Save both fine-tuned models
            best_state = {'model_state': self.model.module.state_dict(), 'args': args}
            torch.save(best_state, os.path.join(args.dir_path, 'best_deblur_{}.pth'.format(args.model_name)))

            best_state = {'model_dm_state': self.model_dm.module.state_dict(), 'args': args}
            torch.save(best_state, os.path.join(args.dir_path, 'best_dm_{}.pth'.format(args.model_name)))

            print('Saving model with best PSNR {:.3f}...'.format(self.best_psnr))
            logging.info('Saving model with best PSNR {:.3f}...'.format(self.best_psnr))
            
    def save_model(self):
        """Saves current state of both models, optimizer, and scheduler."""
        training_state = {'epoch': self.epoch, 
                          'model_state': self.model.module.state_dict(),
                          'model_dm_state': self.model_dm.module.state_dict(),
                          'optimizer_state': self.optimizer.state_dict(),
                          'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
                          'best_panr': self.best_psnr,
                          'args': args}
        torch.save(training_state, os.path.join(self.args.dir_path, 'last_{}.pth'.format(self.args.model_name)))

        if (self.epoch % self.args.check_point_epoch) == 0:
            torch.save(training_state, os.path.join(self.args.dir_path, 'epoch_{}_{}.pth'.format(self.epoch, self.args.model_name)))

        if self.epoch == self.args.end_epoch:
            model_state = {'model_state': self.model.module.state_dict(), 'args': args}
            torch.save(model_state, os.path.join(args.dir_path, 'final_deblur_{}.pth'.format(args.model_name)))

            model_state = {'model_dm_state': self.model_dm.module.state_dict(), 'args': args}
            torch.save(model_state, os.path.join(args.dir_path, 'final_dm_{}.pth'.format(args.model_name)))

    @torch.no_grad()
    def val_save_image(self, dir_path, dataset, val_num=3):
        """Runs the full pipeline on a few random validation samples and saves the images."""
        os.makedirs(dir_path, exist_ok=True)
        self.model.eval()
        self.model_dm.eval()
        for idx in random.sample(range(0, len(dataset)), val_num):
            sample = dataset[idx]
            blur, sharp = sample['blur'].unsqueeze(0).to(device), sample['sharp'].unsqueeze(0).to(device)
            b, c, h, w = blur.shape
            
            # Padding for model compatibility
            factor = 8
            h_n = (factor - h % factor) % factor
            w_n = (factor - w % factor) % factor
            blur = torch.nn.functional.pad(blur, (0, w_n, 0, h_n), mode='reflect')
            
            # Predict Latent -> Reconstruct Image
            z_pred = self.model_dm(blur)
            output = self.model(blur, z_pred) 
            
            # Crop padding and clamp
            output = output[2][:, :, :h, :w]
            output = output.clamp(-0.5, 0.5) 

            save_img_dir_path = os.path.join(dir_path, f'visualization', 'output')
            os.makedirs(save_img_dir_path, exist_ok=True)
            save_sharp_dir_path = os.path.join(dir_path, f'visualization', 'sharp')
            os.makedirs(save_sharp_dir_path, exist_ok=True)

            # Convert to standard OpenCV format and write to disk
            save_img_path = os.path.join(save_img_dir_path, f'{self.epoch:05d}_{idx:05d}.png')
            output = tensor2cv(output + 0.5)
            cv2.imwrite(save_img_path, output)

            save_sharp_path = os.path.join(save_sharp_dir_path, f'{self.epoch:05d}_{idx:05d}.png')
            sharp = tensor2cv(sharp + 0.5)
            cv2.imwrite(save_sharp_path, sharp)

if __name__ == "__main__":
    # --- Hyperparameters and Command Line Arguments Parsing ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--end_epoch", default=3000, type=int)
    parser.add_argument("--start_epoch", default=1, type=int)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--crop_size", default=256, type=int)
    parser.add_argument("--validation_epoch", default=25, type=int)
    parser.add_argument("--check_point_epoch", default=100, type=int)
    parser.add_argument("--init_lr", default=1e-4, type=float)
    parser.add_argument("--min_lr", default=1e-6, type=float)
    parser.add_argument("--gamma", default=0.5, type=float)
    parser.add_argument("--optimizer", default='adam', type=str)
    parser.add_argument("--criterion", default='l1', type=str)
    parser.add_argument("--data_path", default='./dataset/GOPRO_Large', type= str)
    parser.add_argument("--dir_path", default='./experiments/MIMO_UNet/GoPro/stage3', type=str) # Stage 3 Path
    parser.add_argument("--model_name", default='MIMOUNetBlurDM', type=str)
    parser.add_argument("--model", default='MIMOUNetBlurDM', type=str)
    
    # CRUCIAL FOR STAGE 3: Paths to load weights from Stage 1 (model_path) and Stage 2 (model_dm_path)
    parser.add_argument("--model_dm_path", default=None, type=str)
    parser.add_argument("--model_path", default=None, type=str)
    
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--val_save_epochs", default=100, type=int)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--only_use_generate_data", action='store_true', help="only use generated data to train model.")
    parser.add_argument("--num_workers", default=0 if os.name == "nt" else 8, type=int)
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    
    args = parser.parse_args()

    # --- Distributed Data Parallel (DDP) Initialization ---
    if args.local_rank != -1:
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
        else:
            device = torch.device("cpu")
        backend = "nccl" if (torch.cuda.is_available() and dist.is_nccl_available() and os.name != "nt") else "gloo"
        dist.init_process_group(backend=backend, init_method='env://')
    else:
        if torch.cuda.is_available():
            args.local_rank = 0
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
        else:
            args.local_rank = 0
            device = torch.device("cpu")
        if not dist.is_initialized():
            master_port = os.getenv("MASTER_PORT", "29500")
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://127.0.0.1:{master_port}",
                rank=0,
                world_size=1,
            )

    # --- Initialize Models ---
    net = build_MIMOUnet_net(args.model)
    net_dm = LatentExposureDiffusion()

    # --- Load Pre-Trained Weights for Fine-Tuning ---
    # Load Stage 1 Deblur Model
    load_model_state = torch.load(args.model_path)
    # Load Stage 2 Latent Diffusion Model
    load_dm_model_state = torch.load(args.model_dm_path)

    # Strip DDP "module." prefixes and load weights into architectures
    load_model_state["model_state"] = judge_and_remove_module_dict(load_model_state["model_state"])
    net.load_state_dict(load_model_state["model_state"])
    
    load_dm_model_state["model_dm_state"] = judge_and_remove_module_dict(load_dm_model_state["model_dm_state"])
    net_dm.load_state_dict(load_dm_model_state["model_dm_state"])

    # --- Seed Everything for Reproducibility ---
    seed = args.seed + args.local_rank
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args.device = device
    print("device:", device)
    num_gpus = torch.cuda.device_count()
    net.to(device)
    net_dm.to(device)

    print(args.__dict__.items())

    # --- Collect Parameters for Joint Optimization ---
    # In Stage 3, we optimize the parameters of BOTH networks simultaneously
    optim_params = []
    for k, v in net.named_parameters():
        if v.requires_grad:
            optim_params.append(v)

    for k, v in net_dm.named_parameters():
        if v.requires_grad:
            optim_params.append(v)

    # --- Initialize Optimizer and Scheduler ---
    if args.optimizer == "adam":
        optimizer = optim.Adam([{'params': optim_params}], lr=args.init_lr)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW([{'params': optim_params}], lr=args.init_lr, weight_decay=1e-4)
    else:
        raise ValueError(f"optimizer not supported {args.optimizer}")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.end_epoch, eta_min=args.min_lr)

    # --- Load Stage 3 Resuming Checkpoints (if interrupted) ---
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    
    # Auto-resume from Stage 3's own 'last' checkpoint
    if os.path.exists(os.path.join(args.dir_path, 'last_{}.pth'.format(args.model_name))):
        print('load_pretrained')
        training_state = (torch.load(os.path.join(args.dir_path, 'last_{}.pth'.format(args.model_name)), map_location=map_location))
        args.start_epoch = training_state['epoch'] + 1
        if 'best_psnr' in training_state['args']:
            args.best_psnr = training_state['args'].best_psnr
            
        # Load Deblur Model State
        new_weight = net.state_dict()
        training_state["model_state"] = judge_and_remove_module_dict(training_state["model_state"])
        new_weight.update(training_state['model_state'])
        net.load_state_dict(new_weight)

        # Load Diffusion Model State
        new_weight = net_dm.state_dict()
        training_state["model_dm_state"] = judge_and_remove_module_dict(training_state["model_dm_state"])
        new_weight.update(training_state['model_dm_state'])
        net_dm.load_state_dict(new_weight)

        # Restore Optimizer and Scheduler
        new_optimizer = optimizer.state_dict()
        new_optimizer.update(training_state['optimizer_state'])
        optimizer.load_state_dict(new_optimizer)
        
        new_scheduler = scheduler.state_dict()
        new_scheduler.update(training_state['scheduler_state'])
        scheduler.load_state_dict(new_scheduler)
        
    # Manual resume logic
    elif args.resume:
        print('load_resume_pretrained')
        model_load = torch.load(args.resume, map_location=map_location)
        if 'model_state' in model_load.keys():
            model_load["model_state"] = judge_and_remove_module_dict(model_load["model_state"])
            net.load_state_dict(model_load['model_state'])

            model_load["model_dm_state"] = judge_and_remove_module_dict(model_load["model_dm_state"])
            net_dm.load_state_dict(model_load['model_dm_state'])
        os.makedirs(args.dir_path, exist_ok=True)
    else:
        os.makedirs(args.dir_path, exist_ok=True)
    
    # --- Wrap Models in Distributed Data Parallel ---
    net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank],
                                          output_device=args.local_rank)

    net_dm = nn.parallel.DistributedDataParallel(net_dm, device_ids=[args.local_rank],
                                          output_device=args.local_rank)
                                          
    # --- Datasets and DataLoaders Setup ---
    train_data_path = args.data_path
    if args.only_use_generate_data:
        train_data_path = None # Logic to fallback to an alternate dataset if specified
    
    dataset_name = args.data_path.split('/')[-1]
    if dataset_name == "GOPRO_Large":
        Train_set = Multi_GoPro_Loader(data_path=train_data_path, mode="train", crop_size=args.crop_size)
    elif (dataset_name == "Realblur_J") or (dataset_name == "Realblur_R"):
        Train_set = RealBlur_Loader(data_path=train_data_path, mode="train", crop_size=args.crop_size, ZeroToOne=False)
    
    train_sampler = DistributedSampler(Train_set)
    dataloader_train = DataLoader(
        Train_set,
        sampler=train_sampler,
        batch_size=args.batch_size // num_gpus,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Validation Set Loader Setup
    if dataset_name == "GOPRO_Large":
        Val_set = Multi_GoPro_Loader(data_path=args.data_path, mode="test", crop_size=args.crop_size)
    elif (dataset_name == "Realblur_J") or (dataset_name == "Realblur_R"):
        Val_set = RealBlur_Loader(data_path=train_data_path, mode="test", crop_size=args.crop_size, ZeroToOne=False)
    
    dataloader_val = DataLoader(
        Val_set,
        batch_size=args.batch_size // num_gpus,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    
    writer = None
    # --- Logging Setup (Only on Master Node/Rank 0) ---
    if dist.get_rank() == 0:
        logging.basicConfig(
            filename=os.path.join(args.dir_path, 'train.log') , format='%(levelname)s:%(message)s', encoding='utf-8', level=logging.INFO)
        
        logging.info(f'args: {args}')
        logging.info(f'model: {net}')
        logging.info(f'latent dm encoder: {net_dm}')
        logging.info(f'model parameters: {count_parameters(net)}')
        logging.info(f'latent dm encoder parameters: {count_parameters(net_dm)}')
        logging.info(f"Optimizer:{optimizer.__class__.__name__}")
        logging.info(f"Train Data length:{len(dataloader_train.dataset)}")

        writer = SummaryWriter(os.path.join("MIMO_log", args.model_name))
        writer.add_text("args", str(args))

    # --- Start Training ---
    trainer = Trainer(dataloader_train, dataloader_val, net, net_dm,optimizer, scheduler, args, writer)
    trainer.train()
