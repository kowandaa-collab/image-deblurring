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
from MIMO_UNet.models.MIMOUNetBlurDM import build_MIMOUnet_net
from MIMO_UNet.models.LatentEncoder import LE_arch
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
# torch.rfft was deprecated in newer PyTorch versions. 
# This block creates a backward-compatible rfft wrapper using the newer torch.fft module.
try:
    from torch import rfft
except ImportError:
    def rfft(x, d):
        # Perform real-to-complex FFT and stack real/imaginary parts along the last dimension
        t = torch.fft.fft(x, dim=(-d))
        r = torch.stack((t.real, t.imag), -1)
        return r

class Trainer():
    """
    Main Trainer class handling the training loop, validation loop, model saving, and logging.
    """
    def __init__(self, dataloader_train, dataloader_val, model, model_le, optimizer, scheduler, args, writer) -> None:
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.model = model             # MIMO-UNet Deblurring Model
        self.model_le = model_le       # Latent Encoder Model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.writer = writer           # Tensorboard writer
        self.epoch = 0
        self.device = self.args.device
        
        # Initialize Image Quality Assessment metrics via pyiqa
        self.psnr_func = pyiqa.create_metric('psnr', device=device)
        self.lpips_func = pyiqa.create_metric('lpips', device=device)
        self.best_psnr = args.best_psnr if hasattr(args, 'best_psnr') else 0
        self.grad_clip = 1

        # Setup learning rate scheduler bounds
        self.scheduler.T_max = self.args.end_epoch
        
        # Select the criterion (loss function) based on user arguments
        if args.criterion == "l1":
            self.criterion = CharbonnierLoss() # A differentiable approximation of L1 loss
        elif args.criterion == "perceptual":
            self.criterion = VGGPerceptualLoss().to(device)
        elif args.criterion == "l1perceptual":
             self.criterion = L1andPerceptualLoss(gamma=args.gamma).to(device)
        else:
            raise ValueError(f"criterion not supported {args.criterion}")
        
    def train(self):
        """Main training loop iterating over epochs."""
        # Only print global training info on the master process (Rank 0) to avoid duplicated terminal output
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

        # Iterate from the starting epoch to the end epoch
        for epoch in range(args.start_epoch, args.end_epoch + 1):
            self.epoch = epoch
            self._train_epoch()

            # Rank 0 handles validation, image saving, and checkpoint saving
            if dist.get_rank() == 0:
                # Perform validation at specified intervals or at the very end
                if (epoch % self.args.validation_epoch) == 0 or epoch == self.args.end_epoch:
                    self.valid()

                # Save sample output images to disk at specified intervals
                if(self.args.val_save_epochs > 0 and epoch % self.args.val_save_epochs == 0 or epoch == self.args.end_epoch):
                    self.val_save_image(dir_path=self.args.dir_path, dataset=self.dataloader_val.dataset)

                # Save model checkpoints
                self.save_model()
    
    def _train_epoch(self):
        """Handles a single epoch of training."""
        # Ensure the distributed sampler shuffles data differently each epoch
        train_sampler.set_epoch(self.epoch)
        
        tq = tqdm.tqdm(self.dataloader_train, total=len(self.dataloader_train))
        tq.set_description(f'Epoch [{self.epoch}/{self.args.end_epoch}] training')
        
        # Track moving averages of losses and metrics
        total_train_loss = AverageMeter()
        total_train_psnr = AverageMeter()
        total_train_lpips = AverageMeter()
        
        for idx, sample in enumerate(tq):
            self.model.train()
            self.model_le.train()
            self.optimizer.zero_grad() # Clear previous gradients
            
            # Fetch blurred and sharp (ground truth) images
            blur, sharp = sample['blur'].to(device), sample['sharp'].to(device)
            
            # Forward pass: Generate latent feature from Encoder, then pass to MIMO-UNet
            z_pred = self.model_le(blur, sharp)
            outputs = self.model(blur, z_pred)
            
            # Clamp outputs to valid image range [-0.5, 0.5]
            outputs =  [output.clamp(-0.5, 0.5) for output in outputs]   # [B, C, H, W]
            
            # MIMO-UNet outputs multi-scale predictions. Generate multi-scale Ground Truths.
            gt_img2 = F.interpolate(sharp, scale_factor=0.5, mode='bilinear')  # 1/2 scale
            gt_img4 = F.interpolate(sharp, scale_factor=0.25, mode='bilinear') # 1/4 scale
            
            # Calculate content loss at all 3 scales
            l1 = self.criterion(outputs[0], gt_img4)
            l2 = self.criterion(outputs[1], gt_img2)
            l3 = self.criterion(outputs[2], sharp)
            loss_content = l1+l2+l3

            # Calculate Frequency (FFT) loss at all 3 scales to improve high-frequency detail restoration
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

            # Total loss is content loss plus a weighted FFT loss
            loss = loss_content + 0.1 * loss_fft
            
            # Backpropagation
            loss.backward()

            # Optional: Gradient clipping to prevent exploding gradients (currently commented out)
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            # Update weights
            self.optimizer.step()

            # Update metrics
            total_train_loss.update(loss.detach().item())
            psnr = calc_psnr(outputs[2].detach(), sharp.detach()) # PSNR on full-scale output
            total_train_psnr.update(psnr)

            # Update progress bar
            tq.set_postfix({'loss': total_train_loss.avg, 'psnr': total_train_psnr.avg, 'lpips': total_train_lpips.avg,'lr': optimizer.param_groups[0]['lr']})

        # Update learning rate schedule at the end of the epoch
        if self.scheduler:
            self.scheduler.step()
            
        # Log to Tensorboard and text log on Master process
        if self.writer and dist.get_rank() == 0:
            self.writer.add_scalar('Loss/Train_loss', total_train_loss.avg, self.epoch)
            self.writer.add_scalar('Loss/Train_psnr', total_train_psnr.avg, self.epoch)
            self.writer.add_scalar('Loss/Train_lpips', total_train_lpips.avg, self.epoch)
            logging.info(
                f'Epoch [{self.epoch}/{args.end_epoch}]: Train_loss: {total_train_loss.avg:.4f} Train_psnr:{total_train_psnr.avg:.4f} Train_lpips:{total_train_lpips.avg:.4f}')
    
    @torch.no_grad() # Disable gradient calculation for validation to save memory/compute
    def _valid(self, blur, sharp):
        """Processes a single validation batch and returns metrics."""
        self.model.eval()
        self.model_le.eval()
        
        # Forward Pass
        z_pred = self.model_le(blur, sharp)
        outputs = self.model(blur, z_pred)
        outputs =  [output.clamp(-0.5, 0.5) for output in outputs]
        
        # Multi-scale setup for Loss calculation
        gt_img2 = F.interpolate(sharp, scale_factor=0.5, mode='bilinear')
        gt_img4 = F.interpolate(sharp, scale_factor=0.25, mode='bilinear')
        
        # Losses
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
        
        # Calculate full-scale visual metrics (shift from [-0.5, 0.5] to [0, 1] range)
        psnr = torch.mean(self.psnr_func(outputs[2].detach()+0.5, sharp.detach()+0.5)).item()
        lpips = torch.mean(self.lpips_func(outputs[2].detach()+0.5, sharp.detach()+0.5)).item()
        return psnr, lpips, loss.item()
    
    @torch.no_grad()
    def valid(self):
        """Loops through the validation dataset and logs final performance."""
        self.model.eval()
        self.model_le.eval()
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

        # Log validation metrics
        self.writer.add_scalar('Val/Test_lpips', total_val_lpips.avg, self.epoch)
        self.writer.add_scalar('Val/Test_psnr', total_val_psnr.avg, self.epoch)
        self.writer.add_scalar('Val/Test_loss', total_val_loss.avg, self.epoch)
        logging.info(
            f'Crop Validation Epoch [{self.epoch}/{args.end_epoch}]: Test Loss: {total_val_loss.avg:.4f} Test lpips: {total_val_lpips.avg:.4f} Test psnr:{total_val_psnr.avg:.4f}')
        
        # Check if current model is the best performing based on PSNR and save it
        if self.best_psnr < total_val_psnr.avg:
            self.best_psnr = total_val_psnr.avg
            args.best_psnr = self.best_psnr
            
            # Save MIMO model
            best_state = {'model_state': self.model.module.state_dict(), 'args': args}
            torch.save(best_state, os.path.join(args.dir_path, 'best_deblur_{}.pth'.format(args.model_name)))

            # Save Latent Encoder
            best_state = {'model_le_state': self.model_le.module.state_dict(), 'args': args}
            torch.save(best_state, os.path.join(args.dir_path, 'best_le_{}.pth'.format(args.model_name)))

            print('Saving model with best PSNR {:.3f}...'.format(self.best_psnr))
            logging.info('Saving model with best PSNR {:.3f}...'.format(self.best_psnr))
            
    def save_model(self):
        """Saves current state of the model, optimizer, and scheduler."""
        # Create a comprehensive state dict for seamless resuming
        training_state = {'epoch': self.epoch, 
                          'model_state': self.model.module.state_dict(),
                          'model_le_state': self.model_le.module.state_dict(),
                          'optimizer_state': self.optimizer.state_dict(),
                          'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
                          'best_panr': self.best_psnr,
                          'args': args}
        
        # Overwrite the latest checkpoint
        torch.save(training_state, os.path.join(self.args.dir_path, 'last_{}.pth'.format(self.args.model_name)))

        # Save milestone checkpoints periodically
        if (self.epoch % self.args.check_point_epoch) == 0:
            torch.save(training_state, os.path.join(self.args.dir_path, 'epoch_{}_{}.pth'.format(self.epoch, self.args.model_name)))

        # Save the final model cleanly at the end of training
        if self.epoch == self.args.end_epoch:
            model_state = {'model_state': self.model.module.state_dict(), 'args': args}
            torch.save(model_state, os.path.join(args.dir_path, 'final_deblur_{}.pth'.format(args.model_name)))

            model_state = {'model_le_state': self.model_le.module.state_dict(), 'args': args}
            torch.save(model_state, os.path.join(args.dir_path, 'final_le_{}.pth'.format(args.model_name)))

    @torch.no_grad()
    def val_save_image(self, dir_path, dataset, val_num=3):
        """Randomly selects images from the validation set, runs inference, and saves the output locally."""
        os.makedirs(dir_path, exist_ok=True)
        self.model.eval()
        self.model_le.eval()
        
        # Pick 'val_num' random samples
        for idx in random.sample(range(0, len(dataset)), val_num):
            sample = dataset[idx]
            blur, sharp = sample['blur'].unsqueeze(0).to(device), sample['sharp'].unsqueeze(0).to(device)
            b, c, h, w = blur.shape
            
            # Pad images to be divisible by the network's max downsampling factor (8 in MIMO-UNet)
            factor = 8
            h_n = (factor - h % factor) % factor
            w_n = (factor - w % factor) % factor
            blur = torch.nn.functional.pad(blur, (0, w_n, 0, h_n), mode='reflect')
            sharp_in = torch.nn.functional.pad(sharp, (0, w_n, 0, h_n), mode='reflect')
            
            # Forward pass
            z_pred = self.model_le(blur, sharp_in)
            output = self.model(blur, z_pred) 
            
            # Crop padding off the final full-scale prediction (output[2])
            output = output[2][:, :, :h, :w]
            output = output.clamp(-0.5, 0.5) 

            # Create visualization directories
            save_img_dir_path = os.path.join(dir_path, f'visualization', 'output')
            os.makedirs(save_img_dir_path, exist_ok=True)
            save_sharp_dir_path = os.path.join(dir_path, f'visualization', 'sharp')
            os.makedirs(save_sharp_dir_path, exist_ok=True)

            # Convert tensors back to OpenCV images (numpy arrays) and save
            save_img_path = os.path.join(save_img_dir_path, f'{self.epoch:05d}_{idx:05d}.png')
            output = tensor2cv(output + 0.5)
            cv2.imwrite(save_img_path, output)

            save_sharp_path = os.path.join(save_sharp_dir_path, f'{self.epoch:05d}_{idx:05d}.png')
            sharp = tensor2cv(sharp + 0.5)
            cv2.imwrite(save_sharp_path, sharp)

if __name__ == "__main__":
    # --- Hyperparameters and Command Line Arguments Parsing ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--end_epoch", default=2, type=int)
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
    parser.add_argument("--data_path", default='/home/jthe/DeblurDM/dataset/GOPRO_Large', type= str)
    parser.add_argument("--dir_path", default='./experiments/MIMO_UNet/GoPro/stage1', type=str)
    parser.add_argument("--model_name", default='MIMOUNetBlurDM', type=str)
    parser.add_argument("--model", default='MIMOUNetBlurDM', type=str)
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--val_save_epochs", default=100, type=int)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--only_use_generate_data", action='store_true', help="only use generated data to train model.")
    parser.add_argument("--num_workers", default=0 if os.name == "nt" else 8, type=int)
    # LOCAL_RANK is automatically injected by PyTorch Distributed Data Parallel launchers
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    
    args = parser.parse_args()

    # --- Distributed Data Parallel (DDP) Initialization ---
    if args.local_rank != -1: # Multi-GPU Training path
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
        else:
            device = torch.device("cpu")
        # NCCL backend is highly recommended for Nvidia GPUs; gloo is fallback for CPU/Windows
        backend = "nccl" if (torch.cuda.is_available() and dist.is_nccl_available() and os.name != "nt") else "gloo"
        dist.init_process_group(backend=backend, init_method='env://')
    else: # Single-GPU/CPU debugging fallback path
        if torch.cuda.is_available():
            args.local_rank = 0
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
        else:
            args.local_rank = 0
            device = torch.device("cpu")
        # Still mock a distributed environment with World Size 1 so DDP wrappers don't crash
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
    net_le = LE_arch()

    # --- Seed Everything for Reproducibility ---
    # Adding local rank to the seed ensures each GPU processes slightly different augmentations/noise
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
    
    # Move models to the allocated device
    net.to(device)
    net_le.to(device)

    print(args.__dict__.items())

    # --- Collect Parameters for the Optimizer ---
    optim_params = []
    for k, v in net.named_parameters():
        if v.requires_grad:
            optim_params.append(v)

    for k, v in net_le.named_parameters():
        if v.requires_grad:
            optim_params.append(v)

    # --- Initialize Optimizer and Scheduler ---
    if args.optimizer == "adam":
        optimizer = optim.Adam([{'params': optim_params}], lr=args.init_lr)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW([{'params': optim_params}], lr=args.init_lr, weight_decay=1e-4)
    else:
        raise ValueError(f"optimizer not supported {args.optimizer}")

    # Cosine Annealing reduces the learning rate to eta_min following a cosine curve
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.end_epoch, eta_min=args.min_lr)
        
    # --- Load Checkpoints/Pretrained Weights ---
    # Map weights onto the proper GPU index to avoid VRAM spikes on GPU 0
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    
    # Attempt to load the last training state automatically
    if os.path.exists(os.path.join(args.dir_path, 'last_{}.pth'.format(args.model_name))):
        print('load_pretrained')
        training_state = (torch.load(os.path.join(args.dir_path, 'last_{}.pth'.format(args.model_name)), map_location=map_location))
        
        # Resume training step
        args.start_epoch = training_state['epoch'] + 1
        if 'best_psnr' in training_state['args']:
            args.best_psnr = training_state['args'].best_psnr
            
        # Load Deblur Model Weights
        new_weight = net.state_dict()
        # judge_and_remove_module_dict strips "module." prefix added by DDP if necessary
        training_state["model_state"] = judge_and_remove_module_dict(training_state["model_state"])
        new_weight.update(training_state['model_state'])
        net.load_state_dict(new_weight)

        # Load Latent Encoder Weights
        new_weight = net_le.state_dict()
        training_state["model_le_state"] = judge_and_remove_module_dict(training_state["model_le_state"])
        new_weight.update(training_state['model_le_state'])
        net_le.load_state_dict(new_weight)

        # Load Optimizer & Scheduler states to preserve momentum/LR curves
        new_optimizer = optimizer.state_dict()
        new_optimizer.update(training_state['optimizer_state'])
        optimizer.load_state_dict(new_optimizer)
        
        new_scheduler = scheduler.state_dict()
        new_scheduler.update(training_state['scheduler_state'])
        scheduler.load_state_dict(new_scheduler)
        
    # Explicitly resuming from a specific path
    elif args.resume:
        print('load_resume_pretrained')
        model_load = torch.load(args.resume, map_location=map_location)
        
        model_load["model_state"] = judge_and_remove_module_dict(model_load["model_state"])
        net.load_state_dict(model_load['model_state'])

        model_load["model_le_state"] = judge_and_remove_module_dict(model_load["model_le_state"])
        net_le.load_state_dict(model_load['model_le_state'])
        os.makedirs(args.dir_path, exist_ok=True)
    else:
        # First time training initialization
        os.makedirs(args.dir_path, exist_ok=True)
    
    # --- Wrap Models in Distributed Data Parallel ---
    # Synchronizes gradients across multiple GPUs automatically during backward pass
    net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank],
                                          output_device=args.local_rank)
    net_le = nn.parallel.DistributedDataParallel(net_le, device_ids=[args.local_rank],
                                          output_device=args.local_rank)
                                          
    # --- Datasets and DataLoaders Setup ---
    dataset_name = args.data_path.split('/')[-1]
    train_data_path = args.data_path

    # Instantiate the correct Dataset object based on folder name
    if dataset_name == "GOPRO_Large":
        Train_set = Multi_GoPro_Loader(data_path=train_data_path, mode="train", crop_size=args.crop_size)
    elif (dataset_name == "Realblur_J") or (dataset_name == "Realblur_R"):
        Train_set = RealBlur_Loader(data_path=train_data_path, mode="train", crop_size=args.crop_size, ZeroToOne=False)
    
    # DistributedSampler ensures different GPUs see different subsets of the data without overlap
    train_sampler = DistributedSampler(Train_set)
    dataloader_train = DataLoader(
        Train_set,
        sampler=train_sampler,
        batch_size=args.batch_size // num_gpus, # Divide batch globally across GPUs
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),   # Faster CPU-to-GPU data transfer
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
        # Standard python text logger
        logging.basicConfig(
            filename=os.path.join(args.dir_path, 'train.log') , format='%(levelname)s:%(message)s', encoding='utf-8', level=logging.INFO)
        
        # Log metadata for historical tracking
        logging.info(f'args: {args}')
        logging.info(f'model: {net}')
        logging.info(f'latent encoder: {net_le}')
        logging.info(f'model parameters: {count_parameters(net)}')
        logging.info(f'latent encoder parameters: {count_parameters(net_le)}')
        logging.info(f"Optimizer:{optimizer.__class__.__name__}")
        logging.info(f"Train Data length:{len(dataloader_train.dataset)}")

        # Tensorboard Writer
        writer = SummaryWriter(os.path.join("MIMO_log", args.model_name))
        writer.add_text("args", str(args))

    # --- Start Training ---
    trainer = Trainer(dataloader_train, dataloader_val, net, net_le, optimizer, scheduler, args, writer)
    trainer.train()

    

    

