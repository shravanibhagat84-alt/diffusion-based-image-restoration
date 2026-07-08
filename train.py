"""
Training Script for Latent-IRSDE Image Restoration Model
Supports deblurring and deraining tasks with realistic degradation simulation.
"""
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import glob

from models.degradation_net import DegradationNet
from models.diffusion_unet import DiffusionUNet
from models.refinement_net import RefineNet, RefineNetWithSkipConnections
from models.autoencoder import Autoencoder, FrozenAutoencoder
from models.degradation import DegradationModel, DeblurSpecificModel, DerainSpecificModel
from losses import CombinedLoss
from utils import load_image


class ImageRestorationDataset(Dataset):
    """
    Dataset for image restoration training with realistic degradations.
    """
    def __init__(self, folder, image_size=256, task='deblurring'):
        """
        Args:
            folder: Path to folder containing training images
            image_size: Size to resize images to
            task: 'deblurring', 'deraining', or 'both'
        """
        # Search recursively in all subfolders for images
        self.paths = sorted(glob.glob(str(Path(folder) / "**" / "*.png"), recursive=True))
        if not self.paths:
            self.paths = sorted(glob.glob(str(Path(folder) / "**" / "*.jpg"), recursive=True))
        if not self.paths:
            self.paths = sorted(glob.glob(str(Path(folder) / "**" / "*.jpeg"), recursive=True))
        
        self.image_size = image_size
        self.task = task
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        # Load clean image
        img = load_image(self.paths[idx], size=(self.image_size, self.image_size))
        
        # Determine task for this sample
        if self.task == 'both':
            task = np.random.choice(['deblurring', 'deraining'])
        else:
            task = self.task
            
        return img, task


class HighResolutionDataset(Dataset):
    """
    Dataset that supports multiple resolutions for training.
    """
    def __init__(self, folder, resolutions=[256, 512, 1024], task='both'):
        # Search recursively in all subfolders for images
        self.paths = sorted(glob.glob(str(Path(folder) / "**" / "*.png"), recursive=True))
        if not self.paths:
            self.paths = sorted(glob.glob(str(Path(folder) / "**" / "*.jpg"), recursive=True))
            
        self.resolutions = resolutions
        self.task = task
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        # Random resolution
        size = np.random.choice(self.resolutions)
        
        # Load image at random resolution
        img = load_image(self.paths[idx], size=(size, size))
        
        # Determine task
        if self.task == 'both':
            task = np.random.choice(['deblurring', 'deraining'])
        else:
            task = self.task
            
        return img, task


class LatentIRSDEPipeline(nn.Module):
    """
    Complete Latent-IRSDE pipeline for training.
    Combines degradation prediction, latent diffusion, and refinement.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize networks
        self.degradation_net = DegradationNet(
            in_channels=3,
            base_channels=config.get('degradation_channels', 32),
            use_pretrained=config.get('use_pretrained_encoder', False)
        ).to(config['device'])
        
        self.diffusion = DiffusionUNet(
            in_channels=3,
            base_channels=config.get('diffusion_channels', 64),
            patch_size=config.get('patch_size', 8)
        ).to(config['device'])
        
        # Use RefineNet with skip connections for better quality
        if config.get('use_skip_connections', True):
            self.refiner = RefineNetWithSkipConnections(
                in_channels=3,
                base_channels=config.get('refiner_channels', 64)
            ).to(config['device'])
        else:
            self.refiner = RefineNet(
                in_channels=3,
                base_channels=config.get('refiner_channels', 32),
                use_cnn=config.get('use_cnn_refiner', True)
            ).to(config['device'])
        
        # Optional: Autoencoder for latent space processing
        if config.get('use_latent_diffusion', False):
            self.autoencoder = Autoencoder(
                in_channels=3,
                out_channels=3,
                base_channels=128,
                latent_channels=4
            ).to(config['device'])
            
            # Wrap in frozen autoencoder
            self.autoencoder = FrozenAutoencoder(
                self.autoencoder,
                freeze_encoder=True,
                freeze_decoder=True
            )
        else:
            self.autoencoder = None
        
    def forward(self, x, task='deblurring'):
        """
        Forward pass through the restoration pipeline.
        
        Args:
            x: Input degraded image (B, C, H, W)
            task: 'deblurring' or 'deraining'
            
        Returns:
            Restored image
        """
        # Step 1: DegradationNet - predict degradation mask
        mask = self.degradation_net(x)
        
        # Step 2: Apply mask and run diffusion model
        masked_img = x * mask
        
        # Optional: Encode to latent space
        if self.autoencoder is not None:
            with torch.no_grad():
                z = self.autoencoder.get_latent(masked_img)
                z = self.autoencoder.decode(z)
            coarse = self.diffusion.restore(z)
        else:
            coarse = self.diffusion.restore(masked_img)
        
        # Step 3: RefineNet - final restoration
        output = self.refiner(coarse)
        
        return output
    
    def restore(self, x, task='deblurring', steps=30):
        """
        Restoration with specified number of diffusion steps.
        """
        return self.forward(x, task)


def create_degraded_image(clean_img, task, device):
    """
    Create degraded image from clean image based on task.
    
    Args:
        clean_img: Clean image tensor (C, H, W)
        task: 'deblurring' or 'deraining'
        
    Returns:
        Degraded image tensor
    """
    # Add batch dimension
    if clean_img.dim() == 3:
        clean_img = clean_img.unsqueeze(0)
    
    # Move to device
    clean_img = clean_img.to(device)
    
    # Apply degradation based on task
    if task == 'deblurring':
        deblur_model = DeblurSpecificModel().to(device)
        degraded = deblur_model(clean_img)
    elif task == 'deraining':
        derain_model = DerainSpecificModel().to(device)
        degraded = derain_model(clean_img)
    else:
        # Default: use general degradation model
        deg_model = DegradationModel().to(device)
        degraded = deg_model(clean_img)
    
    # Remove batch dimension
    return degraded.squeeze(0)


def train_latent_irsde(config):
    """
    Main training function for Latent-IRSDE.
    
    Args:
        config: Dictionary containing training configuration
    """
    device = config['device']
    print(f"Training on device: {device}")
    
    # Create dataset
    if config.get('multi_resolution', False):
        dataset = HighResolutionDataset(
            folder=config['data_folder'],
            resolutions=config.get('resolutions', [256, 512, 1024]),
            task=config.get('task', 'both')
        )
    else:
        dataset = ImageRestorationDataset(
            folder=config['data_folder'],
            image_size=config.get('image_size', 256),
            task=config.get('task', 'both')
        )
    
    print(f"Dataset size: {len(dataset)} images")
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 2),
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model pipeline
    model = LatentIRSDEPipeline(config).to(device)
    
    # Create optimizer
    optimizer = AdamW(
        list(model.degradation_net.parameters()) +
        list(model.diffusion.parameters()) +
        list(model.refiner.parameters()),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.get('num_epochs', 100),
        eta_min=config.get('min_lr', 1e-6)
    )
    
    # Loss function
    criterion = CombinedLoss(
        lambda_l1=config.get('lambda_l1', 1.0),
        lambda_ssim=config.get('lambda_ssim', 0.5)
    )
    
    # Training loop
    num_epochs = config.get('num_epochs', 100)
    accumulation_steps = config.get('gradient_accumulation', 1)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, tasks) in enumerate(loader):
            # Move to device
            images = images.to(device)
            
            # Create degraded images on-the-fly
            degraded_images = []
            clean_targets = []
            
            for i in range(images.shape[0]):
                # Get task for this sample
                task = tasks[i] if isinstance(tasks, (list, tuple)) else tasks
                
                # Create degradation
                degraded = create_degraded_image(images[i], task, device)
                degraded_images.append(degraded)
                clean_targets.append(images[i])
            
            # Stack tensors
            degraded_batch = torch.stack(degraded_images)
            clean_batch = torch.stack(clean_targets)
            
            # Forward pass
            restored = model(degraded_batch, task='deblurring')
            
            # Compute loss
            loss = criterion(restored, clean_batch)
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.get('max_grad_norm', 1.0)
                )
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * accumulation_steps
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Batch [{batch_idx}/{len(loader)}] "
                      f"Loss: {loss.item() * accumulation_steps:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch summary
        avg_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_loss:.4f} "
              f"- LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % config.get('save_interval', 10) == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}.pt")
            print(f"Checkpoint saved: checkpoint_epoch_{epoch+1}.pt")
    
    # Save final model
    torch.save(model.state_dict(), "latent_irsde_final.pt")
    print("Training complete! Final model saved to latent_irsde_final.pt")
    
    return model


def main():
    """Main entry point for training."""
    # Configuration
    config = {
        'data_folder': 'dataset/Aerial_Landscapes',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 2,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'min_lr': 1e-6,
        'weight_decay': 0.01,
        'image_size': 256,
        'num_workers': 0,
        
        # Model configuration
        'degradation_channels': 32,
        'diffusion_channels': 64,
        'refiner_channels': 64,
        'patch_size': 8,
        
        # Training options
        'use_pretrained_encoder': False,
        'use_latent_diffusion': False,
        'use_skip_connections': True,
        'use_cnn_refiner': True,
        'multi_resolution': False,
        
        # Loss configuration
        'lambda_l1': 1.0,
        'lambda_ssim': 0.5,
        
        # Optimization
        'gradient_accumulation': 1,
        'max_grad_norm': 1.0,
        
        # Task
        'task': 'both',  # 'deblurring', 'deraining', or 'both'
        
        # Saving
        'save_interval': 10,
    }
    
    print("=" * 60)
    print("Latent-IRSDE Training Configuration")
    print("=" * 60)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 60)
    
    # Start training
    model = train_latent_irsde(config)
    
    return model


if __name__ == '__main__':
    main()
