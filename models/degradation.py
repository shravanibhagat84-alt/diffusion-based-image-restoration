"""
Degradation models for realistic image restoration training.
Implements motion blur, rain streaks, and raindrops for UAV aerial images.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage


class BlurKernel:
    """Generate realistic motion blur kernels"""
    
    @staticmethod
    def generate_linear_kernel(size=15, angle=None, sigma=3.0):
        """
        Generate a linear motion blur kernel.
        
        Args:
            size: Kernel size
            angle: Blur angle in degrees (random if None)
            sigma: Kernel spread
        """
        if angle is None:
            angle = np.random.uniform(0, 180)
        
        # Create coordinate grid
        center = size // 2
        y, x = np.ogrid[:size, :size]
        x = x - center
        y = y - center
        
        # Rotate by angle
        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Distance from line
        dist = np.abs(x * sin_a - y * cos_a)
        
        # Gaussian-like kernel along the line
        kernel = np.exp(-dist ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        
        return kernel.astype(np.float32)
    
    @staticmethod
    def generate_disk_kernel(size=15, radius=None):
        """Generate a disk/average blur kernel"""
        if radius is None:
            radius = size // 4
        
        y, x = np.ogrid[:size, :size]
        center = size // 2
        dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        
        kernel = (dist <= radius).astype(np.float32)
        kernel = kernel / kernel.sum()
        
        return kernel
    
    @staticmethod
    def generate_gaussian_kernel(size=15, sigma=3.0):
        """Generate a Gaussian blur kernel"""
        y, x = np.ogrid[:size, :size]
        center = size // 2
        
        kernel = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        
        return kernel.astype(np.float32)


class RainSimulation:
    """Simulate rain effects for training"""
    
    @staticmethod
    def add_rain_streaks(image, intensity=0.3, num_streaks=100):
        """
        Add rain streaks to an image.
        
        Args:
            image: Input image tensor (B, C, H, W) in range [0, 1]
            intensity: Rain intensity
            num_streaks: Number of rain streaks
        """
        B, C, H, W = image.shape
        output = image.clone()
        
        for b in range(B):
            # Random angle for this batch
            angle = np.random.uniform(-15, 15)  # Slight angle for rain
            
            for _ in range(num_streaks):
                # Random position
                y = np.random.randint(0, H)
                x = np.random.randint(0, W)
                
                # Random length and thickness
                length = np.random.randint(10, 30)
                thickness = np.random.randint(1, 3)
                
                # Calculate endpoint
                dx = length * np.sin(np.deg2rad(angle))
                dy = length * np.cos(np.deg2rad(angle))
                
                # Draw line (simplified)
                for i in range(length):
                    ny = int(y + dy * i / length)
                    nx = int(x + dx * i / length)
                    
                    if 0 <= ny < H and 0 <= nx < W:
                        # Add bright line
                        for c in range(C):
                            output[b, c, ny, nx] = torch.clamp(
                                output[b, c, ny, nx] + intensity * np.random.uniform(0.5, 1.0),
                                0, 1
                            )
        
        return output
    
    @staticmethod
    def add_rain_droplets(image, intensity=0.2, num_droplets=50):
        """
        Add rain droplets to an image.
        
        Args:
            image: Input image tensor (B, C, H, W) in range [0, 1]
            intensity: Droplet intensity
            num_droplets: Number of droplets
        """
        B, C, H, W = image.shape
        output = image.clone()
        
        for b in range(B):
            for _ in range(num_droplets):
                # Random position
                y = np.random.randint(5, H - 5)
                x = np.random.randint(5, W - 5)
                
                # Random size
                size = np.random.randint(2, 6)
                
                # Create droplet (bright spot with soft edges)
                for dy in range(-size, size + 1):
                    for dx in range(-size, size + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            dist = np.sqrt(dx ** 2 + dy ** 2)
                            if dist <= size:
                                brightness = intensity * (1 - dist / size) * np.random.uniform(0.5, 1.0)
                                for c in range(C):
                                    output[b, c, ny, nx] = torch.clamp(
                                        output[b, c, ny, nx] + brightness,
                                        0, 1
                                    )
        
        return output
    
    @staticmethod
    def add_rain_fog(image, intensity=0.15):
        """
        Add light rain fog/haze effect.
        
        Args:
            image: Input image tensor
            intensity: Fog intensity
        """
        B, C, H, W = image.shape
        
        # Add uniform haze
        fog = torch.rand(B, C, H, W, device=image.device) * intensity
        
        # Blend with original
        output = image * (1 - intensity) + fog * intensity
        
        # Slight desaturation
        gray = output.mean(dim=1, keepdim=True)
        output = gray * 0.2 + output * 0.8
        
        return output.clamp(0, 1)


class DegradationModel(nn.Module):
    """
    Complete degradation model for UAV aerial image restoration.
    Combines blur, rain, and noise for realistic training data.
    """
    def __init__(self, 
                 blur_prob=0.5,
                 rain_prob=0.3,
                 noise_prob=0.2,
                 high_res=True):
        super().__init__()
        
        self.blur_prob = blur_prob
        self.rain_prob = rain_prob
        self.noise_prob = noise_prob
        self.high_res = high_res
        
        # Kernel generator
        self.kernel_gen = BlurKernel()
        
    def apply_blur(self, image):
        """Apply motion blur to image"""
        B, C, H, W = image.shape
        device = image.device
        
        # Random kernel type
        kernel_type = np.random.choice(['linear', 'disk', 'gaussian'])
        
        if kernel_type == 'linear':
            size = np.random.choice([7, 9, 11, 15, 21])
            angle = np.random.uniform(0, 180)
            sigma = np.random.uniform(1.5, 4.0)
            kernel = self.kernel_gen.generate_linear_kernel(size, angle, sigma)
        elif kernel_type == 'disk':
            size = np.random.choice([7, 9, 11, 15])
            kernel = self.kernel_gen.generate_disk_kernel(size)
        else:
            size = np.random.choice([7, 9, 11, 15])
            sigma = np.random.uniform(1.0, 3.0)
            kernel = self.kernel_gen.generate_gaussian_kernel(size, sigma)
        
        # Convert to tensor
        kernel = torch.from_numpy(kernel).to(device)
        
        # Apply blur using conv2d
        # Need to reshape for convolution
        kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
        kernel = kernel.repeat(C, 1, 1, 1)
        
        # Pad image
        pad = kernel.shape[-1] // 2
        image_padded = F.pad(image, [pad, pad, pad, pad], mode='replicate')
        
        # Apply convolution
        blurred = F.conv2d(image_padded.view(1, B * C, H + 2 * pad, W + 2 * pad), 
                          kernel.view(B * C, 1, kernel.shape[2], kernel.shape[3]),
                          groups=B * C)
        blurred = blurred.view(B, C, H, W)
        
        return blurred
    
    def apply_rain(self, image):
        """Apply rain effects to image"""
        rain_type = np.random.choice(['streaks', 'droplets', 'fog', 'mixed'])
        
        if rain_type == 'streaks':
            image = RainSimulation.add_rain_streaks(image, intensity=0.2)
        elif rain_type == 'droplets':
            image = RainSimulation.add_rain_droplets(image, intensity=0.15)
        elif rain_type == 'fog':
            image = RainSimulation.add_rain_fog(image, intensity=0.1)
        else:  # mixed
            if np.random.random() > 0.5:
                image = RainSimulation.add_rain_streaks(image, intensity=0.15)
            if np.random.random() > 0.5:
                image = RainSimulation.add_rain_droplets(image, intensity=0.1)
        
        return image
    
    def apply_noise(self, image):
        """Apply sensor noise to image"""
        # Gaussian noise
        if np.random.random() > 0.5:
            sigma = np.random.uniform(0.01, 0.03)
            noise = torch.randn_like(image) * sigma
            image = image + noise
        
        # Salt and pepper noise
        if np.random.random() > 0.7:
            prob = np.random.uniform(0.01, 0.03)
            mask = torch.rand_like(image)
            image = torch.where(mask < prob, torch.zeros_like(image), 
                              torch.where(mask > 1 - prob, torch.ones_like(image), image))
        
        return image.clamp(0, 1)
    
    def forward(self, clean_image):
        """
        Apply random degradation to clean image.
        
        Args:
            clean_image: Clean image tensor (B, C, H, W) in range [0, 1]
            
        Returns:
            Degraded image tensor
        """
        degraded = clean_image.clone()
        
        # Apply blur
        if np.random.random() < self.blur_prob:
            degraded = self.apply_blur(degraded)
        
        # Apply rain
        if np.random.random() < self.rain_prob:
            degraded = self.apply_rain(degraded)
        
        # Apply noise
        if np.random.random() < self.noise_prob:
            degraded = self.apply_noise(degraded)
        
        return degraded.clamp(0, 1)


class DeblurSpecificModel(nn.Module):
    """
    Deblurring-specific degradation model.
    Focuses on various blur types for deblurring training.
    """
    def __init__(self):
        super().__init__()
        self.kernel_gen = BlurKernel()
        
    def forward(self, image):
        """Apply blur degradation for deblurring training"""
        B, C, H, W = image.shape
        device = image.device
        
        # Various blur types
        blur_types = ['linear', 'disk', 'gaussian', 'motion']
        blur_type = np.random.choice(blur_types)
        
        if blur_type == 'linear':
            size = np.random.choice([9, 11, 15, 21])
            angle = np.random.uniform(0, 180)
            sigma = np.random.uniform(2.0, 5.0)
            kernel = self.kernel_gen.generate_linear_kernel(size, angle, sigma)
        elif blur_type == 'disk':
            size = np.random.choice([9, 11, 15])
            kernel = self.kernel_gen.generate_disk_kernel(size)
        elif blur_type == 'gaussian':
            size = np.random.choice([9, 11, 15])
            sigma = np.random.uniform(2.0, 5.0)
            kernel = self.kernel_gen.generate_gaussian_kernel(size, sigma)
        else:  # motion - multiple linear kernels
            kernel = self.kernel_gen.generate_linear_kernel(21, np.random.uniform(0, 180), 3.0)
        
        # Convert to tensor
        kernel = torch.from_numpy(kernel).to(device)
        
        # Apply
        kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
        kernel = kernel.repeat(C, 1, 1, 1)
        
        pad = kernel.shape[-1] // 2
        image_padded = F.pad(image, [pad, pad, pad, pad], mode='replicate')
        
        blurred = F.conv2d(image_padded.view(1, B * C, H + 2 * pad, W + 2 * pad), 
                          kernel.view(B * C, 1, kernel.shape[2], kernel.shape[3]),
                          groups=B * C)
        blurred = blurred.view(B, C, H, W)
        
        # Add slight noise
        noise = torch.randn_like(blurred) * 0.01
        blurred = blurred + noise
        
        return blurred.clamp(0, 1)


class DerainSpecificModel(nn.Module):
    """
    Deraining-specific degradation model.
    Focuses on rain effects for deraining training.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, image):
        """Apply rain degradation for deraining training"""
        # Random rain type
        rain_type = np.random.choice(['streaks', 'droplets', 'fog', 'heavy'])
        
        if rain_type == 'streaks':
            degraded = RainSimulation.add_rain_streaks(image, intensity=0.25)
        elif rain_type == 'droplets':
            degraded = RainSimulation.add_rain_droplets(image, intensity=0.2)
        elif rain_type == 'fog':
            degraded = RainSimulation.add_rain_fog(image, intensity=0.15)
        else:  # heavy - multiple effects
            degraded = image
            degraded = RainSimulation.add_rain_streaks(degraded, intensity=0.2)
            degraded = RainSimulation.add_rain_droplets(degraded, intensity=0.15)
            degraded = RainSimulation.add_rain_fog(degraded, intensity=0.1)
        
        # Add sensor noise
        noise = torch.randn_like(degraded) * 0.015
        degraded = degraded + noise
        
        return degraded.clamp(0, 1)
