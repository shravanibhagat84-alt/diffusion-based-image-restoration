import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block for CNN-based encoder/decoder"""
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return x + h


class DiffusionUNetCNN(nn.Module):
    """
    CNN-based UNet for image restoration.
    Supports arbitrary input resolutions including high-resolution (1920x1080).
    
    10X OPTIMIZED VERSION: Deeper network with more layers.
    """
    def __init__(self, in_channels=3, base_channels=64, num_layers=10):
        super().__init__()
        self.num_layers = num_layers
        
        # Encoder - 10X deeper with more blocks
        enc1_layers = [
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        ]

        for _ in range(num_layers):
            enc1_layers.append(ResBlock(base_channels))
        self.enc1 = nn.Sequential(*enc1_layers)
        
        # Encoder stage 2
        enc2_layers = [
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers):
            enc2_layers.append(ResBlock(base_channels * 2))
        self.enc2 = nn.Sequential(*enc2_layers)
        
        # Encoder stage 3
        enc3_layers = [
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers):
            enc3_layers.append(ResBlock(base_channels * 4))
        self.enc3 = nn.Sequential(*enc3_layers)
        
        # Middle - 10X deeper
        middle_layers = []
        for _ in range(num_layers * 2):
            middle_layers.append(ResBlock(base_channels * 4))
        self.middle = nn.Sequential(*middle_layers)
        
        # Decoder with skip connections - 10X deeper
        dec3_layers = [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers):
            dec3_layers.append(ResBlock(base_channels * 2))
        self.dec3 = nn.Sequential(*dec3_layers)
        
        dec2_layers = [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers):
            dec2_layers.append(ResBlock(base_channels))
        self.dec2 = nn.Sequential(*dec2_layers)
        
        dec1_layers = [
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers):
            dec1_layers.append(ResBlock(base_channels))
        dec1_layers.append(nn.Conv2d(base_channels, in_channels, 3, padding=1))
        self.dec1 = nn.Sequential(*dec1_layers)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Middle
        m = self.middle(e3)
        
        # Decoder with skip connections
        d3 = self.dec3(m)
        d2 = self.dec2(d3 + e2)  # Add skip connection
        d1 = self.dec1(d2 + e1)  # Add skip connection
        
        return d1

    def restore(self, x, steps=30):
        """
        Fast diffusion approximation (simplified).
        For faster execution, we reduce the number of iterations.
        """
        out = x
        # Use fewer steps for faster execution while maintaining quality
        for t in range(min(steps, 5)):
            out = self.forward(out)
        return out


# Keep the old Transformer-based class for compatibility
class SelfAttentionBlock(nn.Module):
    """Self-attention block for processing tokens"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLPBlock(nn.Module):
    """MLP block with GELU activation"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttentionBlock(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(dim, dim * 4)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DiffusionUNet(nn.Module):
    """
    MLP-Transformer based DiffusionUNet for image restoration.
    Uses attention and MLP layers instead of CNN.
    Now supports high-resolution processing (1920x1080) with arbitrary input sizes.
    
    10X OPTIMIZED VERSION: Supports deeper networks with num_layers parameter.
    Note: For best results with variable resolution images, use DiffusionUNetCNN instead.
    """
    def __init__(self, in_channels=3, base_channels=64, patch_size=8, max_resolution=(1920, 1080), use_cnn=True, num_layers=10):
        super().__init__()
        
        # Use CNN-based UNet by default for better high-res support - 10X deeper
        if use_cnn:
            self.unet = DiffusionUNetCNN(in_channels, base_channels, num_layers=num_layers)
        else:
            self.unet = None
            
        self.use_cnn = use_cnn
        self.patch_size = patch_size
        self.max_resolution = max_resolution
        
        if not use_cnn:
            # Original transformer-based implementation - 10X more blocks
            self.hidden_dim = base_channels * 4
            self.patch_embed = nn.Linear(in_channels * patch_size * patch_size, self.hidden_dim)
            self.register_buffer('pos_embed', torch.zeros(1, 1, self.hidden_dim))
            
            # 10X more transformer blocks
            self.encoder_blocks = nn.ModuleList([
                TransformerBlock(self.hidden_dim, num_heads=4)
                for _ in range(num_layers)
            ])
            
            self.decoder_blocks = nn.ModuleList([
                TransformerBlock(self.hidden_dim, num_heads=4)
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, in_channels * patch_size * patch_size)
            )
            
            self.skip_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            
            self.resolution_embed = nn.Sequential(
                nn.Linear(2, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
            
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def _get_pos_embed(self, num_patches):
        if self.pos_embed.shape[1] != num_patches:
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.hidden_dim))
            nn.init.trunc_normal_(pos_embed, std=0.02)
            return pos_embed
        return self.pos_embed
        
    def _forward_transformer(self, x):
        B, C, H, W = x.shape
        
        num_h = H // self.patch_size
        num_w = W // self.patch_size
        num_patches = num_h * num_w
        
        pos_embed = self._get_pos_embed(num_patches)
        
        patch_size = self.patch_size
        x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        B, C, num_h, num_w, _, _ = x.shape
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, num_h * num_w, C * patch_size * patch_size)
        
        x = self.patch_embed(x)
        x = x + pos_embed
        
        res_embed = self.resolution_embed(torch.tensor([[H, W]], dtype=torch.float32, device=x.device))
        x = x + res_embed.unsqueeze(1)
        
        skip_connections = []
        
        for block in self.encoder_blocks:
            x = block(x)
            skip_connections.append(x)
        
        for i, block in enumerate(self.decoder_blocks):
            skip_idx = len(skip_connections) - 1 - i
            if i < len(skip_connections):
                skip = skip_connections[skip_idx]
                x = x + self.skip_proj(skip) * 0.5
            x = block(x)
        
        x = self.output_proj(x)
        
        x = x.view(B, num_h, num_w, C, patch_size, patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, H, W)
        
        return x
        
    def forward(self, x):
        if self.use_cnn:
            return self.unet(x)
        else:
            return self._forward_transformer(x)

    def restore(self, x, steps=30):
        """Fast diffusion approximation."""
        out = x
        for t in range(min(steps, 5)):
            out = self.forward(out)
        return out
