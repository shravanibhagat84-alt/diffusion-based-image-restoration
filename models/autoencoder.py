import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResBlock(nn.Module):
    """Residual block with GroupNorm and SiLU activation"""
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


class AttentionBlock(nn.Module):
    """Spatial attention block for feature refinement"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        # Normalize
        x_norm = self.norm(x)
        
        # QKV projection
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = rearrange(q, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
        k = rearrange(k, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
        
        # Attention
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=H, y=W)
        
        # Project and add residual
        out = self.proj(out)
        return x + out


class Downsample(nn.Module):
    """Downsampling layer"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class Autoencoder(nn.Module):
    """
    Pretrained Autoencoder for latent space diffusion.
    Encodes high-resolution images into compact latent representations
    and decodes them back to image space.
    
    This is the core component of Latent-IRSDE that enables
    efficient high-resolution image restoration.
    """
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, latent_channels=4,
                 resolution=256, use_attention=True, num_res_blocks=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.latent_channels = latent_channels
        
        # Encoder
        self.encoder_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        for i in range(3):  # 3 downsampling layers (256->128->64->32)
            self.down_blocks.append(nn.ModuleList([
                ResBlock(ch, dropout=0.0) for _ in range(num_res_blocks)
            ]))
            if use_attention:
                self.down_blocks[-1].append(AttentionBlock(ch))
            self.down_blocks.append(Downsample(ch))
            ch = ch * 2
        
        # Middle
        self.mid_block = nn.ModuleList([
            ResBlock(ch),
            AttentionBlock(ch) if use_attention else nn.Identity(),
            ResBlock(ch)
        ])
        
        # Latent space projection
        self.to_latent = nn.Conv2d(ch, latent_channels, 3, padding=1)
        self.from_latent = nn.Conv2d(latent_channels, ch, 3, padding=1)
        
        # Decoder
        # Upsampling blocks (reverse of encoder)
        self.up_blocks = nn.ModuleList()
        for i in range(3):  # 3 upsampling layers (32->64->128->256)
            self.up_blocks.append(Upsample(ch))
            ch = ch // 2
            self.up_blocks.append(nn.ModuleList([
                ResBlock(ch, dropout=0.0) for _ in range(num_res_blocks + 1)
            ]))
            if use_attention and i < 2:  # Add attention to first two up blocks
                self.up_blocks[-1].append(AttentionBlock(ch))
        
        # Output
        self.encoder_out = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        """Encode image to latent space"""
        # Initial convolution
        h = self.encoder_in(x)
        
        # Downsampling
        for block in self.down_blocks:
            if isinstance(block, nn.ModuleList):
                for res_block in block:
                    if isinstance(res_block, AttentionBlock):
                        h = res_block(h)
                    else:
                        h = res_block(h)
            else:
                h = block(h)
        
        # Middle
        for block in self.mid_block:
            h = block(h)
        
        # To latent
        h = self.to_latent(h)
        return h
    
    def decode(self, z):
        """Decode from latent space to image"""
        # From latent
        h = self.from_latent(z)
        
        # Upsampling
        for block in self.up_blocks:
            if isinstance(block, Upsample):
                h = block(h)
            elif isinstance(block, nn.ModuleList):
                for res_block in block:
                    if isinstance(res_block, AttentionBlock):
                        h = res_block(h)
                    else:
                        h = res_block(h)
        
        # Output
        h = self.encoder_out(h)
        return h
    
    def forward(self, x):
        """Full encode-decode pass"""
        z = self.encode(x)
        recon = self.decode(z)
        return recon
    
    def get_latent(self, x):
        """Get latent representation without decoding"""
        return self.encode(x)


class FrozenAutoencoder(nn.Module):
    """
    Frozen pretrained autoencoder for Latent-IRSDE.
    Uses frozen internal representations to perform efficient
    high-resolution aerial image restoration.
    """
    def __init__(self, autoencoder, freeze_encoder=True, freeze_decoder=True):
        super().__init__()
        self.autoencoder = autoencoder
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in autoencoder.encoder_in.parameters():
                param.requires_grad = False
            for param in autoencoder.down_blocks.parameters():
                param.requires_grad = False
            for param in autoencoder.mid_block.parameters():
                param.requires_grad = False
            for param in autoencoder.to_latent.parameters():
                param.requires_grad = False
        
        # Freeze decoder if requested
        if freeze_decoder:
            for param in autoencoder.from_latent.parameters():
                param.requires_grad = False
            for param in autoencoder.up_blocks.parameters():
                param.requires_grad = False
            for param in autoencoder.encoder_out.parameters():
                param.requires_grad = False
    
    def encode(self, x):
        """Encode with frozen encoder"""
        with torch.no_grad():
            return self.autoencoder.encode(x)
    
    def decode(self, z):
        """Decode with frozen or trainable decoder"""
        return self.autoencoder.decode(z)
    
    def forward(self, x):
        """Full pass"""
        z = self.encode(x)
        recon = self.decode(z)
        return recon
    
    def get_latent(self, x):
        """Get latent representation"""
        return self.encode(x)


def create_pretrained_autoencoder(device='cuda', **kwargs):
    """
    Create a pretrained autoencoder model.
    In practice, you would load pretrained weights here.
    For now, we create a new model that can be trained.
    """
    model = Autoencoder(**kwargs)
    return model.to(device)


# Convenience function to create Latent-IRSDE compatible autoencoder
def create_latent_irsde_autoencoder(image_size=256, latent_channels=4):
    """
    Create autoencoder optimized for Latent-IRSDE.
    Uses smaller latent dimension for efficiency.
    """
    return Autoencoder(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        latent_channels=latent_channels,
        resolution=image_size,
        use_attention=True,
        num_res_blocks=2
    )
