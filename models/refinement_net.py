import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPMixerBlock(nn.Module):
    """MLP-Mixer style block with token mixing and channel mixing"""
    def __init__(self, num_tokens, token_dim, channel_dim):
        super().__init__()
        # Token mixing - operates across tokens (dimension N)
        self.token_norm = nn.LayerNorm(num_tokens)
        self.token_mixer = nn.Sequential(
            nn.Linear(num_tokens, num_tokens * 4),
            nn.GELU(),
            nn.Linear(num_tokens * 4, num_tokens)
        )
        # Channel mixing - operates across channels (dimension C)
        self.channel_norm = nn.LayerNorm(channel_dim)
        self.channel_mixer = nn.Sequential(
            nn.Linear(channel_dim, channel_dim * 4),
            nn.GELU(),
            nn.Linear(channel_dim * 4, channel_dim)
        )
        
    def forward(self, x):
        # x shape: (B, N, C) where N is num_tokens, C is channel_dim
        # Token mixing (residual)
        residual = x
        # Transpose to (B, C, N) for token mixing, apply norm on N
        x = self.token_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.token_mixer(x.transpose(1, 2)).transpose(1, 2) + residual
        # Channel mixing (residual)
        residual = x
        x = self.channel_norm(x)
        x = self.channel_mixer(x) + residual
        return x


class ResBlock(nn.Module):
    """Residual block for CNN-based refinement"""
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


class RefineNet(nn.Module):
    """
    MLP-Mixer based RefineNet for image refinement.
    Uses patch tokenization and MLP layers instead of CNN.
    Now supports high-resolution processing (1920x1080) with arbitrary input sizes.
    
    Can also use a CNN-based architecture for better high-resolution handling.
    """
    def __init__(self, in_channels=3, base_channels=32, patch_size=8,
                 use_cnn=False):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.use_cnn = use_cnn
        
        if use_cnn:
            # CNN-based refinement network (better for high-res)
            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                ResBlock(base_channels),
                nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
                ResBlock(base_channels * 2),
                nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1),
                ResBlock(base_channels * 2),
            )
            
            # Decoder with upsampling
            self.decoder = nn.Sequential(
                ResBlock(base_channels * 2),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ResBlock(base_channels * 2),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ResBlock(base_channels),
                nn.Conv2d(base_channels, base_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, in_channels, 3, padding=1)
            )
        else:
            # Original MLP-Mixer approach
            # Token dimension
            self.token_dim = in_channels * patch_size * patch_size  # 3 * 8 * 8 = 192
            
            # Patch embedding - linear projection of flattened patches
            self.patch_embed = nn.Linear(self.token_dim, base_channels * 4)
            
            # Learnable class token and positional embedding (will be updated dynamically)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, base_channels * 4))
            self.register_buffer('pos_embed', torch.zeros(1, 1, base_channels * 4))
            
            # MLP-Mixer blocks
            self.mixer_blocks = nn.ModuleList([
                MLPMixerBlock(1, base_channels * 4, base_channels * 4)  # Dynamic num_tokens
                for _ in range(6)
            ])
            
            # Output head - reconstruct image
            self.output_head = nn.Sequential(
                nn.LayerNorm(base_channels * 4),
                nn.Linear(base_channels * 4, base_channels * 4),
                nn.GELU(),
                nn.Linear(base_channels * 4, self.token_dim)
            )
            
            # Resolution-aware embedding
            self.resolution_embed = nn.Sequential(
                nn.Linear(2, base_channels * 4),
                nn.GELU(),
                nn.Linear(base_channels * 4, base_channels * 4)
            )
            
            # Initialize
            nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def _update_pos_embed(self, num_patches):
        """Update positional embedding based on number of patches"""
        if self.pos_embed.shape[1] != num_patches + 1:
            pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.cls_token.shape[-1]))
            nn.init.trunc_normal_(pos_embed, std=0.02)
            self.pos_embed = pos_embed
        return self.pos_embed
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        if self.use_cnn:
            # Use CNN-based refinement
            features = self.encoder(x)
            output = self.decoder(features)
            # Ensure output size matches input
            output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
            return output
        else:
            # Use MLP-Mixer with dynamic patch handling
            patch_size = self.patch_size
            num_h = H // patch_size
            num_w = W // patch_size
            num_patches = num_h * num_w
            
            # Update positional embedding
            self._update_pos_embed(num_patches)
            
            # Patchify: (B, C, H, W) -> (B, num_patches, token_dim)
            x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            B, C, num_h, num_w, _, _ = x.shape
            x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
            x = x.view(B, num_h * num_w, C * patch_size * patch_size)
            
            # Linear projection
            x = self.patch_embed(x)
            
            # Add cls token
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
            # Add positional embedding
            x = x + self.pos_embed
            
            # Add resolution embedding for high-res support
            res_embed = self.resolution_embed(torch.tensor([[H, W]], dtype=torch.float32, device=x.device))
            x = x + res_embed.unsqueeze(1)
            
            # MLP-Mixer blocks
            for block in self.mixer_blocks:
                x = block(x)
            
            # Remove cls token and get patch tokens
            patch_tokens = x[:, 1:]
            
            # Output projection
            x = self.output_head(patch_tokens)
            
            # Reshape back to image: (B, num_patches, token_dim) -> (B, C, H, W)
            x = x.view(B, num_h, num_w, C, patch_size, patch_size)
            x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
            x = x.view(B, C, H, W)
            
            return x


class RefineNetWithSkipConnections(nn.Module):
    """
    RefineNet with skip connections for better detail preservation.
    
    10X OPTIMIZED VERSION: Supports deeper networks with num_layers parameter.
    """
    def __init__(self, in_channels=3, base_channels=64, num_layers=10):
        super().__init__()
        self.num_layers = num_layers
        
        # Encoder with intermediate features for skip connections - 10X deeper
        enc1_layers = [
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers):
            enc1_layers.append(ResBlock(base_channels))
        self.enc1 = nn.Sequential(*enc1_layers)
        
        enc2_layers = [
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers):
            enc2_layers.append(ResBlock(base_channels * 2))
        self.enc2 = nn.Sequential(*enc2_layers)
        
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
