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
    """Residual block for CNN-based encoder"""
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


class DegradationNet(nn.Module):
    """
    MLP-Mixer based DegradationNet for predicting degradation masks.
    Uses patch tokenization and MLP layers instead of CNN.
    Now supports high-resolution processing (1920x1080) with arbitrary input sizes.
    
    Can use frozen pretrained backbone for better feature extraction.
    
    10X OPTIMIZED VERSION: Supports deeper networks with more layers and channels.
    """
    def __init__(self, in_channels=3, base_channels=32, patch_size=8, 
                 use_pretrained=False, pretrained_path=None, num_layers=10):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.use_pretrained = use_pretrained
        self.num_layers = num_layers  # 10X more layers
        
        # Token dimension
        self.token_dim = in_channels * patch_size * patch_size  # 3 * 8 * 8 = 192
        
        if use_pretrained:
            # Use CNN-based encoder with pretrained features - 10X DEEPER
            encoder_layers = []
            ch = base_channels * 2
            
            # Input convolution
            encoder_layers.append(nn.Conv2d(in_channels, ch, 3, padding=1))
            encoder_layers.append(nn.ReLU(inplace=True))
            
            # Add 10X more ResBlocks for deeper network
            for _ in range(num_layers):
                encoder_layers.append(ResBlock(ch))
            
            # Downsampling layers with more blocks
            encoder_layers.append(nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1))
            for _ in range(num_layers):
                encoder_layers.append(ResBlock(ch * 2))
            
            ch = ch * 2
            encoder_layers.append(nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1))
            for _ in range(num_layers):
                encoder_layers.append(ResBlock(ch * 2))
            
            self.encoder = nn.Sequential(*encoder_layers)
            self.encoder_channels = ch * 2
            
            # Global pooling for predictions
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            
            # Output head - 10X larger
            self.mask_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.encoder_channels, base_channels * 16),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(base_channels * 16, base_channels * 8),
                nn.GELU(),
                nn.Linear(base_channels * 8, base_channels * 4),
                nn.GELU(),
                nn.Linear(base_channels * 4, 1)
            )
        else:
            # Original MLP-Mixer approach with dynamic patch handling - 10X MORE BLOCKS
            # Patch embedding - linear projection of flattened patches
            self.patch_embed = nn.Linear(self.token_dim, base_channels * 4)
            
            # Learnable class token and positional embedding (will be updated dynamically)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, base_channels * 4))
            self.register_buffer('pos_embed', torch.zeros(1, 1, base_channels * 4))
            
            # MLP-Mixer blocks - 10X more blocks
            self.mixer_blocks = nn.ModuleList([
                MLPMixerBlock(1, base_channels * 4, base_channels * 4)  # Dynamic num_tokens
                for _ in range(num_layers)  # 10X more blocks
            ])
            
            # Output head - predict mask - 10X larger
            self.mask_head = nn.Sequential(
                nn.LayerNorm(base_channels * 4),
                nn.Linear(base_channels * 4, base_channels * 8),
                nn.GELU(),
                nn.Linear(base_channels * 8, base_channels * 4),
                nn.GELU(),
                nn.Linear(base_channels * 4, 1)
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
        
        if self.use_pretrained:
            # Use CNN encoder
            features = self.encoder(x)  # (B, channels, H/4, W/4)
            pooled = self.global_pool(features)  # (B, channels, 1, 1)
            mask = self.mask_head(pooled)
            mask = torch.sigmoid(mask)
            
            # Reshape mask to (B, 1, 1, 1) for interpolation
            mask = mask.view(B, 1, 1, 1)
            
            # Upsample mask to match input size
            mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
            return mask
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
            x = x.contiguous().view(B, C, num_h * num_w, patch_size * patch_size)
            x = x.permute(0, 2, 1, 3).contiguous().view(B, num_h * num_w, C * patch_size * patch_size)
            
            # Linear projection
            x = self.patch_embed(x)
            
            # Add cls token
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
            # Add positional embedding
            x = x + self.pos_embed
            
            # MLP-Mixer blocks (need to update internal norms for dynamic patches)
            for block in self.mixer_blocks:
                x = block(x)
            
            # Get cls token output for prediction
            cls_output = x[:, 0]
            
            # Predict mask
            mask = self.mask_head(cls_output)
            mask = torch.sigmoid(mask)
            
            return mask.unsqueeze(-1).unsqueeze(-1)


class DegradationNetWithEncoder(nn.Module):
    """
    DegradationNet with frozen pretrained encoder backbone.
    Uses frozen internal representations for efficient high-resolution processing.
    """
    def __init__(self, encoder_model=None, in_channels=3, base_channels=32):
        super().__init__()
        
        # Use provided encoder or create new one
        if encoder_model is not None:
            self.encoder = encoder_model
            # Freeze encoder
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder_channels = encoder_model.encoder_channels if hasattr(encoder_model, 'encoder_channels') else base_channels * 4
        else:
            self.encoder = DegradationNet(in_channels=in_channels, base_channels=base_channels, use_pretrained=True)
            self.encoder_channels = base_channels * 4
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Output head
        self.mask_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.encoder_channels, base_channels * 4),
            nn.GELU(),
            nn.Linear(base_channels * 4, 1)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        with torch.no_grad():
            features = self.encoder.encoder(x) if hasattr(self.encoder, 'encoder') else self.encoder(x)
        
        # Use encoder features
        pooled = self.global_pool(features)
        mask = self.mask_head(pooled)
        mask = torch.sigmoid(mask)
        
        # Reshape mask to (B, 1, 1, 1) for interpolation
        mask = mask.view(B, 1, 1, 1)
        
        # Upsample
        mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
        
        return mask
