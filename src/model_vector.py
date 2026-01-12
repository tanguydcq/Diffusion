import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Multi-Attribute Conditioning Module ---

class MultiAttributeEmbedding(nn.Module):
    """
    Embedding module for multi-attribute conditioning (e.g., CryptoPunks).
    Handles both categorical (type) and multi-hot (accessories) attributes.
    
    Args:
        num_types: Number of type categories (e.g., 5 for Male/Female/Zombie/Ape/Alien)
        num_accessories: Number of possible accessories (e.g., 87)
        embed_dim: Output embedding dimension
    """
    def __init__(self, num_types, num_accessories, embed_dim):
        super().__init__()
        self.num_types = num_types
        self.num_accessories = num_accessories
        self.embed_dim = embed_dim
        
        # Type embedding (categorical)
        self.type_embedding = nn.Embedding(num_types, embed_dim // 2)
        
        # Accessories embedding (multi-hot -> embedding via linear projection)
        self.accessory_projection = nn.Linear(num_accessories, embed_dim // 2)
        
        # Final MLP to combine
        self.combine_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, type_idx, accessory_vector):
        """
        Args:
            type_idx: (B,) tensor of type indices
            accessory_vector: (B, num_accessories) multi-hot tensor
        Returns:
            (B, embed_dim) conditioning embedding
        """
        # Type embedding
        type_emb = self.type_embedding(type_idx)  # (B, embed_dim // 2)
        
        # Accessory embedding
        acc_emb = self.accessory_projection(accessory_vector.float())  # (B, embed_dim // 2)
        
        # Combine
        combined = torch.cat([type_emb, acc_emb], dim=-1)  # (B, embed_dim)
        return self.combine_mlp(combined)


# --- Complex UNet for CryptoPunks ---

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class UNet(nn.Module):
    """
    UNet for diffusion models with optional multi-attribute conditioning and concept vector injection.
    
    This is identical to model.py's UNet but with the added capability to inject concept vectors
    into the latent space (h-space) for interpretable text-to-image generation.
    
    Args:
        c_in: Input image channels
        c_out: Output image channels
        time_dim: Time embedding dimension
        device: Device to use
        num_types: Number of type categories for conditioning (None = unconditional)
        num_accessories: Number of accessory categories for conditioning (None = unconditional)
    """
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda", 
                 num_types=None, num_accessories=None):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.num_types = num_types
        self.num_accessories = num_accessories
        
        # Multi-attribute conditioning (optional)
        if num_types is not None and num_accessories is not None:
            self.attr_embedding = MultiAttributeEmbedding(
                num_types=num_types,
                num_accessories=num_accessories,
                embed_dim=time_dim
            )
        else:
            self.attr_embedding = None
        
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128, emb_dim=time_dim)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256, emb_dim=time_dim)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256, emb_dim=time_dim)
        self.sa3 = SelfAttention(256)

        # Bottleneck (h-space) - where concept vectors are injected
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128, emb_dim=time_dim)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64, emb_dim=time_dim)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64, emb_dim=time_dim)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, type_idx=None, accessory_vector=None, concept_vector=None):
        """
        Forward pass with optional conditioning and concept vector injection.
        
        Args:
            x: (B, C, H, W) input image
            t: (B,) timesteps
            type_idx: (B,) type indices (optional, for conditional generation)
            accessory_vector: (B, num_accessories) multi-hot vector (optional)
            concept_vector: (B, 512, H_latent, W_latent) vector to inject in h-space (optional)
                          For img_size=32: shape is (B, 512, 4, 4)
                          This enables interpretable text-to-image generation
        """
        t = t.unsqueeze(-1).type(torch.float)
        t_emb = self.pos_encoding(t, self.time_dim)
        
        # Add attribute conditioning if available
        if self.attr_embedding is not None and type_idx is not None and accessory_vector is not None:
            attr_emb = self.attr_embedding(type_idx, accessory_vector)
            t_emb = t_emb + attr_emb  # Add conditioning to time embedding
        elif self.attr_embedding is not None:
            # Conditional model but no labels provided (unconditional generation)
            # Use zero embedding
            batch_size = x.shape[0]
            zero_type = torch.zeros(batch_size, dtype=torch.long, device=x.device)
            zero_acc = torch.zeros(batch_size, self.num_accessories, device=x.device)
            # Don't add anything - generate unconditionally
            pass

        x1 = self.inc(x)
        x2 = self.down1(x1, t_emb)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t_emb)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t_emb)
        x4 = self.sa3(x4)

        # Bottleneck with concept vector injection (h-space manipulation)
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        
        # CONCEPT VECTOR INJECTION: Linear addition in latent space
        # This is where text-to-image magic happens for interpretable generation
        if concept_vector is not None:
            x4 = x4 + concept_vector
        
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t_emb)
        x = self.sa4(x)
        x = self.up2(x, x2, t_emb)
        x = self.sa5(x)
        x = self.up3(x, x1, t_emb)
        x = self.sa6(x)
        output = self.outc(x)
        return output

def get_model(args):
    # CryptoPunks and other datasets
    num_types = getattr(args, 'num_types', None)
    num_accessories = getattr(args, 'num_accessories', None)
    
    return UNet(
        c_in=args.image_channels, 
        c_out=args.image_channels, 
        time_dim=args.time_emb_dim, 
        device=args.device,
        num_types=num_types,
        num_accessories=num_accessories
    )
