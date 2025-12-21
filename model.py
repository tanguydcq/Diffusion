import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Block(nn.Module):
    """
    Bloc de base du réseau : Conv -> Norm -> Activation.
    Il intègre aussi l'information temporelle (t) via une addition.
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        
        # Si on est dans la descente (Encoder), on garde la taille ou on réduit
        # Si on est dans la montée (Decoder), on a besoin de transformations spécifiques
        # Ici, pour simplifier sur du 16x16, je fais juste des Convolutions qui gardent la taille (padding=1)
        # Le changement de taille (Upsample/Downsample) sera fait manuellement dans le forward
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1) if not up else nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        
        self.norm1 = nn.GroupNorm(8, out_ch) # GroupNorm est plus stable que BatchNorm pour la diffusion
        self.act1 = nn.SiLU() # SiLU (Swish) est l'activation standard pour les modèles de diffusion
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU()

    def forward(self, x, t):
        # 1. Première convolution
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        
        # 2. Injection du Temps : C'est CRUCIAL.
        # Le réseau doit savoir s'il cherche un tout petit bruit (t=1) ou un gros bruit (t=T).
        # On projette le vecteur temps pour qu'il matche les canaux de l'image.
        time_emb = self.time_mlp(t) # (Batch, out_ch)
        # On étend les dimensions pour additionner : (Batch, out_ch, 1, 1)
        h = h + time_emb[(..., ) + (None, ) * 2]
        
        # 3. Deuxième convolution
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        return h

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 1 # MNIST est en noir et blanc
        down_channels = (32, 64, 128) # Nombre de filtres à chaque étage
        up_channels = (128, 64, 32)
        out_dim = 1 
        time_emb_dim = 32

        # -- Encodage du temps --
        # Projection simple d'un entier t vers un vecteur dense
        self.time_mlp = nn.Sequential(
            # SinusoidalPositionEmbeddings serait mieux théoriquement, 
            # mais pour MNIST 16x16, un Embedding apprenable suffit largement et simplifie le code.
            nn.Embedding(1000, time_emb_dim), # Suppose T=1000 max
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        # -- Encoder (Descente) --
        # Entrée : (Batch, 1, 16, 16) -> Sortie : (Batch, 32, 16, 16)
        self.conv_in = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
        
        # Block 1 : (32 -> 64), passe de 16x16 à 8x8
        self.down1 = Block(down_channels[0], down_channels[1], time_emb_dim)
        self.pool1 = nn.MaxPool2d(2) 
        
        # Block 2 : (64 -> 128), passe de 8x8 à 4x4
        self.down2 = Block(down_channels[1], down_channels[2], time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)

        # -- Bottleneck (Le fond du U) --
        # Espace latent de dimension (128, 4, 4)
        self.bot1 = Block(down_channels[2], down_channels[2], time_emb_dim)

        # -- Decoder (Montée) --
        # Block Up 1 : Prend 128 + 64 (skip connection) canaux -> sort 64 canaux. 4x4 -> 8x8
        self.up1 = nn.ConvTranspose2d(down_channels[2], down_channels[1], 2, 2) # Upsample dur
        self.up_block1 = Block(down_channels[1] + down_channels[1], down_channels[1], time_emb_dim)
        
        # Block Up 2 : Prend 64 + 32 (skip connection) canaux -> sort 32 canaux. 8x8 -> 16x16
        self.up2 = nn.ConvTranspose2d(down_channels[1], down_channels[0], 2, 2)
        self.up_block2 = Block(down_channels[0] + down_channels[0], down_channels[0], time_emb_dim)
        
        # -- Sortie --
        # Projection finale vers 1 canal (le bruit prédit)
        # Pas d'activation (le bruit peut être positif ou négatif)
        self.out = nn.Conv2d(down_channels[0], out_dim, 1)

    def forward(self, x, t):
        # x: (Batch, 1, 16, 16)  <- Image bruitée
        # t: (Batch)            <- Entiers représentant le step (ex: [42, 900, 12...])

        # 1. Traitement du temps
        t = self.time_mlp(t) # Devient un vecteur (Batch, 32)
        
        # 2. Conv initiale
        x1 = self.conv_in(x) # (Batch, 32, 16, 16)
        
        # 3. Downsampling (Encoder)
        # On garde x1 et x2 en mémoire pour les "Skip Connections" (les flèches grises horizontales sur les schémas U-Net)
        x2 = self.down1(x1, t) # Process avec temps
        x2_small = self.pool1(x2) # (Batch, 64, 8, 8)
        
        x3 = self.down2(x2_small, t)
        x3_small = self.pool2(x3) # (Batch, 128, 4, 4)
        
        # 4. Bottleneck
        x_bot = self.bot1(x3_small, t) # (Batch, 128, 4, 4)
        
        # 5. Upsampling (Decoder) + Skip Connections
        # On remonte et on colle (concat) l'info spatiale précise qu'on avait capturée à la descente
        
        # Remontée 1
        x_up1 = self.up1(x_bot) # Revient à 8x8
        # Concaténation : on colle x2 (qui vient d'en haut) avec x_up1 (qui vient d'en bas)
        # C'est ça qui permet au réseau de générer des détails nets
        x_cat1 = torch.cat([x_up1, x2], dim=1) 
        x_dec1 = self.up_block1(x_cat1, t)
        
        # Remontée 2
        x_up2 = self.up2(x_dec1) # Revient à 16x16
        x_cat2 = torch.cat([x_up2, x1], dim=1)
        x_dec2 = self.up_block2(x_cat2, t)
        
        # 6. Prédiction finale
        output = self.out(x_dec2) # (Batch, 1, 16, 16)
        
        return output # Ceci est xi_theta (l'estimation du bruit)