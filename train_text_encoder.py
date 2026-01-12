import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Imports internes
from src.model_vector import get_model
from src.diffusion import Diffusion
from src.config import get_config_by_name


class TextEncoder(nn.Module):
    """
    Encodeur simple qui transforme des embeddings de mots en vecteurs de concept.
    Architecture: Embedding -> Transformer -> Projection vers espace latent (512, 4, 4)
    """
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, latent_channels=512, latent_size=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Transformer encoder pour capturer les relations entre mots
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=8, 
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Projection vers l'espace latent du UNet
        self.to_latent = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_channels * latent_size * latent_size),
        )
        
        self.latent_channels = latent_channels
        self.latent_size = latent_size
    
    def forward(self, token_ids, attention_mask=None):
        """
        Args:
            token_ids: (batch, seq_len) indices de tokens
            attention_mask: (batch, seq_len) masque d'attention (1=valide, 0=padding)
        Returns:
            concept_vector: (batch, 512, 4, 4) vecteur à injecter dans le UNet
        """
        # Embedding des tokens
        x = self.embedding(token_ids)  # (batch, seq_len, embed_dim)
        
        # Transformer avec masque d'attention
        if attention_mask is not None:
            # Transformer attend des masques booléens où True = ignoré
            mask = (attention_mask == 0)
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        
        # Pooling: moyenne des tokens (en ignorant le padding)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(x.size())
            sum_embeddings = torch.sum(x * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            x = sum_embeddings / sum_mask
        else:
            x = x.mean(dim=1)  # (batch, embed_dim)
        
        # Projection vers espace latent
        x = self.to_latent(x)  # (batch, 512*4*4)
        
        # Reshape en feature map
        x = x.view(-1, self.latent_channels, self.latent_size, self.latent_size)
        return x


class TextImageDataset(Dataset):
    """
    Dataset qui combine descriptions textuelles et images CryptoPunks.
    """
    def __init__(self, metadata_path, vocab, max_length=32):
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.vocab = vocab
        self.max_length = max_length
        
        # Types mapping
        self.type_to_text = {
            0: "alien cryptopunk",
            1: "ape cryptopunk", 
            2: "female cryptopunk",
            3: "male cryptopunk",
            4: "zombie cryptopunk"
        }
    
    def __len__(self):
        return len(self.metadata)
    
    def tokenize(self, text):
        """Tokenize simple par mots"""
        words = text.lower().split()
        tokens = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        
        # Padding/truncation
        if len(tokens) < self.max_length:
            attention_mask = [1] * len(tokens) + [0] * (self.max_length - len(tokens))
            tokens = tokens + [0] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
            attention_mask = [1] * self.max_length
        
        return torch.tensor(tokens), torch.tensor(attention_mask)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Construire la description textuelle
        type_desc = self.type_to_text[item['type']]
        accessories = item.get('accessories', [])
        
        if accessories:
            # Format: "male cryptopunk with earring and mohawk"
            acc_text = " with " + " and ".join(accessories[:5])  # Limiter à 5 accessoires
            text = type_desc + acc_text
        else:
            text = type_desc
        
        # Tokenize
        token_ids, attention_mask = self.tokenize(text)
        
        # Retourner aussi les labels originaux pour comparaison
        type_idx = item['type']
        accessory_vector = torch.zeros(87)  # Adapter selon num_accessories
        for acc in accessories:
            if acc in self.vocab and self.vocab[acc] < 87:
                accessory_vector[self.vocab[acc]] = 1.0
        
        return {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'type_idx': type_idx,
            'accessory_vector': accessory_vector,
            'text': text
        }


def build_vocabulary(metadata_path):
    """Construire le vocabulaire à partir des métadonnées"""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    
    # Ajouter les mots de base
    for word in ['alien', 'ape', 'female', 'male', 'zombie', 'cryptopunk', 'with', 'and']:
        vocab[word] = idx
        idx += 1
    
    # Ajouter tous les accessoires
    all_accessories = set()
    for item in metadata:
        all_accessories.update(item.get('accessories', []))
    
    for acc in sorted(all_accessories):
        vocab[acc] = idx
        idx += 1
    
    return vocab


def train_text_encoder(config_name, epochs=50, batch_size=16, lr=1e-4):
    """
    Entraîne un encodeur de texte qui mappe les descriptions vers l'espace latent du UNet.
    """
    print("[*] Initialisation de l'entraînement du Text Encoder...")
    
    args = get_config_by_name(config_name)
    device = args.device
    
    # Créer répertoires
    os.makedirs("models/text_encoder", exist_ok=True)
    os.makedirs("results/text_encoder", exist_ok=True)
    
    # 1. Charger le modèle diffusion (frozen)
    print("[>] Chargement du modèle diffusion maître (frozen)...")
    diffusion_model = get_model(args).to(device)
    ckpt_path = os.path.join("models", args.dataset_name, config_name, "ckpt_final.pt")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint introuvable : {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    diffusion_model.load_state_dict(
        checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    )
    diffusion_model.eval()
    for p in diffusion_model.parameters():
        p.requires_grad = False
    
    # 2. Construire vocabulaire et dataset
    print("[>] Construction du vocabulaire...")
    metadata_path = "data/CRYPTOPUNKS_CLASSES/metadata.json"
    vocab = build_vocabulary(metadata_path)
    print(f"[+] Vocabulaire construit: {len(vocab)} tokens")
    
    # Sauvegarder le vocabulaire
    vocab_path = "models/text_encoder/vocab.json"
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"[+] Vocabulaire sauvegardé: {vocab_path}")
    
    dataset = TextImageDataset(metadata_path, vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # 3. Créer et initialiser le Text Encoder
    print("[>] Initialisation du Text Encoder...")
    text_encoder = TextEncoder(
        vocab_size=len(vocab),
        embed_dim=256,
        hidden_dim=512,
        latent_channels=512,
        latent_size=args.img_size // 8  # 4 pour img_size=32
    ).to(device)
    
    optimizer = optim.AdamW(text_encoder.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    
    diffusion = Diffusion(
        img_size=args.img_size,
        device=device,
        noise_steps=args.T,
        img_channels=args.image_channels
    )
    
    # 4. Boucle d'entraînement
    print(f"[>] Début de l'entraînement ({epochs} epochs)...")
    
    for epoch in range(epochs):
        text_encoder.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            token_ids = batch['token_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            type_idx = batch['type_idx'].to(device)
            accessory_vector = batch['accessory_vector'].to(device)
            
            # Générer le concept vector depuis le texte
            concept_vector = text_encoder(token_ids, attention_mask)
            
            # Créer un batch d'images synthétiques avec les vrais labels
            with torch.no_grad():
                x_real = diffusion.sample_cfg(
                    diffusion_model, 
                    n=batch_size,
                    type_idx=type_idx,
                    accessory_vector=accessory_vector,
                    guidance_scale=3.0
                )
                x_real = x_real * 2 - 1  # Normaliser [-1, 1]
            
            # Sample random timesteps
            t = torch.randint(0, args.T, (batch_size,)).to(device)
            x_t, noise_added = diffusion.noise_images(x_real, t)
            
            # Prédire avec le concept vector du texte
            noise_pred = diffusion_model(
                x_t, t, 
                type_idx=None,  # On n'utilise PAS les labels, seulement le texte
                accessory_vector=None,
                concept_vector=concept_vector
            )
            
            # Loss: le texte encodé doit produire le même bruit que les vrais labels
            loss = mse_loss(noise_pred, noise_added)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Loss moyenne: {avg_loss:.6f}")
        
        # Sauvegarder checkpoint tous les 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_save_path = f"models/text_encoder/text_encoder_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': text_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab': vocab,
            }, ckpt_save_path)
            print(f"[+] Checkpoint sauvegardé: {ckpt_save_path}")
    
    # 5. Sauvegarder le modèle final
    final_path = "models/text_encoder/text_encoder_final.pt"
    torch.save({
        'epoch': epochs,
        'model_state_dict': text_encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab': vocab,
    }, final_path)
    print(f"[+] Modèle final sauvegardé: {final_path}")
    
    print("[✓] Entraînement terminé!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='cryptopunks_classes', 
                        help='Config du modèle diffusion')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    cli_args = parser.parse_args()
    train_text_encoder(
        cli_args.config, 
        epochs=cli_args.epochs,
        batch_size=cli_args.batch_size,
        lr=cli_args.lr
    )
