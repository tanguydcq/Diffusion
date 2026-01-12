import torch
import json
import argparse
from torchvision.utils import save_image

from src.model_vector import get_model
from src.diffusion import Diffusion
from src.config import get_config_by_name
from train_text_encoder import TextEncoder


def load_text_encoder(checkpoint_path, device='cuda'):
    """Charger le text encoder depuis un checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    vocab = checkpoint['vocab']
    vocab_size = len(vocab)
    
    # Créer le modèle avec les bonnes dimensions
    text_encoder = TextEncoder(
        vocab_size=vocab_size,
        embed_dim=256,
        hidden_dim=512,
        latent_channels=512,
        latent_size=4  # Pour img_size=32
    )
    
    text_encoder.load_state_dict(checkpoint['model_state_dict'])
    text_encoder.to(device)
    text_encoder.eval()
    
    return text_encoder, vocab


def tokenize_text(text, vocab, max_length=32):
    """Tokenize une description textuelle"""
    words = text.lower().split()
    tokens = [vocab.get(word, vocab.get('<UNK>', 1)) for word in words]
    
    # Padding/truncation
    if len(tokens) < max_length:
        attention_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
        tokens = tokens + [0] * (max_length - len(tokens))
    else:
        tokens = tokens[:max_length]
        attention_mask = [1] * max_length
    
    return torch.tensor(tokens), torch.tensor(attention_mask)


def generate_from_text(text_prompt, config_name='cryptopunks_classes', 
                      text_encoder_path='models/text_encoder/text_encoder_final.pt',
                      guidance_scale=3.0, n_images=8, strength=1.0):
    """
    Génère des CryptoPunks à partir d'une description textuelle.
    
    Args:
        text_prompt: Description textuelle (ex: "male cryptopunk with earring and mohawk")
        config_name: Nom de la config du modèle diffusion
        text_encoder_path: Chemin vers le checkpoint du text encoder
        guidance_scale: Force du CFG
        n_images: Nombre d'images à générer
        strength: Multiplicateur du concept vector (1.0 = normal, >1 = plus fort)
    """
    print(f"[*] Génération d'images depuis: '{text_prompt}'")
    
    args = get_config_by_name(config_name)
    device = args.device
    
    # 1. Charger le text encoder
    print("[>] Chargement du Text Encoder...")
    text_encoder, vocab = load_text_encoder(text_encoder_path, device)
    
    # 2. Charger le modèle diffusion
    print("[>] Chargement du modèle Diffusion...")
    diffusion_model = get_model(args).to(device)
    ckpt_path = f"models/{args.dataset_name}/{config_name}/ckpt_final.pt"
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    diffusion_model.load_state_dict(
        checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    )
    diffusion_model.eval()
    
    diffusion = Diffusion(
        img_size=args.img_size,
        device=device,
        noise_steps=args.T,
        img_channels=args.image_channels,
        beta_start=args.beta_start,
        beta_end=args.beta_end
    )
    
    # 3. Encoder le texte en concept vector
    print(f"[>] Encoding du texte...")
    token_ids, attention_mask = tokenize_text(text_prompt, vocab)
    token_ids = token_ids.unsqueeze(0).to(device)  # (1, seq_len)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    
    with torch.no_grad():
        concept_vector = text_encoder(token_ids, attention_mask)
        # Répéter pour n_images
        concept_vector = concept_vector.repeat(n_images, 1, 1, 1) * strength
    
    # 4. Générer les images
    print(f"[>] Génération de {n_images} images...")
    with torch.no_grad():
        images = diffusion.sample_cfg(
            diffusion_model,
            n=n_images,
            type_idx=None,  # Pas de labels, seulement le texte
            accessory_vector=None,
            concept_vector=concept_vector,
            guidance_scale=guidance_scale
        )
    
    # 5. Sauvegarder
    output_path = f"results/text_encoder/generated_{text_prompt.replace(' ', '_')[:50]}.png"
    save_image(images, output_path, nrow=4)
    print(f"[+] Images sauvegardées: {output_path}")
    
    return images


def interactive_mode(config_name='cryptopunks_classes'):
    """Mode interactif pour générer des images depuis du texte"""
    print("\n" + "="*60)
    print("  TEXT-TO-IMAGE GENERATOR - CryptoPunks")
    print("="*60)
    print("\nExemples de prompts:")
    print("  - 'male cryptopunk with earring and mohawk'")
    print("  - 'female cryptopunk with lipstick'")
    print("  - 'zombie cryptopunk'")
    print("  - 'alien cryptopunk with cap'")
    print("\nTapez 'quit' pour quitter\n")
    
    while True:
        text_prompt = input("Prompt > ").strip()
        
        if text_prompt.lower() in ['quit', 'exit', 'q']:
            print("Bye!")
            break
        
        if not text_prompt:
            continue
        
        try:
            generate_from_text(
                text_prompt,
                config_name=config_name,
                n_images=8,
                guidance_scale=3.0,
                strength=1.2
            )
            print("✓ Génération terminée!\n")
        except Exception as e:
            print(f"Erreur: {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Générer des CryptoPunks depuis du texte")
    parser.add_argument('--prompt', type=str, help='Description textuelle')
    parser.add_argument('--config', type=str, default='cryptopunks_classes')
    parser.add_argument('--text-encoder', type=str, 
                       default='models/text_encoder/text_encoder_final.pt')
    parser.add_argument('--guidance', type=float, default=3.0)
    parser.add_argument('--strength', type=float, default=1.2)
    parser.add_argument('--n-images', type=int, default=8)
    parser.add_argument('--interactive', action='store_true', 
                       help='Mode interactif')
    
    cli_args = parser.parse_args()
    
    if cli_args.interactive:
        interactive_mode(cli_args.config)
    elif cli_args.prompt:
        generate_from_text(
            cli_args.prompt,
            config_name=cli_args.config,
            text_encoder_path=cli_args.text_encoder,
            guidance_scale=cli_args.guidance,
            n_images=cli_args.n_images,
            strength=cli_args.strength
        )
    else:
        print("Utilisez --prompt 'votre texte' ou --interactive")
        print("Exemple: python infer_text_to_image.py --prompt 'male cryptopunk with earring'")
