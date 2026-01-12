import torch
import argparse
import os
from src.model_vector import get_model
from src.diffusion import Diffusion
from src.config import get_config_by_name
from torchvision.utils import save_image

def generate_phrase(phrase_list, strength=1.5, n_images=16, output_path="result_phrase.png", guidance_scale=3.0):
    """
    Génère des images en combinant plusieurs concepts vectoriels.
    
    Args:
        phrase_list: Liste de noms de concepts (ex: ['type_alien', 'acc_pipe'])
        strength: Multiplicateur de force des concepts (1.0 = normal)
        n_images: Nombre d'images à générer
        output_path: Chemin de sauvegarde
        guidance_scale: Force du CFG
    """
    config_name = "cryptopunks_classes"
    args = get_config_by_name(config_name)
    device = args.device 
    
    print(f"[*] Chargement du modèle ({config_name})...")
    model = get_model(args).to(device)
    
    # Chargement du checkpoint
    ckpt_path = f"models/CRYPTOPUNKS_CLASSES/{config_name}/ckpt_final.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint introuvable : {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(
        checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    )
    model.eval()
    
    diffusion = Diffusion(
        img_size=args.img_size, 
        device=device, 
        noise_steps=args.T,
        img_channels=args.image_channels,
        beta_start=args.beta_start,
        beta_end=args.beta_end
    )

    # Calcul de la taille du bottleneck
    bottleneck_size = args.img_size // 8  # 32 -> 4
    
    # Somme des vecteurs conceptuels
    print(f"[>] Chargement et combinaison des concepts : {phrase_list}")
    final_c = torch.zeros((1, 512, bottleneck_size, bottleneck_size), device=device)
    loaded_concepts = []
    
    for concept in phrase_list:
        path = f"concepts/{concept}.pt"
        if os.path.exists(path):
            concept_vec = torch.load(path, map_location=device)
            final_c += concept_vec
            loaded_concepts.append(concept)
            print(f"  ✓ {concept}")
        else:
            print(f"  ✗ Warning: Concept '{concept}' introuvable dans concepts/")
    
    if not loaded_concepts:
        print("[!] Aucun concept chargé ! Génération inconditionnelle.")
        final_c = None
    else:
        print(f"[+] {len(loaded_concepts)} concept(s) combiné(s) avec strength={strength}")
        final_c = final_c * strength
        # Répéter pour batch
        final_c = final_c.repeat(n_images, 1, 1, 1)

    # Génération avec steering conceptuel
    print(f"[>] Génération de {n_images} images (guidance_scale={guidance_scale})...")
    with torch.no_grad():
        images = diffusion.sample_cfg(
            model, 
            n=n_images, 
            type_idx=None,  # Pas de conditionnement par labels
            accessory_vector=None,
            concept_vector=final_c,
            guidance_scale=guidance_scale
        )
    
    # Sauvegarde
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    save_image(images, output_path, nrow=4)
    print(f"[✓] Images générées : {output_path}")
    
    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Générer des CryptoPunks en combinant des concepts vectoriels"
    )
    parser.add_argument(
        '--tags', 
        nargs='+', 
        required=True,
        help='Liste des concepts à combiner (ex: type_alien acc_pipe acc_shades)'
    )
    parser.add_argument(
        '--strength', 
        type=float, 
        default=1.5,
        help='Force des concepts (défaut: 1.5)'
    )
    parser.add_argument(
        '--n-images', 
        type=int, 
        default=16,
        help='Nombre d\'images à générer (défaut: 16)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='results/concepts_validation/steered_generation.png',
        help='Chemin de sortie (défaut: results/concepts_validation/steered_generation.png)'
    )
    parser.add_argument(
        '--guidance', 
        type=float, 
        default=3.0,
        help='CFG guidance scale (défaut: 3.0)'
    )
    
    args = parser.parse_args()
    
    generate_phrase(
        phrase_list=args.tags,
        strength=args.strength,
        n_images=args.n_images,
        output_path=args.output,
        guidance_scale=args.guidance
    )
