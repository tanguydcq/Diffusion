import torch
import os
import json
import argparse
from torchvision.utils import save_image

# Imports de votre architecture
from src.model_vector import get_model
from src.diffusion import Diffusion
from src.config import get_config_by_name

def load_accessory_mapping(metadata_path="data/CRYPTOPUNKS_CLASSES/metadata.json"):
    """
    Charge le mapping nom_accessoire -> index depuis les métadonnées.
    """
    if not os.path.exists(metadata_path):
        return {}
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Le fichier contient déjà accessory_to_idx
    return metadata.get('accessory_to_idx', {})

def generate(tags, strength=1.5, n_samples=4, config_name="cryptopunks_classes", output_path=None):
    """
    Génère une image en combinant les vecteurs sémantiques basés sur une phrase d'entrée.
    """
    args = get_config_by_name(config_name)
    device = args.device
    
    # 1. Chargement du modèle maître
    print(f"[*] Chargement du modèle : {config_name}")
    model = get_model(args).to(device)
    ckpt_path = os.path.join("models", args.dataset_name, config_name, "ckpt_final.pt")
    
    if not os.path.exists(ckpt_path):
        print(f"[!] Erreur : Checkpoint introuvable à {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()

    diffusion = Diffusion(img_size=args.img_size, device=device, noise_steps=args.T)

    # 2. Chargement du mapping accessoires
    print(f"[*] Chargement du mapping des accessoires...")
    acc_to_idx = load_accessory_mapping()
    print(f"[+] {len(acc_to_idx)} accessoires disponibles")

    # 3. Construction du vecteur composite (La "Phrase")
    # Calcul de la taille du bottleneck: img_size // 8 (32->4 pour CryptoPunks)
    bottleneck_size = args.img_size // 8
    final_concept = torch.zeros((1, 512, bottleneck_size, bottleneck_size), device=device)
    applied_concepts = []

    for tag in tags:
        found = False
        
        # Essayer d'abord en tant que type
        type_path = f"concepts/type_{tag}.pt"
        if os.path.exists(type_path):
            concept_v = torch.load(type_path, map_location=device)
            final_concept += concept_v
            applied_concepts.append(f"type:{tag}")
            found = True
            print(f"  ✓ Type '{tag}' chargé")
            continue
        
        # Essayer ensuite en tant qu'accessoire (par nom)
        if tag in acc_to_idx:
            acc_idx = acc_to_idx[tag]
            acc_path = f"concepts/acc_{acc_idx}.pt"
            if os.path.exists(acc_path):
                concept_v = torch.load(acc_path, map_location=device)
                final_concept += concept_v
                applied_concepts.append(f"acc:{tag}")
                found = True
                print(f"  ✓ Accessoire '{tag}' (idx={acc_idx}) chargé")
                continue
        
        # Essayer aussi par index direct si c'est un nombre
        try:
            acc_idx = int(tag)
            acc_path = f"concepts/acc_{acc_idx}.pt"
            if os.path.exists(acc_path):
                concept_v = torch.load(acc_path, map_location=device)
                final_concept += concept_v
                applied_concepts.append(f"acc:{acc_idx}")
                found = True
                print(f"  ✓ Accessoire index {acc_idx} chargé")
                continue
        except ValueError:
            pass
        
        if not found:
            print(f"  ✗ Warning : Le concept '{tag}' n'est pas dans la bibliothèque.")

    if len(applied_concepts) == 0:
        print("[!] Aucun concept valide trouvé. Génération neutre...")
        final_concept = None
    else:
        print(f"[+] Composition sémantique : {len(applied_concepts)} concepts (Force: {strength})")

    # 4. Inférence Steering (Génération)
    # On utilise un conditionnement neutre (Male) par défaut pour laisser les vecteurs agir
    # Type 3 = Male dans l'index standard
    base_labels = torch.full((n_samples,), 3, dtype=torch.long, device=device)
    acc_zeros = torch.zeros(n_samples, args.num_accessories, device=device)

    print(f"[*] Échantillonnage de {n_samples} images...")
    with torch.no_grad():
        # On injecte la somme des vecteurs multipliée par la force (lambda)
        cv = final_concept * strength if final_concept is not None else None
        samples = diffusion.sample_cfg(
            model, 
            n=n_samples, 
            type_idx=base_labels, 
            accessory_vector=acc_zeros,
            concept_vector=cv,
            guidance_scale=5.0
        )

    # 5. Sauvegarde
    os.makedirs("output", exist_ok=True)
    
    if output_path:
        filename = output_path
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    else:
        safe_name = '_'.join([c.split(':')[-1] for c in applied_concepts])[:50]  # Limiter la longueur
        filename = f"output/gen_{safe_name}.png"
    
    save_image(samples, filename, nrow=2)
    print(f"[!] Résultat sauvegardé dans : {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Générateur CryptoPunk par phrase sémantique")
    parser.add_argument("phrase", nargs="+", help="Liste de tags (ex: alien pipe shades)")
    parser.add_argument("--strength", type=float, default=1.5, help="Intensité de l'injection (lambda)")
    parser.add_argument("--config", type=str, default="cryptopunks_classes")
    parser.add_argument("--n", type=int, default=4, help="Nombre d'images à générer")
    parser.add_argument("--output", type=str, default=None, help="Chemin de sortie personnalisé")

    args = parser.parse_args()
    generate(args.phrase, args.strength, args.n, args.config, args.output)