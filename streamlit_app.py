import streamlit as st
import torch
import numpy as np
import json
import os
from PIL import Image
from src.diffusion import Diffusion
from src.model import get_model
from src.config import get_config_by_name
import argparse

# Page configuration
st.set_page_config(
    page_title="CryptoPunk Generator",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 12px;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model_and_metadata(config_name="cryptopunks_classes_fast"):
    """Load the trained model and metadata"""
    # Get configuration
    config = get_config_by_name(config_name)
    args = argparse.Namespace(**vars(config))
    
    # Load metadata
    metadata_path = "data/CRYPTOPUNKS_CLASSES/metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Setup model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    model = get_model(args).to(device)
    
    # Load checkpoint
    run_name = os.path.join(args.dataset_name, config_name)
    models_dir = os.path.join("models", run_name)
    ckpt_path = None
    latest_epoch = -1
    
    if os.path.exists(models_dir):
        # Scan for latest checkpoint
        for f in os.listdir(models_dir):
            if f.startswith('ckpt_') and f.endswith('.pt') and f != 'ckpt_final.pt':
                try:
                    epoch_num = int(f.split('_')[1].split('.')[0])
                    if epoch_num > latest_epoch:
                        latest_epoch = epoch_num
                        ckpt_path = os.path.join(models_dir, f)
                except ValueError:
                    continue
        
        # Check final checkpoint
        final_path = os.path.join(models_dir, "ckpt_final.pt")
        if os.path.exists(final_path):
            try:
                final_ckpt = torch.load(final_path, map_location='cpu')
                final_epoch = final_ckpt.get('epoch', -1) if isinstance(final_ckpt, dict) else -1
                if final_epoch >= latest_epoch:
                    latest_epoch = final_epoch
                    ckpt_path = final_path
            except:
                pass
    
    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        epoch_info = f"Epoch {latest_epoch}"
    else:
        epoch_info = "No checkpoint found - using random weights"
    
    model.eval()
    
    # Setup diffusion
    diffusion = Diffusion(
        img_size=args.img_size,
        img_channels=args.image_channels,
        device=device,
        noise_steps=args.T,
        beta_start=args.beta_start,
        beta_end=args.beta_end
    )
    
    return model, diffusion, metadata, device, epoch_info

def generate_punk(model, diffusion, device, type_idx, accessory_indices):
    """Generate a CryptoPunk with specified attributes"""
    with torch.no_grad():
        # Prepare inputs
        type_tensor = torch.tensor([type_idx], device=device)
        
        # Create multi-hot accessory vector
        accessory_vector = torch.zeros(1, 87, device=device)
        for idx in accessory_indices:
            accessory_vector[0, idx] = 1.0
        
        # Generate image
        x = torch.randn(1, 3, 32, 32, device=device)
        
        for t_idx in reversed(range(diffusion.noise_steps)):
            t = torch.full((1,), t_idx, device=device, dtype=torch.long)
            
            # Predict noise with conditioning
            predicted_noise = model(x, t, type_idx=type_tensor, accessory_vector=accessory_vector)
            
            # Denoise step
            alpha = diffusion.alpha[t][:, None, None, None]
            alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
            beta = diffusion.beta[t][:, None, None, None]
            
            if t_idx > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = 1 / torch.sqrt(alpha) * (
                x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
            ) + torch.sqrt(beta) * noise
        
        # Convert to image
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        img_array = x[0].permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(img_array)
        
        return img

# Main app
st.title("🎨 CryptoPunk Generator")
st.markdown("Crée ton propre CryptoPunk en choisissant le type et les accessoires !")

# Load model and metadata
try:
    with st.spinner("Chargement du modèle..."):
        model, diffusion, metadata, device, epoch_info = load_model_and_metadata()
    st.success(f"✅ Modèle chargé ({epoch_info})")
except Exception as e:
    st.error(f"❌ Erreur lors du chargement du modèle: {e}")
    st.stop()

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Caractéristiques")
    
    # Type selection
    types = list(metadata['type_to_idx'].keys())
    selected_type = st.selectbox(
        "🧬 Type de Punk",
        types,
        help="Choisis le type de CryptoPunk"
    )
    type_idx = metadata['type_to_idx'][selected_type]
    
    # Accessory selection
    st.subheader("🎭 Accessoires")
    accessories = sorted(metadata['accessory_to_idx'].keys())
    
    # Group accessories by category for easier selection
    st.markdown("**Sélectionne les accessoires (multi-choix) :**")
    selected_accessories = st.multiselect(
        "Accessoires",
        accessories,
        help="Tu peux sélectionner plusieurs accessoires",
        label_visibility="collapsed"
    )
    
    # Get indices of selected accessories
    accessory_indices = [metadata['accessory_to_idx'][acc] for acc in selected_accessories]
    
    # Display selection summary
    st.info(f"**Sélection:** {selected_type} avec {len(selected_accessories)} accessoire(s)")
    
    # Generate button
    generate_clicked = st.button("🎲 Générer mon CryptoPunk", use_container_width=True)
    
    # Random option
    st.markdown("---")
    if st.button("🎰 Générer aléatoirement", use_container_width=True):
        # Random type
        selected_type = np.random.choice(types)
        type_idx = metadata['type_to_idx'][selected_type]
        
        # Random accessories (2-4 accessories)
        num_accessories = np.random.randint(2, 5)
        selected_accessories = list(np.random.choice(accessories, size=num_accessories, replace=False))
        accessory_indices = [metadata['accessory_to_idx'][acc] for acc in selected_accessories]
        
        generate_clicked = True

with col2:
    st.header("Résultat")
    
    if generate_clicked:
        with st.spinner("Génération en cours..."):
            try:
                # Generate the punk
                img = generate_punk(model, diffusion, device, type_idx, accessory_indices)
                
                # Display the image (enlarged)
                st.image(img, caption=f"{selected_type} punk", use_container_width=True)
                
                # Display selected attributes
                st.markdown("### Attributs:")
                st.markdown(f"- **Type:** {selected_type}")
                if selected_accessories:
                    st.markdown("- **Accessoires:**")
                    for acc in selected_accessories:
                        st.markdown(f"  - {acc}")
                else:
                    st.markdown("- **Accessoires:** Aucun")
                
                # Download button
                img_resized = img.resize((256, 256), Image.NEAREST)  # Pixel art scaling
                img_bytes = img_resized.tobytes()
                st.download_button(
                    label="💾 Télécharger l'image",
                    data=img_resized.tobytes(),
                    file_name=f"cryptopunk_{selected_type.lower()}.png",
                    mime="image/png"
                )
                
            except Exception as e:
                st.error(f"❌ Erreur lors de la génération: {e}")
    else:
        st.info("👆 Choisis les caractéristiques et clique sur 'Générer' pour créer ton CryptoPunk !")
        
        # Show example punk
        st.markdown("### Exemple de génération:")
        st.image("https://via.placeholder.com/256x256?text=Ton+CryptoPunk+ici", 
                use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>🔥 Génère des CryptoPunks uniques avec l'IA</p>
        <p style='font-size: 12px; color: gray;'>Modèle: Diffusion conditionnelle multi-attributs</p>
    </div>
    """, unsafe_allow_html=True)
