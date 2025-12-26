import streamlit as st
import torch
import numpy as np
import json
import os
import subprocess
import glob
from pathlib import Path
from PIL import Image
from datetime import datetime
import pandas as pd
from io import BytesIO
from diffusion import Diffusion
from model import get_model

# Page configuration
st.set_page_config(
    page_title="Diffusion Training Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🚀 Navigation")
page = st.sidebar.radio(
    "Choisir une page:",
    ["📊 Dashboard", "🎯 Training", "⚙️ Configurations", "🎨 Inference"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Diffusion Model Manager**
- Gérer les entraînements
- Suivre les modèles
- Générer des images
""")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_available_configs():
    """Get all available configurations from config.py"""
    try:
        from config import CONFIGS
        return CONFIGS
    except:
        return {}

def scan_trained_models():
    """Scan models directory to find trained models"""
    models_info = []
    models_dir = Path("models")
    
    if not models_dir.exists():
        return pd.DataFrame()
    
    for dataset_dir in models_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        for config_dir in dataset_dir.iterdir():
            if not config_dir.is_dir():
                continue
            
            # Find checkpoints
            checkpoints = list(config_dir.glob("ckpt_*.pt"))
            if not checkpoints:
                continue
            
            # Get latest checkpoint
            latest_epoch = 0
            latest_ckpt = None
            
            for ckpt in checkpoints:
                if ckpt.name == "ckpt_final.pt":
                    try:
                        checkpoint = torch.load(ckpt, map_location='cpu')
                        epoch = checkpoint.get('epoch', 0) if isinstance(checkpoint, dict) else 0
                        if epoch >= latest_epoch:
                            latest_epoch = epoch
                            latest_ckpt = ckpt
                    except:
                        pass
                else:
                    try:
                        epoch_num = int(ckpt.stem.split('_')[1])
                        if epoch_num > latest_epoch:
                            latest_epoch = epoch_num
                            latest_ckpt = ckpt
                    except:
                        pass
            
            if latest_ckpt:
                # Get file size and modification time
                size_mb = latest_ckpt.stat().st_size / (1024 * 1024)
                mod_time = datetime.fromtimestamp(latest_ckpt.stat().st_mtime)
                
                models_info.append({
                    "Dataset": dataset_dir.name,
                    "Config": config_dir.name,
                    "Epoch": latest_epoch,
                    "Checkpoint": latest_ckpt.name,
                    "Size (MB)": f"{size_mb:.1f}",
                    "Last Modified": mod_time.strftime("%Y-%m-%d %H:%M"),
                    "Path": str(config_dir)
                })
    
    return pd.DataFrame(models_info)

def check_tensorboard_running():
    """Check if TensorBoard is running"""
    try:
        result = subprocess.run(
            ["powershell", "-Command", "Get-Process | Where-Object {$_.ProcessName -eq 'tensorboard'} | Select-Object -First 1"],
            capture_output=True,
            text=True,
            timeout=2
        )
        return "tensorboard" in result.stdout.lower()
    except:
        return False

def get_tensorboard_url():
    """Get TensorBoard URL"""
    return "http://localhost:6006"

def start_tensorboard():
    """Start TensorBoard in background"""
    try:
        subprocess.Popen(
            ["tensorboard", "--logdir=runs", "--port=6006"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
        )
        return True
    except Exception as e:
        st.error(f"Erreur lors du lancement de TensorBoard: {e}")
        return False

def get_sample_images(dataset, config):
    """Get sample images from results directory"""
    results_dir = Path(f"results/{dataset}/{config}")
    if not results_dir.exists():
        return []
    
    # Find sample images
    images = sorted(results_dir.glob("sample_epoch_*.png"))
    return images[-5:] if images else []  # Last 5 samples

@st.cache_resource
def load_cryptopunk_model(config_name="cryptopunks_classes_fast"):
    """Load CryptoPunk model and metadata for interactive generation"""
    try:
        from config import get_config_by_name
        import argparse
        
        # Get configuration
        config = get_config_by_name(config_name)
        args = argparse.Namespace(**vars(config))
        
        # Load metadata
        metadata_path = "data/CRYPTOPUNKS_CLASSES/metadata.json"
        if not os.path.exists(metadata_path):
            return None, None, None, None, "Metadata not found"
        
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
        
        if not ckpt_path or not os.path.exists(ckpt_path):
            return None, None, None, None, "No checkpoint found"
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
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
        
        epoch_info = f"Epoch {latest_epoch}"
        return model, diffusion, metadata, device, epoch_info
    except Exception as e:
        return None, None, None, None, str(e)

def generate_cryptopunk(model, diffusion, device, type_idx, accessory_indices, cfg_scale=3.0):
    """Generate a CryptoPunk with specified attributes using CFG"""
    with torch.no_grad():
        # Prepare inputs
        type_tensor = torch.tensor([type_idx], device=device)
        
        # Create multi-hot accessory vector
        accessory_vector = torch.zeros(1, 87, device=device)
        for idx in accessory_indices:
            accessory_vector[0, idx] = 1.0
        
        # Generate image with CFG
        use_cfg = cfg_scale > 1.0
        
        x = torch.randn(1, 3, 32, 32, device=device)
        
        for t_idx in reversed(range(diffusion.noise_steps)):
            t = torch.full((1,), t_idx, device=device, dtype=torch.long)
            
            # === CONDITIONAL PREDICTION ===
            noise_cond = model(x, t, type_idx=type_tensor, accessory_vector=accessory_vector)
            
            if use_cfg:
                # === UNCONDITIONAL PREDICTION ===
                noise_uncond = model(x, t, type_idx=None, accessory_vector=None)
                
                # === CFG COMBINATION ===
                predicted_noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            else:
                predicted_noise = noise_cond
            
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

def generate_simple_images(config_name, num_samples=4, cfg_scale=3.0):
    """Generate images for simple conditional or unconditional models"""
    try:
        from config import get_config_by_name
        
        # Load config
        args = get_config_by_name(config_name)
        device = args.device
        
        # Load model
        model = get_model(args).to(device)
        
        # Load checkpoint
        run_name = os.path.join(args.dataset_name, config_name)
        models_dir = os.path.join("models", run_name)
        ckpt_path = None
        latest_epoch = -1
        
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.startswith('ckpt_') and f.endswith('.pt') and f != 'ckpt_final.pt':
                    try:
                        epoch_num = int(f.split('_')[1].split('.')[0])
                        if epoch_num > latest_epoch:
                            latest_epoch = epoch_num
                            ckpt_path = os.path.join(models_dir, f)
                    except ValueError:
                        continue
            
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
        
        if not ckpt_path or not os.path.exists(ckpt_path):
            return None, "No checkpoint found"
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
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
        
        # Generate images
        use_cfg = cfg_scale > 1.0 and (
            (hasattr(model, 'num_classes') and model.num_classes is not None) or
            (hasattr(model, 'attr_embedding') and model.attr_embedding is not None)
        )
        
        # Prepare conditioning if needed
        labels = None
        if hasattr(model, 'num_classes') and model.num_classes is not None:
            # Class conditioning (MNIST) - cycle through classes
            labels = torch.tensor([i % model.num_classes for i in range(num_samples)]).to(device)
        
        # Sample images
        if use_cfg:
            sampled_images = diffusion.sample_cfg(
                model, n=num_samples, guidance_scale=cfg_scale, 
                save_gif=False, labels=labels
            )
        else:
            sampled_images = diffusion.sample(
                model, n=num_samples, save_gif=False, labels=labels
            )
        
        # Convert to PIL images
        images = []
        for i in range(num_samples):
            img_tensor = sampled_images[i]
            img_array = (img_tensor * 255).type(torch.uint8).permute(1, 2, 0).cpu().numpy()
            
            # Handle grayscale
            if img_array.shape[2] == 1:
                img_array = img_array.squeeze(-1)
            
            img = Image.fromarray(img_array)
            images.append(img)
        
        return images, f"Epoch {latest_epoch}"
    
    except Exception as e:
        return None, str(e)

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================

if page == "📊 Dashboard":
    st.markdown("<div class='main-header'>📊 Training Dashboard</div>", unsafe_allow_html=True)
    
    # TensorBoard Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 TensorBoard")
        tb_running = check_tensorboard_running()
        
        if tb_running:
            st.success("✅ TensorBoard est en cours d'exécution")
            st.markdown(f"🔗 **Lien:** [{get_tensorboard_url()}]({get_tensorboard_url()})")
        else:
            st.warning("⚠️ TensorBoard n'est pas en cours d'exécution")
    
    with col2:
        st.write("")
        st.write("")
        if not tb_running:
            if st.button("🚀 Lancer TensorBoard", use_container_width=True):
                if start_tensorboard():
                    st.success("TensorBoard lancé avec succès!")
                    st.rerun()
        else:
            st.info("Accède à TensorBoard via le lien")
    
    st.markdown("---")
    
    # Models Overview
    st.subheader("🤖 Modèles Entraînés")
    
    models_df = scan_trained_models()
    
    if models_df.empty:
        st.info("Aucun modèle entraîné trouvé. Lance un entraînement dans l'onglet 🎯 Training!")
    else:
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📦 Total Modèles", len(models_df))
        
        with col2:
            datasets = models_df['Dataset'].nunique()
            st.metric("📁 Datasets", datasets)
        
        with col3:
            total_size = models_df['Size (MB)'].str.replace(' MB', '').astype(float).sum()
            st.metric("💾 Taille Totale", f"{total_size:.1f} MB")
        
        with col4:
            max_epoch = models_df['Epoch'].max()
            st.metric("🏆 Epoch Max", int(max_epoch))
        
        st.markdown("---")
        
        # Display table
        st.dataframe(
            models_df.drop(columns=['Path']),
            use_container_width=True,
            hide_index=True
        )
        
        # Show sample images for selected model
        st.markdown("---")
        st.subheader("🖼️ Aperçu des Générations")
        
        if len(models_df) > 0:
            selected_idx = st.selectbox(
                "Sélectionner un modèle:",
                range(len(models_df)),
                format_func=lambda i: f"{models_df.iloc[i]['Dataset']} - {models_df.iloc[i]['Config']} (Epoch {models_df.iloc[i]['Epoch']})"
            )
            
            selected_model = models_df.iloc[selected_idx]
            sample_images = get_sample_images(selected_model['Dataset'], selected_model['Config'])
            
            if sample_images:
                cols = st.columns(min(5, len(sample_images)))
                for idx, img_path in enumerate(sample_images):
                    with cols[idx % 5]:
                        img = Image.open(img_path)
                        epoch = img_path.stem.split('_')[-1]
                        st.image(img, caption=f"Epoch {epoch}", use_container_width=True)
            else:
                st.info("Aucune image de génération disponible pour ce modèle.")

# ============================================================================
# PAGE: TRAINING
# ============================================================================

elif page == "🎯 Training":
    st.markdown("<div class='main-header'>🎯 Lancer un Entraînement</div>", unsafe_allow_html=True)
    
    # Configuration selection
    configs = get_available_configs()
    
    if not configs:
        st.error("Aucune configuration trouvée dans config.py")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("⚙️ Configuration")
        
        config_name = st.selectbox(
            "Choisir une configuration:",
            list(configs.keys()),
            help="Sélectionne la configuration d'entraînement"
        )
        
        # Display config details
        if config_name:
            config = configs[config_name]
            
            st.markdown("**Paramètres de la configuration:**")
            
            # Convert dict to Config object if needed
            if isinstance(config, dict):
                from config import Config
                config_obj = Config(**config)
            else:
                config_obj = config
            
            config_info = {
                "Dataset": config_obj.dataset_name,
                "Epochs": config_obj.epochs,
                "Batch Size": config_obj.batch_size,
                "Learning Rate": config_obj.lr,
                "T (steps)": config_obj.T,
                "Image Size": f"{config_obj.img_size}x{config_obj.img_size}",
                "Channels": config_obj.image_channels,
            }
            
            if hasattr(config_obj, 'num_classes') and config_obj.num_classes:
                config_info["Classes"] = config_obj.num_classes
            
            if hasattr(config_obj, 'num_types') and config_obj.num_types:
                config_info["Types"] = config_obj.num_types
                config_info["Accessories"] = config_obj.num_accessories
            
            # Display as table
            config_df = pd.DataFrame(list(config_info.items()), columns=["Paramètre", "Valeur"])
            st.dataframe(config_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("🎮 Options")
        
        reset_training = st.checkbox(
            "🔄 Reset (supprimer checkpoints)",
            value=False,
            help="Supprime les checkpoints existants et recommence"
        )
        
        use_cuda = st.checkbox(
            "⚡ Utiliser CUDA",
            value=torch.cuda.is_available(),
            disabled=not torch.cuda.is_available(),
            help="Entraîner sur GPU (si disponible)"
        )
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            st.success(f"✅ GPU: {gpu_name}")
        else:
            st.warning("⚠️ Aucun GPU détecté")
    
    st.markdown("---")
    
    # Training controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🚀 Lancer l'Entraînement", use_container_width=True, type="primary"):
            if config_name:
                with st.spinner("Démarrage de l'entraînement..."):
                    try:
                        # Build command
                        cmd = [".venv\\Scripts\\python.exe", "train.py", "--config", config_name]
                        if reset_training:
                            cmd.append("--reset")
                        
                        # Launch in new console window
                        subprocess.Popen(
                            cmd,
                            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0,
                            cwd=os.getcwd()
                        )
                        st.success(f"✅ Entraînement lancé: {config_name}")
                        st.info("📺 L'entraînement tourne dans un nouveau terminal. Ferme le terminal pour l'arrêter.")
                    except Exception as e:
                        st.error(f"❌ Erreur: {e}")
    
    with col2:
        if st.button("📊 Ouvrir TensorBoard", use_container_width=True):
            if not check_tensorboard_running():
                with st.spinner("Lancement de TensorBoard..."):
                    if start_tensorboard():
                        st.success("✅ TensorBoard lancé!")
                    else:
                        st.error("❌ Échec du lancement")
            st.markdown(f"🔗 [Ouvrir TensorBoard]({get_tensorboard_url()})")
    
    with col3:
        if st.button("📁 Ouvrir Dossier Modèles", use_container_width=True):
            models_dir = Path("models")
            if models_dir.exists():
                os.startfile(str(models_dir))
                st.success("📂 Dossier ouvert!")
            else:
                st.error("Dossier models introuvable")
    
    st.markdown("---")
    
    # Training tips
    with st.expander("💡 Conseils d'Entraînement"):
        st.markdown("""
        **Configurations recommandées:**
        
        - **Fast prototyping:** Utilise les configs `*_fast` (T=500, moins d'epochs)
        - **Qualité optimale:** Utilise les configs standards (T=1000, 100-200 epochs)
        - **Classes conditionnelles:** Utilise les configs `*_classes` pour génération conditionnelle
        
        **Monitoring:**
        - TensorBoard te montre les courbes de MSE, grad norm, learning rate
        - Les samples sont générés tous les 10 epochs dans `results/`
        - Les checkpoints sont sauvegardés tous les 20 epochs dans `models/`
        
        **Arrêt d'urgence:**
        - Ferme le terminal pour arrêter l'entraînement
        - Les checkpoints sont automatiquement sauvegardés
        """)

# ============================================================================
# PAGE: CONFIGURATIONS
# ============================================================================

elif page == "⚙️ Configurations":
    st.markdown("<div class='main-header'>⚙️ Configurations Disponibles</div>", unsafe_allow_html=True)
    
    configs = get_available_configs()
    
    if not configs:
        st.error("Aucune configuration trouvée")
        st.stop()
    
    st.write(f"**{len(configs)} configuration(s) disponible(s)**")
    
    # Group configs by dataset
    configs_by_dataset = {}
    for name, config in configs.items():
        # Handle both dict and Config object
        if isinstance(config, dict):
            from config import Config
            config_obj = Config(**config)
        else:
            config_obj = config
        
        dataset = config_obj.dataset_name
        if dataset not in configs_by_dataset:
            configs_by_dataset[dataset] = []
        configs_by_dataset[dataset].append((name, config_obj))
    
    # Display by dataset
    for dataset, dataset_configs in configs_by_dataset.items():
        st.subheader(f"📁 {dataset}")
        
        for config_name, config in dataset_configs:
            with st.expander(f"🔧 {config_name}"):
                col1, col2 = st.columns(2)
                
                # config is already a Config object from the grouping step above
                with col1:
                    st.markdown("**Paramètres d'entraînement:**")
                    st.write(f"- Epochs: `{config.epochs}`")
                    st.write(f"- Batch Size: `{config.batch_size}`")
                    st.write(f"- Learning Rate: `{config.lr}`")
                    st.write(f"- T (timesteps): `{config.T}`")
                    st.write(f"- Beta: `{config.beta_start}` → `{config.beta_end}`")
                
                with col2:
                    st.markdown("**Paramètres du modèle:**")
                    st.write(f"- Image Size: `{config.img_size}x{config.img_size}`")
                    st.write(f"- Channels: `{config.image_channels}`")
                    st.write(f"- Time Embedding: `{config.time_emb_dim}`")
                    
                    if hasattr(config, 'num_classes') and config.num_classes:
                        st.write(f"- Classes: `{config.num_classes}`")
                    
                    if hasattr(config, 'num_types') and config.num_types:
                        st.write(f"- Types: `{config.num_types}`")
                        st.write(f"- Accessories: `{config.num_accessories}`")
                    
                    # CFG parameters
                    if hasattr(config, 'cfg_dropout') and config.cfg_dropout:
                        st.markdown("**Classifier-Free Guidance (CFG):**")
                        st.write(f"- CFG Dropout: `{config.cfg_dropout}` ({int(config.cfg_dropout*100)}% uncond)")
                        st.write(f"- CFG Scale: `{config.cfg_scale}` (inference)")
    
    st.markdown("---")
    
    # Add new config helper
    with st.expander("➕ Créer une Nouvelle Configuration"):
        st.markdown("""
        **Pour ajouter une nouvelle configuration:**
        
        1. Ouvre `config.py`
        2. Crée un nouveau dictionnaire de config:
        ```python
        config_mon_nouveau_modele = {
            "dataset_name": "MNIST",
            "epochs": 100,
            "lr": 3e-4,
            "T": 1000,
            "batch_size": 128,
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "num_classes": 10,  # Optionnel
        }
        ```
        3. Ajoute-la au dictionnaire `CONFIGS`:
        ```python
        CONFIGS = {
            ...
            "mon_nouveau_modele": Config(**config_mon_nouveau_modele),
        }
        ```
        4. Redémarre cette application
        """)

# ============================================================================
# PAGE: INFERENCE
# ============================================================================

elif page == "🎨 Inference":
    st.markdown("<div class='main-header'>🎨 Génération d'Images</div>", unsafe_allow_html=True)
    
    # Check for trained models
    models_df = scan_trained_models()
    
    if models_df.empty:
        st.warning("⚠️ Aucun modèle entraîné trouvé. Lance un entraînement d'abord!")
        st.stop()
    
    # Model selection
    st.subheader("🤖 Sélection du Modèle")
    
    selected_idx = st.selectbox(
        "Choisir un modèle:",
        range(len(models_df)),
        format_func=lambda i: f"{models_df.iloc[i]['Dataset']} - {models_df.iloc[i]['Config']} (Epoch {models_df.iloc[i]['Epoch']})"
    )
    
    selected_model = models_df.iloc[selected_idx]
    dataset = selected_model['Dataset']
    config_name = selected_model['Config']
    
    st.info(f"📦 Modèle sélectionné: **{dataset} / {config_name}** (Epoch {selected_model['Epoch']})")
    
    st.markdown("---")
    
    # Check if it's a CryptoPunks conditional model
    is_cryptopunks = "CRYPTOPUNKS_CLASSES" in dataset
    
    if is_cryptopunks:
        # ====================================================================
        # CRYPTOPUNK GENERATOR INTERFACE
        # ====================================================================
        st.subheader("🎭 CryptoPunk Generator")
        st.markdown("Crée ton propre CryptoPunk en choisissant le type et les accessoires !")
        
        # Load model and metadata
        with st.spinner("Chargement du modèle..."):
            model, diffusion, metadata, device, epoch_info = load_cryptopunk_model(config_name)
        
        if model is None:
            st.error(f"❌ Erreur lors du chargement: {epoch_info}")
            st.stop()
        
        st.success(f"✅ Modèle chargé ({epoch_info})")
        
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Caractéristiques")
            
            # Type selection
            types = list(metadata['type_to_idx'].keys())
            selected_type = st.selectbox(
                "🧬 Type de Punk",
                types,
                help="Choisis le type de CryptoPunk"
            )
            type_idx = metadata['type_to_idx'][selected_type]
            
            # Accessory selection
            st.markdown("**🎭 Accessoires**")
            accessories = sorted(metadata['accessory_to_idx'].keys())
            
            selected_accessories = st.multiselect(
                "Sélectionne les accessoires (multi-choix)",
                accessories,
                help="Tu peux sélectionner plusieurs accessoires"
            )
            
            # Get indices of selected accessories
            accessory_indices = [metadata['accessory_to_idx'][acc] for acc in selected_accessories]
            
            # Display selection summary
            st.info(f"**Sélection:** {selected_type} avec {len(selected_accessories)} accessoire(s)")
            
            # Generate buttons
            col_a, col_b = st.columns(2)
            with col_a:
                generate_clicked = st.button("🎲 Générer", type="primary", use_container_width=True)
            with col_b:
                random_clicked = st.button("🎰 Aléatoire", use_container_width=True)
            
            # CFG Guidance Scale slider
            st.markdown("---")
            st.markdown("### ⚙️ Paramètres Avancés")
            cfg_scale = st.slider(
                "🎯 CFG Scale (Classifier-Free Guidance)",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                help="Contrôle la fidélité aux attributs. 1.0 = pas de guidance, 3.0-7.0 = typique"
            )
            st.caption(f"💡 CFG={cfg_scale:.1f}: {'Créatif' if cfg_scale < 2 else 'Equilibré' if cfg_scale < 5 else 'Très fidèle'}")
            
            if random_clicked:
                # Random type
                selected_type = np.random.choice(types)
                type_idx = metadata['type_to_idx'][selected_type]
                
                # Random accessories (2-4 accessories)
                num_accessories = np.random.randint(2, 5)
                selected_accessories = list(np.random.choice(accessories, size=num_accessories, replace=False))
                accessory_indices = [metadata['accessory_to_idx'][acc] for acc in selected_accessories]
                
                generate_clicked = True
        
        with col2:
            st.markdown("### Résultat")
            
            if generate_clicked:
                with st.spinner("Génération en cours..."):
                    try:
                        # Generate the punk with CFG
                        img = generate_cryptopunk(model, diffusion, device, type_idx, accessory_indices, cfg_scale=cfg_scale)
                        
                        # Display the image (enlarged)
                        st.image(img, caption=f"{selected_type} punk (CFG={cfg_scale:.1f})", use_container_width=True)
                        
                        # Display selected attributes
                        with st.expander("📋 Détails des attributs"):
                            st.markdown(f"**Type:** {selected_type}")
                            if selected_accessories:
                                st.markdown("**Accessoires:**")
                                for acc in selected_accessories:
                                    st.markdown(f"- {acc}")
                            else:
                                st.markdown("**Accessoires:** Aucun")
                        
                        # Download button
                        img_resized = img.resize((256, 256), Image.NEAREST)  # Pixel art scaling
                        
                        # Convert PIL image to bytes for download
                        buf = BytesIO()
                        img_resized.save(buf, format='PNG')
                        img_bytes = buf.getvalue()
                        
                        st.download_button(
                            label="💾 Télécharger",
                            data=img_bytes,
                            file_name=f"cryptopunk_{selected_type.lower()}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la génération: {e}")
            else:
                st.info("👆 Choisis les caractéristiques et clique sur 'Générer'")
                
                # Show placeholder
                placeholder = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                placeholder[:] = [45, 55, 72]  # Dark gray background
                st.image(placeholder, caption="Ton CryptoPunk apparaîtra ici", use_container_width=True)
    
    else:
        # ====================================================================
        # SIMPLE INFERENCE INTERFACE
        # ====================================================================
        st.subheader("🎲 Génération Simple")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Paramètres")
            
            num_samples = st.slider("Nombre d'images:", 1, 16, 4, key="num_samples_slider")
            
            # CFG scale (if model supports conditioning)
            cfg_scale = st.slider(
                "🎯 CFG Scale",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                help="Classifier-Free Guidance. 1.0 = pas de guidance, 3.0-7.0 = typique"
            )
            st.caption(f"💡 CFG={cfg_scale:.1f}: {'Créatif' if cfg_scale < 2 else 'Equilibré' if cfg_scale < 5 else 'Très fidèle'}")
            
            generate_button = st.button("🎨 Générer", use_container_width=True, type="primary")
        
        with col2:
            st.markdown("### 🖼️ Résultats")
            
            if generate_button:
                with st.spinner(f"Génération de {num_samples} image(s) en cours..."):
                    try:
                        images, epoch_info = generate_simple_images(config_name, num_samples, cfg_scale)
                        
                        if images is None:
                            st.error(f"❌ Erreur lors de la génération: {epoch_info}")
                        else:
                            st.success(f"✅ Génération terminée! (Modèle: {epoch_info})")
                            
                            # Display images in grid
                            cols_per_row = 4
                            for i in range(0, len(images), cols_per_row):
                                cols = st.columns(cols_per_row)
                                for j in range(cols_per_row):
                                    idx = i + j
                                    if idx < len(images):
                                        with cols[j]:
                                            st.image(images[idx], caption=f"Image {idx+1}", use_container_width=True)
                            
                            # Download option
                            st.markdown("---")
                            if len(images) == 1:
                                # Single image download
                                buf = BytesIO()
                                images[0].save(buf, format='PNG')
                                st.download_button(
                                    label="💾 Télécharger l'image",
                                    data=buf.getvalue(),
                                    file_name=f"generated_{config_name}.png",
                                    mime="image/png"
                                )
                    except Exception as e:
                        st.error(f"❌ Erreur: {e}")
                        import traceback
                        st.code(traceback.format_exc())
            else:
                st.info("👆 Choisis le nombre d'images et clique sur 'Générer'")
                
                # Show placeholder
                placeholder_cols = st.columns(4)
                for i, col in enumerate(placeholder_cols):
                    with col:
                        placeholder = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                        placeholder[:] = [45, 55, 72]  # Dark gray
                        st.image(placeholder, caption=f"Image {i+1}", use_container_width=True)
        
        # Show recent samples
        st.markdown("---")
        st.subheader("📷 Dernières Générations")
        
        sample_images = get_sample_images(dataset, config_name)
        
        if sample_images:
            cols = st.columns(min(4, len(sample_images)))
            for idx, img_path in enumerate(sample_images):
                with cols[idx % 4]:
                    img = Image.open(img_path)
                    epoch = img_path.stem.split('_')[-1]
                    st.image(img, caption=f"Epoch {epoch}", use_container_width=True)
        else:
            st.info("Aucune image générée pour l'instant.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
        <p>🔥 Diffusion Model Training Dashboard | Powered by Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
