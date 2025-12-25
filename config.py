import torch
import os

# Force l'utilisation de CUDA même si l'architecture n'est pas officiellement supportée
# Pour RTX 5070 (sm_120 / Blackwell)
os.environ['TORCH_CUDA_ARCH_LIST'] = '5.0;6.0;6.1;7.0;7.5;8.0;8.6;9.0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Config:
    def __init__(
        self,
        dataset_name="MNIST",
        epochs=50,
        lr=3e-4,
        T=500,
        batch_size=128,
        beta_start=1e-4,
        beta_end=0.02,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.dataset_name = dataset_name

        if dataset_name == "MNIST":
            self.img_size = 16
            self.image_channels = 1
        elif dataset_name == "CRYPTOPUNKS":
            self.img_size = 32
            self.image_channels = 3

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Paths
        self.dataset_path = f"./data/{self.dataset_name}"
        self.models_path = f"./models/{self.dataset_name}"
        self.results_path = f"./results/{self.dataset_name}"

        # Model specific
        if dataset_name == "MNIST":
            self.time_emb_dim = 32
            self.start_channels = 64
        elif dataset_name == "CRYPTOPUNKS":
            self.time_emb_dim = 64
            self.start_channels = 128


# Config 1: Standard DDPM (Baseline)
# T=1000 is standard for good quality. Beta schedule 1e-4 to 0.02 is classic.
config1_mnist = {
    "dataset_name": "MNIST",
    "epochs": 100,
    "lr": 3e-4,
    "T": 1000,
    "batch_size": 128,
    "beta_start": 1e-4,
    "beta_end": 0.02,
}

# Config 2: Fast Prototyping (Faster training/sampling)
# T=300 is much faster but might be slightly noisier/less detailed.
# Useful for quick debugging.
config2_mnist = {
    "dataset_name": "MNIST",
    "epochs": 100,
    "lr": 3e-4,
    "T": 300,
    "batch_size": 128,
    "beta_start": 1e-4,
    "beta_end": 0.02,
}

# Config 3: High Precision / Smooth Schedule
# Lower beta_end (0.01) creates a "smoother" noise schedule,
# sometimes resulting in higher quality but might need more steps to converge perfectly.
# Config 3: High Precision / Smooth Schedule
# Lower beta_end (0.01) creates a "smoother" noise schedule,
# sometimes resulting in higher quality but might be slightly noisier/less detailed.
config3_mnist = {
    "dataset_name": "MNIST",
    "epochs": 100,
    "lr": 2e-4,  # Lower LR for more stable convergence
    "T": 1000,
    "batch_size": 128,
    "beta_start": 1e-4,
    "beta_end": 0.01,  # Slower noise accumulation
}


# Config 1: Standard DDPM (Baseline)
# T=1000 is standard for good quality. Beta schedule 1e-4 to 0.02 is classic.
config1_cryptopunks = {
    "dataset_name": "CRYPTOPUNKS",
    "epochs": 100,
    "lr": 3e-4,
    "T": 1000,
    "batch_size": 64,
    "beta_start": 1e-4,
    "beta_end": 0.02,
}

# Config 2: Fast Prototyping (Faster training/sampling)
# T=300 is much faster but might be slightly noisier/less detailed.
# Useful for quick debugging.
config2_cryptopunks = {
    "dataset_name": "CRYPTOPUNKS",
    "epochs": 100,
    "lr": 3e-4,
    "T": 300,
    "batch_size": 64,
    "beta_start": 1e-4,
    "beta_end": 0.02,
}

# Config 3: High Precision / Smooth Schedule
# Lower beta_end (0.01) creates a "smoother" noise schedule,
# sometimes resulting in higher quality but might need more steps to converge perfectly.
config3_cryptopunks = {
    "dataset_name": "CRYPTOPUNKS",
    "epochs": 100,
    "lr": 2e-4,  # Lower LR for more stable convergence
    "T": 1000,
    "batch_size": 64,
    "beta_start": 1e-4,
    "beta_end": 0.01,  # Slower noise accumulation
}

CONFIGS = {
    "config1": config1_mnist,
    "config2": config2_mnist,
    "config3": config3_mnist,
    "config1_cryptopunks": config1_cryptopunks,
    "config2_cryptopunks": config2_cryptopunks,
    "config3_cryptopunks": config3_cryptopunks,
}


def get_config_by_name(name):
    if name not in CONFIGS:
        raise ValueError(
            f"Config {name} not found. Available configs: {list(CONFIGS.keys())}"
        )
    return Config(**CONFIGS[name])


config = Config(**config1_mnist)
