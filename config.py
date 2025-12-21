import torch


class Config:
    def __init__(
        self,
        dataset_name="MNIST",
        epochs=50,
        lr=3e-4,
        T=500,
        batch_size=128,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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


config = Config(dataset_name="MNIST")