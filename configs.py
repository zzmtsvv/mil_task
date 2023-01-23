from dataclasses import dataclass
import torch


@dataclass
class base_config:
    random_seed: int = 42
    device_str: str = 'cuda' if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    visualization_threshold = 4000
    image_width = 32
    image_height = 32
    data_directory = './data'


@dataclass
class autoencoder_config(base_config):
    loss_function = "cauchy_loss"  # possible variants: [cauchy_loss, welsch_loss, geman-mcclure_loss, smooth_l1_loss, reconstruction_loss]
    in_channels = 3
    embedding_dim = 128
    num_embeddings = 16
    hidden_dims = [1, 2]
    beta = 0.25
    num_epochs = 50
    batch_size = 16
    lr = 3e-4
    max_grad_norm = 1.0
    weight_decay = 1e-3
    num_workers = 2
    weights_path = f'autoencoder_weights/{loss_function}_best_autoencoder.pt'
    kld_weight = 0.00025


@dataclass
class classifier_config(base_config):
    ae_loss_fn = autoencoder_config.loss_function
    num_epochs = 100
    batch_size = 16
    lr = 3e-4
    num_classes = 10
    max_grad_norm = 1.0
    weight_decay = 1e-3
    num_workers = 2
    weights_path = f'{autoencoder_config.loss_function}_best_classifier.pt'
