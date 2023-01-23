import torch
from torch import nn
from torch.nn import functional as F
from blocks import VectorQuantizer, Residual
from configs import autoencoder_config, classifier_config
import os


class Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # model should forward unnormalized logits

        self.net = nn.Sequential(
            nn.Linear(autoencoder_config.embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, classifier_config.num_classes)
        )
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(1 / 2 * log_var)
        eps = torch.rand_like(std)
        return eps * std + mu

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor):
        x = self.reparameterize(mu, log_var)
        return self.net(x)

    def save(self):
        filename = classifier_config.weights_path
        if not os.path.isdir("classifier_weights"):
            os.makedirs("classifier_weights")

        torch.save(self.state_dict(), filename)
    
    def load(self, filename=classifier_config.weights_path):
        state_dict = torch.load(filename, map_location=classifier_config.device)
        self.load_state_dict(state_dict)
    
    def get_model_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VQVAE(nn.Module):
    '''
    https://arxiv.org/abs/1711.00937
    '''
    def __init__(
        self,
        in_channels=autoencoder_config.in_channels,
        embedding_dim=autoencoder_config.embedding_dim,
        num_embeddings=autoencoder_config.num_embeddings,
        hidden_dims=autoencoder_config.hidden_dims,
        beta=autoencoder_config.beta):
        super(VQVAE, self).__init__()

        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.hidden_dims = hidden_dims
        
        # encoder
        modules = self.build_encoder(self.hidden_dims)
        self.encoder = nn.Sequential(*modules)
        self.vq_layer = VectorQuantizer(self.num_embeddings, self.embedding_dim, self.beta)

        self.hidden_dims.reverse()

        # decoder
        modules = self.build_decoder(self.hidden_dims)
        self.decoder = nn.Sequential(*modules)
        
    def build_encoder(self, hidden_dims):
        modules = []
        in_channels = self.in_channels

        for dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, dim, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU()
                )
            )
            in_channels = dim
        
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU()
            )
        )

        for _ in range(6):
            modules.append(Residual(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, self.embedding_dim, kernel_size=1, stride=1),
            nn.LeakyReLU()
        ))

        return modules
    
    def build_decoder(self, hidden_dims):
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(self.embedding_dim, hidden_dims[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        ))

        for _ in range(6):
            modules.append(Residual(hidden_dims[0], hidden_dims[0]))
        modules.append(nn.LeakyReLU())

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU()
                )
            )
        modules.append(nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], out_channels=autoencoder_config.in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        ))

        return modules
    
    def encode(self, x: torch.Tensor):
        '''
        :param x: input tensor to encoder [N x C x H x W]
        :return: list of latent codes
        '''
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor):
        '''
        maps latent codes onto the image space
        :param z: Tensor [B x D x H x W]
        :return: Tensor [B x C x H x W]
        '''
        return self.decoder(z)
    
    @torch.no_grad()
    def generate(self, x):
        '''
        returns the reconstructed image [B x C x H x W]
        '''
        return self.forward_prop(x)[0]
    
    def forward_prop(self, x: torch.Tensor):
        encoded = self.encoder(x)
        quantized, vq_loss, perplexity, encoded = self.vq_layer(encoded)
        return [self.decode(quantized), x, vq_loss, perplexity, encoded]
    
    def forward(self, x: torch.Tensor):
        #return self.forward_prop(x)[0]
        return self.forward_prop(x)
    
    def loss_function(self, *args):
        '''
        see self.forward_prop() for details of args
        '''
        reconstructed, x, vq_loss, perplexity, _ = args

        reconstruction_loss = F.mse_loss(reconstructed, x)
        loss = reconstruction_loss + vq_loss
        return loss, reconstruction_loss, perplexity
    
    def sample(self, num_samples, device=None):
        '''
        samples from the latent space and maps to the image space
        '''
        if device is None:
            device = autoencoder_config.device

        scale_factor = 2 ** len(self.hidden_dims)

        #z = torch.rand(num_samples, 1, self.vq_layer.K, self.vq_layer.D)
        z = torch.rand(num_samples, self.vq_layer.D, autoencoder_config.image_height // scale_factor, autoencoder_config.image_width // scale_factor)
        z.to(device)

        quantized_latents, _, _, _ = self.vq_layer(z)
        return self.decode(quantized_latents)
    
    def save(self):
        filename = autoencoder_config.weights_path
        if not os.path.isdir("autoencoder_weights"):
            os.makedirs("autoencoder_weights")

        torch.save(self.state_dict(), filename)
    
    def load(self, filename=autoencoder_config.weights_path):
        state_dict = torch.load(filename, map_location=autoencoder_config.device)
        self.load_state_dict(state_dict)
    
    def get_model_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VAE(nn.Module):
    '''
    https://arxiv.org/abs/1312.6114
    '''
    def __init__(
        self,
        in_channels=autoencoder_config.in_channels,
        latent_dim=autoencoder_config.embedding_dim,
        hidden_dims=None):

        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [2 ** i for i in range(5, 10)]
        
        modules = self.build_encoder(hidden_dims)
        self.encoder = nn.Sequential(*modules)
        self.fc_mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_variance = nn.Linear(hidden_dims[-1], latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])
        hidden_dims.reverse()

        modules = self.build_decoder(hidden_dims)
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def build_encoder(self, hidden_dims):
        modules = []
        in_channels = self.in_channels

        for dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.LeakyReLU())
            )
            in_channels = dim

        return modules
    
    def build_decoder(self, hidden_dims):
        modules = []

        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims[i + 1]),
                nn.LeakyReLU()
            ))
        
        return modules

    def generate(self, x):
        '''
        returns the reconstructed image
        :param x: tensor [B x C x H x W]
        :return: tensor [B x C x H x W]
        '''
        return self.forward(x)[0]
    
    def sample(self, num_samples, device):
        '''
        Samples from the latent space and maps to image space map
        '''
        z = torch.randn(num_samples, self.latent_dim)
        z.to(device)

        return self.decode(z)
    
    def forward_prop(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var]
    
    def forward(self, x):
        return self.forward_prop(x)[0]
    
    def reparameterize(self, mu, log_var):
        '''
        Reparameterization trick to sample N(mu, var) from N(0, 1)
        :param mu: Mean of the latent Gaussian [B x D]
        :param log_var: standard deviation of the latent Gaussian [B x D]
        '''
        std = torch.exp(1 / 2 * log_var)
        eps = torch.rand_like(std)
        return eps * std + mu
    
    def decode(self, z):
        '''
        reconstruct the image given the latent embedding
        :param z: latent codes [B x D]
        :return: tensor [B x C x H x W]
        '''
        res = self.decoder_input(z).view(-1, 512, 1, 1)
        res = self.decoder(res)
        return self.final_layer(res)
        

    def encode(self, x: torch.Tensor):
        '''
        Encodes the input by passing through the encoder network and returns latent codes
        :param x: (Tensor) input tensor to encoder [N x C x H x W]
        :return: list of parameters of latent Gaussian distribution
        '''
        res = torch.flatten(self.encoder(x), start_dim=1)

        mu = self.fc_mean(res)
        log_variance = self.fc_variance(res)

        return [mu, log_variance]
    
    def loss_function(self, args, weight=autoencoder_config.kld_weight):
        '''
        Computes the VAE loss function (KL divergence)
        '''
        
        # see self.forward_prop() for details of unpacking the args
        x_hat, x, mu, log_var = args

        # KL divergence weight
        if weight is None:
            weight = 1
        
        reconstruction_loss = F.mse_loss(x_hat, x)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - torch.square(mu) - log_var.exp(), dim=1), dim=0)
        loss = reconstruction_loss + weight * kld_loss

        return loss, reconstruction_loss, kld_loss
    
    def save(self):
        filename = autoencoder_config.weights_path
        if not os.path.isdir("autoencoder_weights"):
            os.makedirs("autoencoder_weights")

        torch.save(self.state_dict(), filename)
    
    def load(self, filename=autoencoder_config.weights_path):
        state_dict = torch.load(filename, map_location=autoencoder_config.device)
        self.load_state_dict(state_dict)
    
    def get_model_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

