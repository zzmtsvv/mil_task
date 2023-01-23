from configs import autoencoder_config, classifier_config
from dataset import CIFARDataset
from models import VAE, Classifier
from blocks import CauchyLoss, WelschLoss, GemanMcClureLoss, AnotherSmoothL1Loss
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm


class AutoEncoderTrainer:
    losses = {
        "cauchy_loss": CauchyLoss,
        "welsch_loss": WelschLoss,
        "geman-mcclure_loss": GemanMcClureLoss,
        "smooth_l1_loss": AnotherSmoothL1Loss,
        "reconstruction_loss": None
    }

    def __init__(self, config = autoencoder_config) -> None:
        self.config = config
        self.converged = False

        self.model = VAE()
        self.model.to(config.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.num_epochs // 4)
        self.criterion = None

        if config.loss_function != "reconstruction_loss":
            self.criterion = self.losses[config.loss_function]()

        self.best_loss = float("inf")
        
        train_dataset = CIFARDataset(train=True)
        val_dataset = CIFARDataset(train=False)

        self.train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    def fit(self):
        print(f"Training starts on {autoencoder_config.device_str}")

        for epoch in range(1, autoencoder_config.num_epochs + 1):
            if self.converged:
                break

            self.train_epoch(epoch)
            self.validate_epoch(epoch)

        print("Training is done, best weights are saved")
        print(f"using {autoencoder_config.loss_function} with the amount of: {self.best_loss}")

    def train_epoch(self, epoch):
        self.model.train()

        pbar = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc=f"Epoch (train) {epoch}")
        
        running_loss = 0

        for i, (img, _) in pbar:
            img = img.to(autoencoder_config.device)

            self.optimizer.zero_grad()

            if self.criterion is None:
                loss, _, _ = self.model.loss_function(self.model.forward_prop(img))
            else:
                x_hat = self.model(img)
                loss = self.criterion(x_hat, img)
            
            running_loss += loss.item()

            pbar.set_postfix(
                dict(criterion_loss = round(running_loss / (i + 1), 5),
                ))
            pbar.update()
            
            if not torch.isnan(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), autoencoder_config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
            else:
                self.converged = True

    def validate_epoch(self, epoch):
        self.model.eval()

        pbar = tqdm(
            enumerate(self.val_dataloader),
            total=len(self.val_dataloader),
            desc=f"Epoch (test) {epoch}")
        
        running_loss = 0

        for i, (img, _) in pbar:
            img = img.to(autoencoder_config.device)

            with torch.inference_mode():
                if self.criterion is None:
                    loss, _, _ = self.model.loss_function(self.model.forward_prop(img))
                else:
                    x_hat = self.model(img)
                    loss = self.criterion(x_hat, img)

            running_loss += loss.item()

            pbar.set_postfix(
                dict(criterion_loss = round(running_loss / (i + 1), 5),
                ))
            pbar.update()
        
        if running_loss / (i + 1) < self.best_loss:
            self.best_loss = running_loss / (i + 1)
            self.save_model()

    def save_model(self):
        self.model.save()
        print(f"Saved model weights at: {autoencoder_config.weights_path}")
    
    def load_model(self, filename=autoencoder_config.weights_path):
        self.model.load(filename)


class ClassifierTrainer:
    def __init__(self, config = classifier_config) -> None:
        self.config = config
        self.encoder = self.download_encoder()  # returns mu and log variance as the latent representation
        self.converged = False
        self.best_loss = float("inf")
        self.best_accuracy = 0

        self.criterion = nn.CrossEntropyLoss()
        self.model = Classifier()
        self.model.to(config.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.num_epochs // 4)

        train_dataset = CIFARDataset(train=True)
        val_dataset = CIFARDataset(train=False)

        self.train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    def download_encoder(self):
        model = VAE()
        model.to(self.config.device)
        model.load()
        model.eval()
        return model.encode

    def fit(self):
        print(f"Training starts on {self.config.device_str}")

        for epoch in range(1, self.config.num_epochs + 1):
            if self.converged:
                break

            self.train_epoch(epoch)
            self.validate_epoch(epoch)

        print("\nTraining is done, best weights are saved")
        print(f"using CrossEntropy Loss with the amount of: {self.best_loss}")
        print(f"Accuracy: {self.best_accuracy}")

    @staticmethod
    def measure_accuracy(outputs: torch.Tensor, targets: torch.Tensor):
        labels = outputs.argmax(dim=1)
        num_correct = (targets == labels).sum() / len(labels)
        return num_correct
    
    def train_epoch(self, epoch):
        self.model.train()

        pbar = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc=f"Epoch (train) {epoch}")
        
        running_loss = 0

        for i, (img, label) in pbar:
            img, label = img.to(self.config.device), label.to(self.config.device)

            with torch.no_grad():
                mu, log_var = self.encoder(img)
            
            score = self.model(mu, log_var)
            loss = self.criterion(score, label)

            running_loss += loss.item()

            pbar.set_postfix(
                dict(criterion_loss = round(running_loss / (i + 1), 5),
                ))
            pbar.update()

            self.converged = True
            if not torch.isnan(loss):
                self.converged = False

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()

    def validate_epoch(self, epoch):
        self.model.eval()

        pbar = tqdm(
            enumerate(self.val_dataloader),
            total=len(self.val_dataloader),
            desc=f"Epoch (test) {epoch}")
        
        running_loss = 0
        running_accuracy = 0

        for i, (img, label) in pbar:
            img, label = img.to(self.config.device), label.to(self.config.device)

            with torch.inference_mode():
                mu, log_var = self.encoder(img)
                scores = self.model(mu, log_var)
                loss = self.criterion(scores, label)
                accuracy = self.measure_accuracy(scores, label)
            
            running_loss += loss.item()
            running_accuracy += accuracy.item()

            pbar.set_postfix(
                dict(
                    accuracy = round(running_accuracy / (i + 1), 5),
                    loss = round(running_loss / (i + 1), 5)))
            pbar.update()
        
        if running_loss / (i + 1) <= self.best_loss and running_accuracy / (i + 1) < self.best_accuracy:
            self.best_accuracy = running_loss / (i + 1)
            self.best_accuracy = running_accuracy / (i + 1)
            self.save_model()

    def save_model(self):
        self.model.save()
        print(f"Saved model weights at: {autoencoder_config.weights_path}")
    
    def load_model(self, filename=autoencoder_config.weights_path):
        self.model.load(filename)
