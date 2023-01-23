import torch
from torch import nn
from torch.nn import functional as F
from albumentations.augmentations import functional as albu_f
import numpy as np
import random
import os
from configs import base_config, classifier_config, autoencoder_config
from dataset import CIFARDataset
from models import VAE
from sklearn.neighbors import NearestNeighbors
import cv2
import matplotlib
from matplotlib import cm, ticker
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def seed_everything(seed: int = base_config.random_seed) -> None:
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def visualize_embeddings(embeddings, labels, num_categories=classifier_config.num_classes):
    threshold = base_config.visualization_threshold
    if len(embeddings) > threshold:
        embeddings = embeddings[:threshold, :]
        labels = labels[:threshold]
    
    cmap = cm.get_cmap('tab20')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for label in range(num_categories):
        indexes = labels == label
        ax.scatter(embeddings[indexes, 0], embeddings[indexes, 1], embeddings[indexes, 2], c=np.array(cmap(label)).reshape(1, 4), label = label, alpha=0.5)
    
    plt.savefig(f'images/scaled_embeddings_{autoencoder_config.loss_function}.png')


def plot_3d(points: np.array, points_color: np.array, title=f"{autoencoder_config.loss_function} low-dimensional representation of embeddings"):
    '''
        points - low-dimensional representation of embeddings
        points_color - appropriate classes
    '''
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(10, 10),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()


class EmbeddingSearch:
    classes10 = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    class2idx = {classname: idx for idx, classname in enumerate(classes10)}
    idx2class = {idx: classname for idx, classname in enumerate(classes10)}
    '''
        wrapper for embeddings search

            encoder: PyTorch model to extract embeddings with
            embeddings - [N, E]: embeddings extracted from data
            labels - [N]: list of labels from data
            num_classes - int: number of groups to cluster data by
    '''
    def __init__(
        self,
        num_classes=classifier_config.num_classes) -> None:

        self.dataset = CIFARDataset(train=False)
        
        self.encoder, self.feature_extractor = self.download()
        embeddings, labels = self.download_embeddings()
        self.embeddings, self.labels = embeddings, labels
        self.nn = NearestNeighbors(n_neighbors=num_classes)
        self.nn.fit(embeddings)

    def download(self):
        model = VAE()
        model.to(classifier_config.device)
        model.load()
        model.eval()
        return model.reparameterize, model.encode
    
    def download_embeddings(self):
        embeddings = []
        labels = []
        with torch.no_grad():
            for image, label in self.dataset:
                image = torch.tensor(image, dtype=torch.float32, device=base_config.device_str).unsqueeze(0)
                mu, log_var = self.feature_extractor(image)
                emb = self.encoder(mu, log_var).detach().squeeze(0).numpy()
                embeddings.append(emb)
                labels.append(label)
        return embeddings, labels
    
    def embeddings_image_search(self, img, n_neighbors=5):
        img = cv2.resize(img, (base_config.image_height, base_config.image_width), cv2.INTER_LINEAR)
        img = albu_f.normalize(img, 0, 1, 255.0)
        img = torch.tensor(img, dtype=torch.float32, device=base_config.device_str)
        img = img.permute(2, 0, 1).unsqueeze(0)

        # extracting embeddings from the images
        with torch.no_grad():
            mu, log_var = self.feature_extractor(img)
            emb = self.encoder(mu, log_var)
            outputs = F.normalize(emb, p=2, dim=1)
            outputs = outputs.cpu().detach().numpy()
        
        # find closest matches
        neighbors = self.nn.kneighbors(outputs, n_neighbors + 1, return_distance=False)[0][1:]
        
        images, labels = [], []
        for x in neighbors:
            img, label = self.dataset[x]
            images.append(img.transpose(1, 2, 0))
            labels.append(label)
        
        return images, labels

    def evaluate_search(self, search_data):
        '''
            search_data should be organised as a dataset with (img_path, label) 
            as an example pair
        '''
        tp_3, fp_3 = 0, 0
        tp_5, fp_5 = 0, 0
        tp_10, fp_10 = 0, 0

        for img_path, label in search_data:
            img = cv2.imread(img_path)

            _, retrieved_labels = self.embeddings_image_search(img, n_neighbors=10)
            
            if label in retrieved_labels:
                tp_10 += 1
            else:
                fp_10 += 1

            if label in retrieved_labels[:5]:
                tp_5 += 1
            else:
                fp_5 += 1

            if label in retrieved_labels[:3]:
                tp_3 += 1
            else:
                fp_3 += 1
        
        recall_at_k = {
            'k3': tp_3 / (tp_3 + fp_3),
            'k5': tp_5 / (tp_5 + fp_5),
            'k10': tp_10 / (tp_10 + fp_10)
        }
        return recall_at_k

    def __call__(self, img, n_neighbors=10):
        return self.embeddings_image_search(img, n_neighbors)
