import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)
    
    def forward(self, x: torch.Tensor):
        '''
        x - latents
        '''
        x = x.permute(0, 2, 3, 1).contiguous() # [B x D x H x W] -> [B x H x W x D]
        latents_shape = x.shape

        flattened_latents = x.view(-1, self.D)

        # L2 distance between latents and embeddings
        dist = torch.sum(flattened_latents ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight ** 2, dim=1)
        dist = dist - 2 * torch.matmul(flattened_latents, self.embedding.weight.t())  # [BHW, K]

        # discretization bottleneck
        encoding_indexes = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # one-hot encoding
        device = x.device
        encoded = torch.zeros(encoding_indexes.size(0), self.K, device=device).scatter_(1, encoding_indexes, 1)  # [BHW, K]

        # quantize the latents
        quantized_latents = torch.matmul(encoded, self.embedding.weight).view(latents_shape)  # [B x H x W x D]

        # VQ losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), x)
        emb_loss = F.mse_loss(quantized_latents, x.detach())
        # stopgradient operation is equivalent to detaching the tensor from the current computational graph
        # (considered as a constant, do not requires the gradient)
        vq_loss = commitment_loss + self.beta * emb_loss

        # residuals back to quantized part
        quantized_latents = x + (quantized_latents - x).detach()

        mean_probs = torch.mean(encoded, dim=0)
        perplexity = torch.exp(-torch.sum(mean_probs * torch.log(mean_probs + 1e-10)))

        # [B x D x H x W]
        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss, perplexity, encoded
   
        
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()

        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        )
    
    def forward(self, x):
        return x + self.resblock(x)


class F1:
    def __init__(self, epsilon=1e-7) -> None:
        self.epsilon = epsilon
    
    @torch.no_grad()
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # convert tensors into 1d np arrays
        return f1_score(y_true, y_pred, average='weighted')


class BaseRobustLoss(nn.modules.loss._Loss):
    def __init__(self, c=0.1, reduction='mean') -> None:
        super().__init__()

        self.c2 = c * c
        self.reduction_ = reduction
    
    def robust_loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        '''
            tensors are a shape of [batch_size, channels, height, width]
        '''
        x = prediction - target
        x = x.norm(dim=(1, 2, 3), p=2).pow(2) / self.c2

        loss = self.robust_loss_fn(x)

        if self.reduction_ == 'mean':
            loss = loss.mean(dim=0)
        if self.reduction_ == 'sum':
            loss = loss.sum(dim=0)
        
        return loss


class CauchyLoss(BaseRobustLoss):
    def __init__(self, c=0.1, reduction='mean') -> None:
        super().__init__(c=c, reduction=reduction)
    
    def robust_loss_fn(self, x: torch.tensor) -> torch.Tensor:
        return torch.log1p(x)
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        return super().forward(prediction, target)


class GemanMcClureLoss(BaseRobustLoss):
    def __init__(self, c=0.1, reduction='mean') -> None:
        super().__init__(c, reduction)
    
    def robust_loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * x / (x + 4) 

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        return super().forward(prediction, target)


class WelschLoss(BaseRobustLoss):
    def __init__(self, c=0.1, reduction='mean') -> None:
        super().__init__(c, reduction)
    
    def robust_loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        return 1 - torch.exp(-x / 2)
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        return super().forward(prediction, target)


class AnotherSmoothL1Loss(BaseRobustLoss):
    def __init__(self, c=0.1, reduction='mean') -> None:
        super().__init__(c, reduction)
    
    def robust_loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x + 1) - 1
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        return super().forward(prediction, target)


class FocalLoss(nn.modules.loss._Loss):
    '''
        Focal loss for multi-class problem
    '''
    def __init__(self, alpha=0.5, gamma=2, ignore_index=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, label_inp: torch.Tensor, label_tgt: torch.Tensor):
        num_classes = label_inp.size(1)
        loss = 0

        if self.ignore_index is not None:
            not_ignored = label_tgt != self.ignore_index
        
        for class_ in range(num_classes):
            class_label_tgt = (label_tgt == class_).long()
            class_label_inp = label_inp[:, class_, ...]

            if self.ignore_index is not None:
                class_label_tgt = class_label_tgt[not_ignored]
                class_label_inp = class_label_inp[not_ignored]
            
            loss += self.sigmoid_fl(class_label_inp, class_label_tgt)
        return loss
    
    def sigmoid_fl(self, inp: torch.Tensor, tgt: torch.Tensor, reduction='mean'):
        tgt = tgt.type(inp.type())

        log_pt = -F.binary_cross_entropy_with_logits(inp, tgt, reduction='none')
        pt = torch.exp(log_pt)

        loss = -((1 - pt).pow(self.gamma)) * log_pt

        if self.alpha is not None:
            loss = loss * (self.alpha * tgt + (1 - self.alpha) * (1 - tgt))
        
        if reduction == 'mean':
            loss = loss.mean()
        if reduction == 'sum':
            loss = loss.sum()
        if reduction == 'batchwise_mean':
            loss = loss.sum(0)
        return loss


class HingeLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true):
        loss = 1 - torch.mul(y_pred, y_true)
        loss[loss < 0] = 0
        return loss
