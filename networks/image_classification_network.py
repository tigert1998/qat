from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .evaluation import evaluate_classification


class ImageClassificationNetwork(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    @staticmethod
    def train_step(model, data_batch, optimizer):
        optimizer.zero_grad()
        preds = model.forward(data_batch[0].to(model.device))
        correct = torch.sum(
            torch.argmax(preds, dim=1) == data_batch[1].to(model.device)
        ).cpu().detach().numpy()
        num_samples = preds.size(0)
        loss = F.cross_entropy(preds, data_batch[1].to(model.device))
        loss.backward()
        optimizer.step()
        return {
            "log_vars": {
                "loss": loss.cpu().detach().numpy(),
                "accuracy": correct / num_samples,
            },
            "num_samples": num_samples
        }

    @staticmethod
    def val_step(model, data_batch, optimizer):
        preds = model.forward(data_batch[0].to(model.device))
        return (preds, data_batch[1])

    @staticmethod
    def get_input_tuple(model, data_batch):
        return (data_batch[0].to(model.device),)

    @staticmethod
    def evaluate(preds: List[torch.Tensor], targets: List[torch.Tensor]) -> dict:
        preds = torch.cat(preds, 0)
        targets = torch.cat(targets, 0)
        return evaluate_classification(preds.cpu().numpy(), targets.cpu().numpy())
