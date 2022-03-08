import argparse

import torch

from networks.resnet import resnet18
from networks.cnn_wrapper import CNNWrapper
from ops import QuantizedOperator
from runner.train import dist_train
from runner.hooks import TensorboardLoggerHook, CheckpointHook, Hook


class SwitchQuantizationModeHook(Hook):
    def after_train_iter(self, runner):
        if (runner.iter + 1) != 50000:
            return
        runner.logger.info("switching to activation quantization")
        for module in runner.model.modules():
            if isinstance(module, QuantizedOperator):
                module.activation_quantization = True


def dist_train_build(rank, world_size, device_id, num_epochs, vars):
    from data.imagenet import get_dist_train_data_loader, get_dist_test_data_loader

    device = f"cuda:{device_id}"
    model = CNNWrapper(resnet18(), device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=vars["lr"], momentum=0.9, weight_decay=1e-4
    )
    train_data_loader = get_dist_train_data_loader(
        rank, world_size, vars["imgs_per_gpu"], vars["root"]
    )
    test_data_loader = get_dist_test_data_loader(
        rank, world_size, vars["imgs_per_gpu"], vars["root"]
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30 * len(train_data_loader), gamma=0.1
    )
    return model, optimizer, lr_scheduler, train_data_loader, test_data_loader


def get_device_id(rank, world_size, vars):
    device_ids = vars["device_ids"]
    return device_ids[rank % len(device_ids)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Quantized ResNet")
    parser.add_argument("--imgs-per-gpu", default=32, type=int)
    parser.add_argument("--root")
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--num-epochs", default=200, type=int)
    parser.add_argument("--work-dir")
    parser.add_argument("--device-ids", default="0,1,2,3")
    parser.add_argument("--num-procs", default=4, type=int)

    args = parser.parse_args()

    dist_train(
        args.num_procs, args.work_dir, args.num_epochs,
        get_device_id, dist_train_build, vars={
            "imgs_per_gpu": args.imgs_per_gpu,
            "root": args.root,
            "lr": args.lr,
            "device_ids": list(map(int, args.device_ids.split(',')))
        },
        hooks=[
            SwitchQuantizationModeHook(),
            TensorboardLoggerHook(1),
            CheckpointHook()
        ]
    )
