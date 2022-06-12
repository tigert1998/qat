import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DistributedSampler, DataLoader


def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_test_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_dist_train_data_loader(rank, world_size, batch_size, root, use_image_folder=True):
    if use_image_folder:
        train_ds = torchvision.datasets.ImageFolder(
            root=f"{root}/train",
            transform=get_train_transform()
        )
    else:
        train_ds = torchvision.datasets.ImageNet(
            root=root, split='train',
            transform=get_train_transform()
        )
    return DataLoader(
        train_ds, batch_size, num_workers=16,
        sampler=DistributedSampler(train_ds, world_size, rank, shuffle=True)
    )


def get_dist_test_data_loader(rank, world_size, batch_size, root, use_image_folder=True):
    if use_image_folder:
        test_ds = torchvision.datasets.ImageFolder(
            root=f'{root}/val',
            transform=get_test_transform()
        )
    else:
        test_ds = torchvision.datasets.ImageNet(
            root=root, split='val',
            transform=get_test_transform()
        )
    return DataLoader(
        test_ds, batch_size, num_workers=16,
        sampler=DistributedSampler(test_ds, world_size, rank, shuffle=False)
    )
