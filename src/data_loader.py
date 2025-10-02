import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_transforms():
    """
    Get CIFAR-10 transforms with proper normalization and data augmentation.
    Following assignment requirements for Question 1(a).
    """
    # CIFAR-10 statistics
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2616)
    
    train_transform = transforms.Compose([
        # Data augmentation
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        
        # Convert to tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        
        # Random erasing for additional regularization
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    
    return train_transform, test_transform

def get_cifar10_dataloaders(batch_size=128, num_workers=4):
    """
    Create CIFAR-10 data loaders for training, validation, and testing.
    """
    train_transform, test_transform = get_cifar10_transforms()
    
    # Download and load training data
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Split training data into train and validation sets
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Apply test transform to validation set
    val_dataset.dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=False,
        transform=test_transform
    )
    val_dataset.indices = val_dataset.indices
    
    # Load test data
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
