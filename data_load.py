import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import struct
import numpy as np

# Define transformations for the datasets
# Define transformations for the standard MNIST dataset (28x28)
# transform_mnist_standard = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# transform_cifar10 = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

transform_mnist_standard = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

transform_cifar100 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

transform_imagenet_tiny = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_imagenet = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define a function to load the datasets
def get_dataset(dataset_name):
    train_dataset = None
    test_dataset = None
    num_classes = None
    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST(root="./data", train=True, transform=transform_mnist_standard, download=True)
        test_dataset = datasets.MNIST(root="./data", train=False, transform=transform_mnist_standard)
        num_classes = 10
    elif dataset_name == "CIFAR-10":
        train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform_cifar10, download=True)
        test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform_cifar10)
        num_classes = 10
    elif dataset_name == "CIFAR-100":
        train_dataset = datasets.CIFAR100(root="./data", train=True, transform=transform_cifar100, download=True)
        test_dataset = datasets.CIFAR100(root="./data", train=False, transform=transform_cifar100)
        num_classes = 100
    elif dataset_name == "ImageNet-Tiny":
        train_dataset = datasets.ImageFolder(root="./data/tiny-imagenet-200/train", transform=transform_imagenet_tiny)
        test_dataset = datasets.ImageFolder(root="./data/tiny-imagenet-200/val", transform=transform_imagenet_tiny)
        num_classes = 200
    elif dataset_name == "ImageNet":
        train_dataset = datasets.ImageFolder(root="./data/imagenet/train", transform=transform_imagenet)
        test_dataset = datasets.ImageFolder(root="./data/imagenet/val", transform=transform_imagenet)
        num_classes = 1000
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_dataset, test_dataset, num_classes

def load_dataset(dataset_name):
    train_dataset, test_dataset, num_classes = get_dataset(dataset_name) 
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader, num_classes



# Function to export dataset
def export_dataset(dataset, images_filename, labels_filename):
    with open(images_filename, 'wb') as img_file, open(labels_filename, 'wb') as lbl_file:
        for image, label in dataset:
            # Convert image to numpy array, ensure it's float32, and flatten
            img_data = image.numpy().astype(np.float32).flatten()

            # Write the flattened image data to the binary file
            img_file.write(struct.pack('f' * len(img_data), *img_data))

            # Ensure label is an int and write to the binary file
            lbl_data = np.array(label).astype(np.uint32)
            lbl_file.write(struct.pack('i', lbl_data))


def export_quantized_dataset(model, dataset, images_filename, labels_filename):
    with open(images_filename, 'wb') as img_file, open(labels_filename, 'wb') as lbl_file:
        for image, label in dataset:
            #Convert image to numpy array, ensure it's float32, and flatten TO DO FIX HERER
            image = model.quant_x(image).int_repr()
            img_data = image.numpy().astype(np.int32).flatten()

            # Write the flattened image data to the binary file
            img_file.write(struct.pack('i' * len(img_data), *img_data))

            # Ensure label is an int and write to the binary file
            #lbl_data = np.array(label).astype(np.uint32)
            #lbl_file.write(struct.pack('I', lbl_data))
            # Handle different types of labels
            if isinstance(label, torch.Tensor):
                label = label.numpy()  # Convert to numpy if it's a tensor
            if label.ndim == 0:
                label = np.expand_dims(label, axis=0)  # Ensure label is at least 1D
            lbl_data = label.astype(np.uint32)

            # Write each label element separately
            for lbl in lbl_data:
                lbl_file.write(struct.pack('I', lbl))


