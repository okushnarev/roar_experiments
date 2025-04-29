# train_baseline.py
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from model import SimpleCNN
import numpy as np
import os

if __name__ == '__main__':

    # --- Configuration ---
    EPOCHS = 15
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    MODEL_SAVE_PATH = './cifar_baseline.pth'
    DATA_DIR = './data'

    # --- Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading and Preprocessing ---
    print("Loading CIFAR-10 data...")
    # Note: ToTensor() converts PIL image [0, 255] to tensor [0.0, 1.0]
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # CIFAR-10 specific means/stds
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    print("Data loaded.")

    # --- Model, Loss, Optimizer ---
    print("Initializing model...")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("Model initialized.")

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training.')

    # --- Save the Trained Model ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'Model saved to {MODEL_SAVE_PATH}')

    # --- Evaluate on Test Set ---
    print("Evaluating on test set...")
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No need to track gradients during evaluation
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')

    # --- Calculate and Save Training Set Mean (for pixel replacement later) ---
    print("Calculating training set mean pixel value...")
    # Reload training set *without* normalization to get original pixel values [0,1]
    mean_calc_transform = transforms.Compose([transforms.ToTensor()])
    mean_calc_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                                                 download=False, transform=mean_calc_transform)
    # Use a smaller subset for faster mean calculation if needed, but full is better
    mean_calc_loader = torch.utils.data.DataLoader(mean_calc_set, batch_size=1000,
                                                   shuffle=False, num_workers=2)
    mean = torch.zeros(3)
    n_samples = 0.0
    for data, _ in mean_calc_loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        n_samples += batch_samples

    mean /= n_samples
    print(f"Calculated Mean (R, G, B): {mean.tolist()}")
    # Save the mean for later use
    np.save(os.path.join(DATA_DIR, 'cifar10_mean.npy'), mean.numpy())
    print(f"Mean saved to {os.path.join(DATA_DIR, 'cifar10_mean.npy')}")
