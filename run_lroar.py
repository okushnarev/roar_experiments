# run_lroar.py
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import json # To save results

# Import our custom modules
from model import SimpleCNN
from modify_dataset import ModifiedCIFAR10Dataset


# --- Helper Functions (Identical to run_roar.py) ---
def train_model(model, trainloader, criterion, optimizer, epochs, description="Fine-tuning"):
    """Fine-tunes a model for a given number of epochs."""
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        start_time = time.time()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_time = time.time() - start_time
        print(f'  Epoch {epoch + 1}/{epochs} [{description}] - Loss: {running_loss / len(trainloader):.4f} - Time: {epoch_time:.2f}s')

def evaluate_model(model, testloader):
    """Evaluates the model on the test set and returns accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

if __name__ == '__main__':
    # --- Configuration ---
    SALIENCY_METHODS = ['random', 'grad']  # Methods to evaluate
    REMOVAL_PERCENTAGES = [0.25, 0.50, 0.75]  # Percentages of pixels to remove

    BASELINE_MODEL_PATH = './cifar_baseline.pth'
    DATA_DIR = './data'
    MEAN_PATH = os.path.join(DATA_DIR, 'cifar10_mean.npy')
    RESULTS_FILE = './lroar_results.json'  # Different results file

    # L-ROAR Fine-tuning Hyperparameters
    LROAR_EPOCHS = 5  # Typically fewer epochs for fine-tuning
    LROAR_BATCH_SIZE = 64
    LROAR_LEARNING_RATE = 0.0001  # Often lower LR for fine-tuning

    # CIFAR-10 Stats (ensure these match those used in baseline training)
    CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR_STD = [0.2023, 0.1994, 0.2010]

    # --- Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Resources ---
    print("Loading resources...")
    # Load dataset mean
    if not os.path.exists(MEAN_PATH):
        raise FileNotFoundError(f"Dataset mean file not found at {MEAN_PATH}. Run train_baseline.py first.")
    dataset_mean = np.load(MEAN_PATH)

    # Load the baseline model state dict (needed for ModifiedCIFAR10Dataset AND for fine-tuning start point)
    if not os.path.exists(BASELINE_MODEL_PATH):
        raise FileNotFoundError(f"Baseline model not found at {BASELINE_MODEL_PATH}. Run train_baseline.py first.")

    # Instantiate a model structure (run on CPU for dataset creation)
    baseline_model_for_saliency = SimpleCNN()
    baseline_model_for_saliency.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location='cpu'))
    baseline_model_for_saliency.eval()

    # Load original datasets (needed for the wrapper)
    raw_transform = transforms.ToTensor() # Just convert to tensor [0,1]
    raw_trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=raw_transform)
    raw_testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=raw_transform)
    print("Resources loaded.")

    # --- L-ROAR Experiment Loop ---
    results = {}
    total_experiments = len(SALIENCY_METHODS) * len(REMOVAL_PERCENTAGES)
    experiment_count = 0

    print("\n--- Starting L-ROAR Experiments (Fine-tuning) ---")

    for method in SALIENCY_METHODS:
        results[method] = {}
        for percentage in REMOVAL_PERCENTAGES:
            experiment_count += 1
            print(f"\nRunning L-ROAR Experiment {experiment_count}/{total_experiments}: Method='{method}', Removal={percentage*100:.0f}%")
            start_time_exp = time.time()

            # 1. Create Modified Datasets (Same as ROAR)
            print("  Creating modified datasets...")
            dataset_device = 'cpu' # Keep dataset creation on CPU
            modified_train_dataset = ModifiedCIFAR10Dataset(
                original_dataset=raw_trainset,
                model=baseline_model_for_saliency, # Use baseline model for saliency maps
                saliency_method_name=method,
                removal_percentage=percentage,
                dataset_mean=CIFAR_MEAN,
                dataset_std=CIFAR_STD,
                device=dataset_device
            )
            modified_test_dataset = ModifiedCIFAR10Dataset(
                original_dataset=raw_testset,
                model=baseline_model_for_saliency, # Use baseline model for saliency maps
                saliency_method_name=method,
                removal_percentage=percentage,
                dataset_mean=CIFAR_MEAN,
                dataset_std=CIFAR_STD,
                device=dataset_device
            )

            # 2. Create DataLoaders (Same as ROAR)
            trainloader = DataLoader(modified_train_dataset, batch_size=LROAR_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True if device=='cuda' else False)
            testloader = DataLoader(modified_test_dataset, batch_size=LROAR_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True if device=='cuda' else False)
            print(f"  Datasets created. Train size: {len(modified_train_dataset)}, Test size: {len(modified_test_dataset)}")

            # 3. Load PRE-TRAINED Baseline Model for Fine-tuning **** KEY DIFFERENCE ****
            print("  Loading pre-trained baseline model...")
            current_model = SimpleCNN().to(device)
            # Load the state dict from the baseline training
            current_model.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location=device))

            # 4. Define Optimizer and Criterion (Use fine-tuning LR)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(current_model.parameters(), lr=LROAR_LEARNING_RATE) # Use lower LR

            # 5. Fine-tune the LOADED model
            print(f"  Starting fine-tuning ({LROAR_EPOCHS} epochs)...")
            train_model(current_model, trainloader, criterion, optimizer, epochs=LROAR_EPOCHS, description="Fine-tuning")
            print("  Fine-tuning finished.")

            # 6. Evaluate the fine-tuned model
            print("  Evaluating on modified test set...")
            accuracy = evaluate_model(current_model, testloader)
            print(f"  Accuracy: {accuracy:.2f}%")

            # 7. Store result
            results[method][percentage] = accuracy
            end_time_exp = time.time()
            print(f"  Experiment finished in {end_time_exp - start_time_exp:.2f}s")


    # --- Save and Print Results ---
    print("\n--- L-ROAR Experiment Summary (Fine-tuning) ---")
    print(json.dumps(results, indent=4))

    # Save results to a file
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {RESULTS_FILE}")