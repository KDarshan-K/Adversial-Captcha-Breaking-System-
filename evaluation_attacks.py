import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import traceback

# Import our custom modules
from config import (
    MODEL_PATH, BATCH_SIZE, N_CHARS, IMAGE_HEIGHT, IMAGE_WIDTH, TEST_DIR
)
from dataset_loader import CaptchaDataset
from model import CaptchaCNN


def fgsm_attack(model, images, labels, epsilon, device):
    """
    FGSM (Fast Gradient Sign Method) attack.
    """
    images = images.clone().detach().requires_grad_(True).to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(images)  # (batch, 5, 36)

    # Calculate loss for all 5 characters
    loss = 0
    for char_idx in range(5):
        loss += nn.functional.cross_entropy(
            outputs[:, char_idx, :],  # (batch, 36)
            labels[:, char_idx]  # (batch,)
        )

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Create adversarial examples
    data_grad = images.grad.data
    perturbed_images = images + epsilon * data_grad.sign()
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    return perturbed_images.detach()


def pgd_attack(model, images, labels, epsilon, alpha, num_iter, device):
    """
    PGD (Projected Gradient Descent) attack.
    """
    images = images.to(device)
    labels = labels.to(device)

    # Start with random perturbation
    perturbed_images = images.clone().detach()
    perturbed_images = perturbed_images + torch.empty_like(perturbed_images).uniform_(-epsilon, epsilon)
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    for i in range(num_iter):
        perturbed_images.requires_grad = True

        # Forward pass
        outputs = model(perturbed_images)

        # Calculate loss
        loss = 0
        for char_idx in range(5):
            loss += nn.functional.cross_entropy(
                outputs[:, char_idx, :],
                labels[:, char_idx]
            )

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Update perturbation
        with torch.no_grad():
            perturbed_images = perturbed_images + alpha * perturbed_images.grad.sign()

            # Project back to epsilon ball
            perturbation = perturbed_images - images
            perturbation = torch.clamp(perturbation, -epsilon, epsilon)
            perturbed_images = images + perturbation
            perturbed_images = torch.clamp(perturbed_images, 0, 1)

    return perturbed_images.detach()


def attack_model(model, attack_fn, test_loader, epsilon, device, attack_name=""):
    """Runs an attack and measures robust accuracy."""
    print(f"  Attacking with epsilon = {epsilon:.4f}...", end='', flush=True)

    correct_chars = 0
    total_chars = 0
    successful_attacks = 0
    total_images = 0

    model.eval()

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)
        total_images += batch_size

        # Get original predictions
        with torch.no_grad():
            orig_outputs = model(images)
            _, orig_predicted = torch.max(orig_outputs, 2)

        # Generate adversarial examples
        if "PGD" in attack_name:
            alpha = epsilon / 10  # Step size
            num_iter = 40
            adv_images = pgd_attack(model, images, labels, epsilon, alpha, num_iter, device)
        else:  # FGSM
            adv_images = fgsm_attack(model, images, labels, epsilon, device)

        # Get adversarial predictions
        with torch.no_grad():
            adv_outputs = model(adv_images)
            _, adv_predicted = torch.max(adv_outputs, 2)

        # Count accuracy and success
        for i in range(batch_size):
            total_chars += 5
            correct_chars += (adv_predicted[i] == labels[i]).sum().item()

            # Check if attack succeeded (any character changed)
            if not torch.all(adv_predicted[i] == orig_predicted[i]):
                successful_attacks += 1

    robust_accuracy = (correct_chars / total_chars) * 100
    attack_success_rate = (successful_attacks / total_images) * 100 if total_images > 0 else 0

    print(f" Robust Acc: {robust_accuracy:.2f}% | Attack Success: {attack_success_rate:.1f}%")

    return robust_accuracy, attack_success_rate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    if not os.path.isdir(TEST_DIR):
        print(f"Error: Test directory not found at {TEST_DIR}")
        return

    print(f"Loading base model from {MODEL_PATH}")
    model = CaptchaCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # IMPORTANT: Convert to grayscale since model expects 1 channel
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),
    ])

    print(f"Loading test data from: {TEST_DIR}")
    test_dataset = CaptchaDataset(data_dir=TEST_DIR, transform=test_transform)
    attack_batch_size = 32

    # Calculate baseline accuracy
    print("\n--- Calculating Baseline Accuracy ---")
    temp_loader = DataLoader(test_dataset, batch_size=attack_batch_size)
    correct_baseline = 0
    total_baseline = 0

    with torch.no_grad():
        for images, labels in temp_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 2)
            correct_baseline += (predicted == labels).sum().item()
            total_baseline += labels.numel()

    baseline_accuracy = (correct_baseline / total_baseline) * 100
    print(f"Baseline Test Accuracy: {baseline_accuracy:.2f}%\n")

    # Define attacks
    attacks_to_run = {
        "FGSM (L-inf)": "fgsm",
        "PGD-40 (L-inf)": "pgd",
    }

    # Epsilon ranges
    epsilons_list = np.linspace(0.0, 0.3, num=16)

    results = {}
    attack_stats = {}

    # Run attacks
    for attack_name, attack_type in attacks_to_run.items():
        print(f"--- Running Attack: {attack_name} ---")

        robust_accuracies = []
        success_rates = []

        for eps in epsilons_list:
            if eps == 0.0:
                robust_accuracies.append(baseline_accuracy)
                success_rates.append(0.0)
                print(f"  Epsilon=0.0, Robust Acc: {baseline_accuracy:.2f}%")
                continue

            attack_loader = DataLoader(test_dataset, batch_size=attack_batch_size, shuffle=False)
            acc, success_rate = attack_model(
                model, attack_type, attack_loader, eps, device, attack_name
            )
            robust_accuracies.append(acc)
            success_rates.append(success_rate)

        results[attack_name] = (epsilons_list, robust_accuracies)
        attack_stats[attack_name] = (epsilons_list, success_rates)

    # Plot results
    print("\n--- Generating Attack Results Plot ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Robust Accuracy
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    for idx, (attack_name, (epsilons, accuracies)) in enumerate(results.items()):
        ax1.plot(epsilons, accuracies, 'o-', label=attack_name,
                 linewidth=2.5, markersize=7, color=colors[idx % len(colors)])

    ax1.axhline(y=baseline_accuracy, color='red', linestyle='--',
                label=f'Baseline ({baseline_accuracy:.1f}%)', linewidth=2, alpha=0.7)
    ax1.set_title("Adversarial Robustness Analysis", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Attack Strength (Epsilon)", fontsize=13)
    ax1.set_ylabel("Model Accuracy (%)", fontsize=13)
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(-5, 105)

    # Plot 2: Attack Success Rate
    for idx, (attack_name, (epsilons, success_rates)) in enumerate(attack_stats.items()):
        ax2.plot(epsilons, success_rates, 's-', label=attack_name,
                 linewidth=2.5, markersize=7, color=colors[idx % len(colors)])

    ax2.set_title("Attack Success Rate", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Attack Strength (Epsilon)", fontsize=13)
    ax2.set_ylabel("Attack Success Rate (%)", fontsize=13)
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(-5, 105)

    plt.tight_layout()
    plt.savefig('attack_results.png', dpi=300, bbox_inches='tight')
    print("Saved plot to 'attack_results.png'")
    plt.show(block=False)
    plt.pause(0.1)

    # Print summary statistics
    print("\n" + "=" * 60)
    print(" " * 15 + "ATTACK SUMMARY STATISTICS")
    print("=" * 60)

    for attack_name, (epsilons, success_rates) in attack_stats.items():
        acc_values = results[attack_name][1]

        print(f"\n{attack_name}:")
        print("-" * 60)

        # Find epsilon where accuracy drops below various thresholds
        thresholds = [70, 60, 50, 40, 30, 20, 10]
        for threshold in thresholds:
            eps_threshold = None
            for i, acc in enumerate(acc_values):
                if acc < threshold:
                    eps_threshold = epsilons[i]
                    break

            if eps_threshold:
                print(f"  Accuracy < {threshold}% at ε = {eps_threshold:.4f}")
            else:
                print(f"  Accuracy never drops below {threshold}%")
                break  # No need to check lower thresholds

        # Max success rate
        max_success = max(success_rates)
        eps_at_max = epsilons[success_rates.index(max_success)]
        print(f"\n  Maximum Attack Success Rate: {max_success:.1f}% at ε = {eps_at_max:.4f}")

        # Final accuracy at max epsilon
        final_acc = acc_values[-1]
        final_eps = epsilons[-1]
        print(f"  Final Accuracy at ε = {final_eps:.4f}: {final_acc:.2f}%")

    print("\n" + "=" * 60)
    print("\n--- Evaluation Complete ---")
    print("Results saved to 'attack_results.png'")
    print("\nPress Enter to close...")
    input()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print("\n--- A CRITICAL ERROR OCCURRED ---")
        print(f"ERROR: {e}")
        print("\n--- FULL TRACEBACK ---")
        traceback.print_exc()
        input("Press Enter to exit...")
    finally:
        try:
            sys.exit(0)
        except SystemExit:
            pass