"""
Attack Visualization Script
Shows original images, attacked images, and the perturbations side-by-side.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from config import (
    MODEL_PATH, ROBUST_MODEL_PATH, IMAGE_HEIGHT, IMAGE_WIDTH, TEST_DIR, CHAR_SET
)
from dataset_loader import CaptchaDataset
from model import CaptchaCNN


def fgsm_attack(model, images, labels, epsilon, device):
    """FGSM attack."""
    images = images.clone().detach().requires_grad_(True).to(device)
    labels = labels.to(device)

    outputs = model(images)
    loss = 0
    for char_idx in range(5):
        loss += nn.functional.cross_entropy(
            outputs[:, char_idx, :],
            labels[:, char_idx]
        )

    model.zero_grad()
    loss.backward()

    data_grad = images.grad.data
    perturbed_images = images + epsilon * data_grad.sign()
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    return perturbed_images.detach()


def pgd_attack(model, images, labels, epsilon, alpha, num_iter, device):
    """PGD attack."""
    images = images.to(device)
    labels = labels.to(device)

    perturbed_images = images.clone().detach()
    perturbed_images = perturbed_images + torch.empty_like(perturbed_images).uniform_(-epsilon, epsilon)
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    for i in range(num_iter):
        perturbed_images.requires_grad = True

        outputs = model(perturbed_images)
        loss = 0
        for char_idx in range(5):
            loss += nn.functional.cross_entropy(
                outputs[:, char_idx, :],
                labels[:, char_idx]
            )

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            perturbed_images = perturbed_images + alpha * perturbed_images.grad.sign()
            perturbation = perturbed_images - images
            perturbation = torch.clamp(perturbation, -epsilon, epsilon)
            perturbed_images = images + perturbation
            perturbed_images = torch.clamp(perturbed_images, 0, 1)

    return perturbed_images.detach()


def decode_predictions(outputs):
    """Convert model outputs to readable strings."""
    _, predicted = torch.max(outputs, 2)  # (batch, 5)
    predicted_strings = []

    for pred in predicted:
        pred_str = ''.join([CHAR_SET[idx] for idx in pred.cpu().numpy()])
        predicted_strings.append(pred_str)

    return predicted_strings


def decode_labels(labels):
    """Convert label indices to readable strings."""
    label_strings = []
    for label in labels:
        label_str = ''.join([CHAR_SET[idx] for idx in label.cpu().numpy()])
        label_strings.append(label_str)
    return label_strings


def visualize_single_attack(model, image, label, epsilon, attack_type, device, model_name="Model"):
    """
    Visualize a single image attack.
    Returns: original image, adversarial image, perturbation, predictions
    """
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    label = label.unsqueeze(0).to(device)

    # Get original prediction
    with torch.no_grad():
        orig_output = model(image)
        orig_pred = decode_predictions(orig_output)[0]

    # Generate adversarial example
    if attack_type == "fgsm":
        adv_image = fgsm_attack(model, image, label, epsilon, device)
    else:  # pgd
        alpha = epsilon / 10
        adv_image = pgd_attack(model, image, label, epsilon, alpha, 40, device)

    # Get adversarial prediction
    with torch.no_grad():
        adv_output = model(adv_image)
        adv_pred = decode_predictions(adv_output)[0]

    # Calculate perturbation
    perturbation = adv_image - image

    # Convert to numpy for visualization
    orig_np = image.squeeze().cpu().numpy()
    adv_np = adv_image.squeeze().cpu().numpy()
    pert_np = perturbation.squeeze().cpu().numpy()

    true_label = decode_labels(label)[0]

    return orig_np, adv_np, pert_np, true_label, orig_pred, adv_pred


def create_attack_comparison_grid(model, test_loader, device, model_name, num_samples=4):
    """
    Create a grid showing multiple attacks on different images.
    """
    # Get sample images
    images_list = []
    labels_list = []

    for images, labels in test_loader:
        for i in range(min(num_samples, len(images))):
            images_list.append(images[i])
            labels_list.append(labels[i])
            if len(images_list) >= num_samples:
                break
        if len(images_list) >= num_samples:
            break

    # Attack configurations
    attacks = [
        ("FGSM ε=0.05", "fgsm", 0.05),
        ("FGSM ε=0.1", "fgsm", 0.1),
        ("FGSM ε=0.2", "fgsm", 0.2),
        ("PGD ε=0.1", "pgd", 0.1),
    ]

    # Create figure
    fig = plt.figure(figsize=(20, 5 * num_samples))

    for sample_idx in range(num_samples):
        image = images_list[sample_idx]
        label = labels_list[sample_idx]

        for attack_idx, (attack_name, attack_type, epsilon) in enumerate(attacks):
            # Get attack results
            orig_np, adv_np, pert_np, true_label, orig_pred, adv_pred = \
                visualize_single_attack(model, image, label, epsilon, attack_type, device, model_name)

            # Calculate row position
            row = sample_idx * 4 + attack_idx

            # Original image
            ax1 = plt.subplot(num_samples * 4, 4, row * 4 + 1)
            ax1.imshow(orig_np, cmap='gray')
            ax1.set_title(f'Original\nTrue: {true_label}\nPred: {orig_pred}',
                          fontsize=10, fontweight='bold')
            ax1.axis('off')

            # Adversarial image
            ax2 = plt.subplot(num_samples * 4, 4, row * 4 + 2)
            ax2.imshow(adv_np, cmap='gray')
            color = 'red' if adv_pred != true_label else 'green'
            ax2.set_title(f'{attack_name}\nPred: {adv_pred}',
                          fontsize=10, fontweight='bold', color=color)
            ax2.axis('off')

            # Perturbation (amplified for visibility)
            ax3 = plt.subplot(num_samples * 4, 4, row * 4 + 3)
            # Amplify perturbation for better visibility
            pert_display = pert_np * 10
            ax3.imshow(pert_display, cmap='RdBu', vmin=-1, vmax=1)
            ax3.set_title(f'Perturbation (×10)\nε={epsilon}', fontsize=10)
            ax3.axis('off')

            # Difference visualization
            ax4 = plt.subplot(num_samples * 4, 4, row * 4 + 4)
            diff = np.abs(adv_np - orig_np)
            ax4.imshow(diff, cmap='hot')
            ax4.set_title(f'Abs. Difference\nMax: {diff.max():.4f}', fontsize=10)
            ax4.axis('off')

    plt.suptitle(f'{model_name} - Attack Visualization',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    return fig


def create_epsilon_progression(model, image, label, attack_type, device, model_name):
    """
    Show how perturbation increases with epsilon.
    """
    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    fig, axes = plt.subplots(3, len(epsilons), figsize=(3 * len(epsilons), 9))

    for idx, epsilon in enumerate(epsilons):
        if epsilon == 0.0:
            # No attack
            orig_np = image.squeeze().cpu().numpy()
            adv_np = orig_np
            pert_np = np.zeros_like(orig_np)

            with torch.no_grad():
                output = model(image.unsqueeze(0).to(device))
                pred = decode_predictions(output)[0]

            true_label = decode_labels(label.unsqueeze(0))[0]
            orig_pred = pred
            adv_pred = pred
        else:
            orig_np, adv_np, pert_np, true_label, orig_pred, adv_pred = \
                visualize_single_attack(model, image, label, epsilon, attack_type, device, model_name)

        # Original/Adversarial image
        axes[0, idx].imshow(adv_np, cmap='gray')
        color = 'red' if adv_pred != true_label else 'green'
        axes[0, idx].set_title(f'ε={epsilon:.2f}\nPred: {adv_pred}',
                               fontweight='bold', color=color, fontsize=11)
        axes[0, idx].axis('off')

        # Perturbation (amplified)
        pert_display = pert_np * 10
        im = axes[1, idx].imshow(pert_display, cmap='RdBu', vmin=-1, vmax=1)
        axes[1, idx].set_title(f'Perturbation (×10)', fontsize=10)
        axes[1, idx].axis('off')

        # Absolute difference
        diff = np.abs(adv_np - orig_np)
        axes[2, idx].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
        axes[2, idx].set_title(f'Max Δ: {diff.max():.4f}', fontsize=10)
        axes[2, idx].axis('off')

    plt.suptitle(f'{model_name} - Epsilon Progression ({attack_type.upper()})\nTrue Label: {true_label}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(" " * 25 + "ATTACK VISUALIZATION TOOL")
    print("=" * 80)
    print(f"Using device: {device}\n")

    # Check if models exist
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Base model not found at {MODEL_PATH}")
        return

    # Load data
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    print(f"Loading test data from: {TEST_DIR}")
    test_dataset = CaptchaDataset(data_dir=TEST_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    # Load base model
    print("Loading base model...")
    base_model = CaptchaCNN().to(device)
    base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    base_model.eval()
    print("Model loaded successfully!\n")

    # Get a sample image for epsilon progression
    sample_images, sample_labels = next(iter(test_loader))
    sample_image = sample_images[0]
    sample_label = sample_labels[0]

    print("Generating visualizations...")
    print("-" * 80)

    # 1. Create attack comparison grid
    print("1. Creating attack comparison grid...")
    fig1 = create_attack_comparison_grid(base_model, test_loader, device,
                                         "Base Model", num_samples=3)
    fig1.savefig('attack_comparison_grid.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved to 'attack_comparison_grid.png'")

    # 2. Create epsilon progression for FGSM
    print("2. Creating FGSM epsilon progression...")
    fig2 = create_epsilon_progression(base_model, sample_image, sample_label,
                                      "fgsm", device, "Base Model")
    fig2.savefig('epsilon_progression_fgsm.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved to 'epsilon_progression_fgsm.png'")

    # 3. Create epsilon progression for PGD
    print("3. Creating PGD epsilon progression...")
    fig3 = create_epsilon_progression(base_model, sample_image, sample_label,
                                      "pgd", device, "Base Model")
    fig3.savefig('epsilon_progression_pgd.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved to 'epsilon_progression_pgd.png'")

    # 4. If robust model exists, create comparison
    if os.path.exists(ROBUST_MODEL_PATH):
        print("\n4. Creating robust model comparison...")
        robust_model = CaptchaCNN().to(device)
        robust_model.load_state_dict(torch.load(ROBUST_MODEL_PATH, map_location=device))
        robust_model.eval()

        # Create side-by-side comparison
        fig4, axes = plt.subplots(2, 7, figsize=(21, 6))
        epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

        for idx, epsilon in enumerate(epsilons):
            # Base model
            if epsilon == 0.0:
                orig_np = sample_image.squeeze().cpu().numpy()
                axes[0, idx].imshow(orig_np, cmap='gray')
                with torch.no_grad():
                    output = base_model(sample_image.unsqueeze(0).to(device))
                    pred = decode_predictions(output)[0]
                true_label = decode_labels(sample_label.unsqueeze(0))[0]
                axes[0, idx].set_title(f'Original\n{pred}', fontweight='bold', fontsize=10)
            else:
                _, adv_np, _, true_label, _, adv_pred = \
                    visualize_single_attack(base_model, sample_image, sample_label,
                                            epsilon, "fgsm", device, "Base")
                axes[0, idx].imshow(adv_np, cmap='gray')
                color = 'red' if adv_pred != true_label else 'green'
                axes[0, idx].set_title(f'ε={epsilon:.2f}\n{adv_pred}',
                                       fontweight='bold', color=color, fontsize=10)
            axes[0, idx].axis('off')

            # Robust model
            if epsilon == 0.0:
                axes[1, idx].imshow(orig_np, cmap='gray')
                with torch.no_grad():
                    output = robust_model(sample_image.unsqueeze(0).to(device))
                    pred = decode_predictions(output)[0]
                axes[1, idx].set_title(f'Original\n{pred}', fontweight='bold', fontsize=10)
            else:
                _, adv_np, _, true_label, _, adv_pred = \
                    visualize_single_attack(robust_model, sample_image, sample_label,
                                            epsilon, "fgsm", device, "Robust")
                axes[1, idx].imshow(adv_np, cmap='gray')
                color = 'red' if adv_pred != true_label else 'green'
                axes[1, idx].set_title(f'ε={epsilon:.2f}\n{adv_pred}',
                                       fontweight='bold', color=color, fontsize=10)
            axes[1, idx].axis('off')

        axes[0, 0].set_ylabel('Base Model', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Robust Model', fontsize=12, fontweight='bold')

        plt.suptitle(f'Base vs Robust Model Comparison (FGSM)\nTrue Label: {true_label}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig4.savefig('base_vs_robust_comparison.png', dpi=300, bbox_inches='tight')
        print("   ✓ Saved to 'base_vs_robust_comparison.png'")

    print("\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. attack_comparison_grid.png - Multiple samples with different attacks")
    print("  2. epsilon_progression_fgsm.png - How FGSM changes with epsilon")
    print("  3. epsilon_progression_pgd.png - How PGD changes with epsilon")
    if os.path.exists(ROBUST_MODEL_PATH):
        print("  4. base_vs_robust_comparison.png - Direct comparison of models")

    print("\nShowing plots...")
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n--- ERROR OCCURRED ---")
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)