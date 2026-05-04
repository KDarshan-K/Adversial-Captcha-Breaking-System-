"""
Simple Attack Visualization
Shows before/after images with predictions for your report.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

from config import MODEL_PATH, IMAGE_HEIGHT, IMAGE_WIDTH, TEST_DIR, CHAR_SET, CAPTCHA_LENGTH
from dataset_loader import CaptchaDataset
from model import CaptchaCNN


def fgsm_attack(model, images, labels, epsilon, device):
    """Generate adversarial examples using FGSM."""
    images = images.clone().detach().requires_grad_(True).to(device)
    labels = labels.to(device)

    outputs = model(images)
    loss = 0
    for char_idx in range(CAPTCHA_LENGTH):
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


def get_prediction(model, image_tensor, device):
    """
    Get model prediction as a string.
    Expects 'image_tensor' to be the 3D tensor [C, H, W]
    """
    with torch.no_grad():
        # Add a batch dimension to make it [1, C, H, W]
        output = model(image_tensor.unsqueeze(0).to(device))
        _, predicted = torch.max(output, 2)
        pred_str = ''.join([CHAR_SET[idx] for idx in predicted[0].cpu().numpy()])
    return pred_str


def get_true_label(label):
    """Convert label to string."""
    return ''.join([CHAR_SET[idx] for idx in label.cpu().numpy()])


def visualize_attacks():
    """Main visualization function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(" " * 20 + "CAPTCHA ATTACK VISUALIZATION")
    print("=" * 80)
    print(f"Device: {device}\n")

    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    model = CaptchaCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully!\n")

    # Load data
    transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    test_dataset = CaptchaDataset(data_dir=TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=True)

    # Get sample images
    try:
        images, labels = next(iter(test_loader))
    except StopIteration:
        print("Error: Test data loader is empty. Check TEST_DIR in config.py")
        return

    # Select epsilon values to visualize
    epsilons = [0.0, 0.05, 0.1, 0.2]
    num_samples = 4  # Number of different CAPTCHAs to show

    # Create figure
    fig = plt.figure(figsize=(20, 5 * num_samples))

    print("Generating visualizations...")
    print("-" * 80)

    for sample_idx in range(num_samples):
        # image tensor shape is [1, 50, 200]
        image = images[sample_idx]
        label = labels[sample_idx]
        true_label = get_true_label(label)

        print(f"\nSample {sample_idx + 1}: True Label = {true_label}")

        for eps_idx, epsilon in enumerate(epsilons):
            # Calculate subplot position
            # This logic was slightly off, correcting it to be 1-indexed and correct
            subplot_idx_base = sample_idx * len(epsilons) * 3 + eps_idx * 3

            # --- This math was complex, simplifying ---
            # We have (num_samples) rows
            # We have (len(epsilons) * 3) columns
            total_cols = len(epsilons) * 3
            row_start_idx = sample_idx * total_cols + 1
            col_start_idx = eps_idx * 3
            subplot_idx = row_start_idx + col_start_idx

            if epsilon == 0.0:
                # Original image (no attack)
                img_display = image.squeeze().cpu().numpy()
                pred = get_prediction(model, image, device)

                # Show original
                ax1 = plt.subplot(num_samples, total_cols, subplot_idx)
                ax1.imshow(img_display, cmap='gray')
                color = 'green' if pred == true_label else 'red'
                ax1.set_title(f'ORIGINAL\nTrue: {true_label}\nPred: {pred}',
                              fontsize=11, fontweight='bold', color=color)
                ax1.axis('off')

                # No perturbation
                ax2 = plt.subplot(num_samples, total_cols, subplot_idx + 1)
                ax2.imshow(np.zeros_like(img_display), cmap='RdBu', vmin=-1, vmax=1)
                ax2.set_title('No Attack\nε=0.00', fontsize=11)
                ax2.axis('off')

                # No difference
                ax3 = plt.subplot(num_samples, total_cols, subplot_idx + 2)
                ax3.imshow(np.zeros_like(img_display), cmap='hot', vmin=0, vmax=0.3)
                ax3.set_title('Difference: 0.0000', fontsize=11)
                ax3.axis('off')

                print(f"  ε=0.00: {pred} (Original)")

            else:
                # Generate adversarial example
                # fgsm_attack returns a batch: [1, 1, 50, 200]
                adv_image_batch = fgsm_attack(model, image.unsqueeze(0),
                                              label.unsqueeze(0), epsilon, device)

                # --- THIS IS THE FIX ---
                # Get the actual image tensor from the batch: [1, 50, 200]
                adv_image_tensor = adv_image_batch[0]

                # Pass the 3D tensor [1, 50, 200] to get_prediction
                adv_pred = get_prediction(model, adv_image_tensor, device)

                # Calculate perturbation: [1, 50, 200] - [1, 50, 200]
                perturbation = adv_image_tensor - image

                # Convert to numpy, squeezing out the channel dim (1) for plotting
                orig_np = image.squeeze().cpu().numpy()
                adv_np = adv_image_tensor.squeeze().cpu().numpy()
                pert_np = perturbation.squeeze().cpu().numpy()
                diff_np = np.abs(adv_np - orig_np)

                # Show adversarial image
                ax1 = plt.subplot(num_samples, total_cols, subplot_idx)
                ax1.imshow(adv_np, cmap='gray')
                color = 'green' if adv_pred == true_label else 'red'
                ax1.set_title(f'ATTACKED\nTrue: {true_label}\nPred: {adv_pred}',
                              fontsize=11, fontweight='bold', color=color)
                ax1.axis('off')

                # Show perturbation (amplified 10x for visibility)
                ax2 = plt.subplot(num_samples, total_cols, subplot_idx + 1)
                pert_display = pert_np * 10
                ax2.imshow(pert_display, cmap='RdBu', vmin=-1, vmax=1)
                ax2.set_title(f'Perturbation (×10)\nε={epsilon:.2f}', fontsize=11)
                ax2.axis('off')

                # Show absolute difference
                ax3 = plt.subplot(num_samples, total_cols, subplot_idx + 2)
                ax3.imshow(diff_np, cmap='hot', vmin=0, vmax=0.3)
                max_diff = diff_np.max()
                ax3.set_title(f'Difference\nMax: {max_diff:.4f}', fontsize=11)
                ax3.axis('off')

                status = "✓ FOOLED" if adv_pred != true_label else "✗ Failed"
                print(f"  ε={epsilon:.2f}: {adv_pred} {status}")

    plt.suptitle('FGSM Attack Visualization: Original vs Attacked CAPTCHAs',
                 fontsize=16, fontweight='bold', y=1.02)  # Adjust y
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # Adjust layout

    # Save figure
    output_file = 'attack_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print("\n" + "=" * 80)
    print(f"Visualization saved to: {output_file}")
    print("=" * 80)

    # Create a simpler side-by-side comparison
    print("\nCreating simple before/after comparison...")

    fig2, axes = plt.subplots(3, 4, figsize=(16, 12))
    epsilon = 0.15  # Fixed epsilon for comparison

    for idx in range(3):
        image = images[idx]  # [1, 50, 200]
        label = labels[idx]
        true_label = get_true_label(label)

        # Original
        orig_np = image.squeeze().cpu().numpy()
        orig_pred = get_prediction(model, image, device)

        axes[idx, 0].imshow(orig_np, cmap='gray')
        axes[idx, 0].set_title(f'Original CAPTCHA\nTrue: {true_label}',
                               fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')

        axes[idx, 1].text(0.5, 0.5, f'Prediction:\n{orig_pred}',
                          ha='center', va='center', fontsize=14, fontweight='bold',
                          color='green' if orig_pred == true_label else 'red')
        axes[idx, 1].axis('off')

        # Attacked
        # --- THIS IS THE SECOND FIX ---
        adv_image_batch = fgsm_attack(model, image.unsqueeze(0),
                                      label.unsqueeze(0), epsilon, device)
        adv_image_tensor = adv_image_batch[0]  # Get [1, 50, 200]

        adv_np = adv_image_tensor.squeeze().cpu().numpy()  # Squeeze to [50, 200] for plot
        adv_pred = get_prediction(model, adv_image_tensor, device)  # Pass [1, 50, 200]

        axes[idx, 2].imshow(adv_np, cmap='gray')
        axes[idx, 2].set_title(f'After FGSM Attack (ε={epsilon})\nTrue: {true_label}',
                               fontsize=12, fontweight='bold')
        axes[idx, 2].axis('off')

        axes[idx, 3].text(0.5, 0.5, f'Prediction:\n{adv_pred}',
                          ha='center', va='center', fontsize=14, fontweight='bold',
                          color='green' if adv_pred == true_label else 'red')
        axes[idx, 3].axis('off')

    plt.suptitle('Before and After FGSM Attack Comparison',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_file2 = 'before_after_comparison.png'
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"Before/after comparison saved to: {output_file2}")

    print("\nShowing plots...")
    plt.show()


if __name__ == "__main__":
    try:
        visualize_attacks()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"{e}")
        import traceback

        traceback.print_exc()
        input("Press Enter to exit...")
