"""
Generate example heatmaps for presentations using random data.
This mimics the exact format from generate_time_series_descriptions.py
but uses synthetic data so real participant information is not shown.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def generate_example_heatmap(output_path: Path, num_bins: int = 8, seed: int = 42):
    """Generate heatmap visualization with random AU activation data.
    
    This creates the exact same visualization as generate_time_series_descriptions.py
    but uses synthetic data for presentation purposes.
    
    Args:
        output_path: Where to save the plot
        num_bins: Number of temporal bins (default 8)
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Define AU names (same as used in the real pipeline)
    au_names = ['AU12_r', 'AU06_r', 'AU04_r', 'AU15_r']
    
    # Generate realistic random data
    # Real AU activations typically range from 0 to ~5, with most values < 3
    therapist_heatmap = np.random.gamma(2, 0.5, size=(len(au_names), num_bins))
    client_heatmap = np.random.gamma(2, 0.5, size=(len(au_names), num_bins))
    
    # Add some temporal patterns to make it more realistic
    # AU12 (smile) - therapist shows increasing pattern
    therapist_heatmap[0, :] = np.linspace(0.5, 2.5, num_bins) + np.random.normal(0, 0.2, num_bins)
    # AU12 - client shows stable low pattern
    client_heatmap[0, :] = np.ones(num_bins) * 0.8 + np.random.normal(0, 0.15, num_bins)
    
    # AU06 (cheek raiser) - both show moderate activation
    therapist_heatmap[1, :] = np.ones(num_bins) * 1.2 + np.random.normal(0, 0.3, num_bins)
    client_heatmap[1, :] = np.ones(num_bins) * 1.5 + np.random.normal(0, 0.25, num_bins)
    
    # AU04 (brow lowerer) - client shows peak in middle
    client_heatmap[2, :] = np.concatenate([
        np.linspace(0.5, 2.0, num_bins//2),
        np.linspace(2.0, 0.5, num_bins - num_bins//2)
    ]) + np.random.normal(0, 0.2, num_bins)
    
    # AU15 (lip corner depressor) - low for both
    therapist_heatmap[3, :] = np.ones(num_bins) * 0.4 + np.random.normal(0, 0.1, num_bins)
    client_heatmap[3, :] = np.ones(num_bins) * 0.6 + np.random.normal(0, 0.15, num_bins)
    
    # Ensure no negative values
    therapist_heatmap = np.maximum(therapist_heatmap, 0)
    client_heatmap = np.maximum(client_heatmap, 0)
    
    # Create figure with two side-by-side heatmaps (exact same layout as original)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Find global min/max for consistent color scaling
    vmin = min(therapist_heatmap.min(), client_heatmap.min())
    vmax = max(therapist_heatmap.max(), client_heatmap.max())
    
    # Therapist heatmap (left)
    im1 = ax1.imshow(therapist_heatmap, aspect='auto', cmap='Blues', 
                     interpolation='nearest', vmin=vmin, vmax=vmax)
    ax1.set_title('THERAPIST AU Activation', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('Action Unit', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time Progression (Start → End)', fontsize=12, fontweight='bold')
    ax1.set_yticks(range(len(au_names)))
    ax1.set_yticklabels(au_names, fontsize=11)
    ax1.set_xticks(range(num_bins))
    ax1.set_xticklabels(['Start', 'Early', 'Early-Mid', 'Mid', 'Late-Mid', 'Late', 'Very Late', 'End'][:num_bins], 
                        fontsize=9, rotation=45, ha='right')
    
    # Add colorbar for therapist
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Activation Level', fontsize=11, fontweight='bold')
    
    # Add value annotations on therapist heatmap
    for i in range(len(au_names)):
        for j in range(num_bins):
            text = ax1.text(j, i, f'{therapist_heatmap[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    # Client heatmap (right)
    im2 = ax2.imshow(client_heatmap, aspect='auto', cmap='Oranges', 
                     interpolation='nearest', vmin=vmin, vmax=vmax)
    ax2.set_title('CLIENT AU Activation', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel('Action Unit', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time Progression (Start → End)', fontsize=12, fontweight='bold')
    ax2.set_yticks(range(len(au_names)))
    ax2.set_yticklabels(au_names, fontsize=11)
    ax2.set_xticks(range(num_bins))
    ax2.set_xticklabels(['Start', 'Early', 'Early-Mid', 'Mid', 'Mid-Late', 'Late', 'Very Late', 'End'][:num_bins], 
                        fontsize=9, rotation=45, ha='right')
    
    # Add colorbar for client
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Activation Level', fontsize=11, fontweight='bold')
    
    # Add value annotations on client heatmap
    for i in range(len(au_names)):
        for j in range(num_bins):
            text = ax2.text(j, i, f'{client_heatmap[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    # Overall title (with example metadata)
    plt.suptitle(f"Example Turn: Therapist speaking (5000-15000ms)\n" + 
                 f"Heatmap shows mean AU activation across {num_bins} temporal phases (SYNTHETIC DATA)", 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Example heatmap saved to {output_path}")


def main():
    """Generate example heatmaps for different scenarios."""
    
    # Create output directory
    output_dir = Path(__file__).parent / "example_outputs"
    output_dir.mkdir(exist_ok=True)
    
    print("🎨 Generating example heatmaps with synthetic data...")
    print("=" * 80)
    
    # Generate a few examples with different random seeds
    for i, seed in enumerate([42, 123, 456]):
        output_path = output_dir / f"example_heatmap_{i+1}.png"
        generate_example_heatmap(output_path, num_bins=8, seed=seed)
        print(f"  Generated example {i+1} with seed {seed}")
    
    print("\n" + "=" * 80)
    print(f"✅ Complete! Generated 3 example heatmaps in {output_dir}/")
    print(f"   These heatmaps use the exact same format as the real pipeline")
    print(f"   but contain synthetic data safe for presentations.")


if __name__ == "__main__":
    main()
