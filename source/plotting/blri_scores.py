"""
Plot histogram of BLRI score differences (therapist - client) across all interviews
and show distribution of binary empathy categories.

This script:
1. Loads data_model.yaml
2. Extracts BLRI scores for therapist (TX_BLRI_ges_In) and client (TX_BLRI_ges_Pr)
3. Calculates difference: therapist_blri - client_blri
4. Discretizes into binary categories: "equally empathic" (-6 to 6) vs "discrepancy" (outside that range)
5. Plots histogram and bar chart showing distribution
6. Saves plots to specified output directory

Positive difference = therapist finds client more empathic
Negative difference = client finds therapist more empathic
"""

import yaml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Tuple
import argparse
import math


def load_data_model(yaml_path: Path) -> dict:
    """Load the data_model.yaml file."""
    print(f"Loading data model from {yaml_path}...")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    print(f"✅ Loaded {len(data['interviews'])} interviews")
    return data


def _is_nan_like(x) -> bool:
    """Return True if x is None or cannot be interpreted as a finite number (NaN/Inf/string nan)."""
    if x is None:
        return True
    # Handle strings like 'nan'
    if isinstance(x, str):
        try:
            xv = float(x)
        except Exception:
            return True
        return not np.isfinite(xv)
    # Handle numeric types
    try:
        xv = float(x)
    except Exception:
        return True
    return not np.isfinite(xv)


def discretize_blri_difference(blri_diff: float) -> str:
    """
    Discretize BLRI difference into binary categories (same as combine_transcripts_with_time_series_descriptions.py).
    
    Args:
        blri_diff: Difference between therapist and client BLRI (therapist - client)
        
    Returns:
        String label for the empathy category: "equally empathic" or "discrepancy"
    """
    if -6 <= blri_diff <= 6:
        return "equally empathic"
    else:
        return "discrepancy"


def extract_blri_differences(data_model: dict) -> Tuple[List[float], List[str], List[dict]]:
    """Extract BLRI differences from all interviews.
    
    Returns:
        Tuple of (blri_differences, labels, interview_details)
        - blri_differences: list of (therapist - client) BLRI scores
        - labels: list of strings describing each data point (patient_id, interview_type)
        - interview_details: list of dicts with full interview metadata
    """
    blri_differences = []
    labels = []
    interview_details = []
    
    for interview in data_model['interviews']:
        patient_id = interview['patient']['patient_id']
        therapist_id = interview['therapist']['therapist_id']
        
        # Process each interview type
        for interview_type in ['bindung', 'personal', 'wunder']:
            if interview_type not in interview.get('types', {}):
                continue
            
            type_data = interview['types'][interview_type]
            labels_data = type_data.get('labels', {})
            
            # Find BLRI keys (they vary by interview type: T3, T5, T7)
            therapist_blri = None
            client_blri = None
            
            for key, value in labels_data.items():
                if 'BLRI_ges_In' in key:  # Interviewer/Therapist
                    therapist_blri = value
                elif 'BLRI_ges_Pr' in key:  # Patient/Client
                    client_blri = value
            
            # Skip if either score is missing or NaN-like
            if therapist_blri is None or client_blri is None:
                continue
            if _is_nan_like(therapist_blri) or _is_nan_like(client_blri):
                # skip entries where one of the BLRI scores is NaN/Inf/unparseable
                continue

            # Calculate difference using numeric values
            therapist_val = float(therapist_blri)
            client_val = float(client_blri)
            diff = therapist_val - client_val
            blri_differences.append(diff)
            labels.append(f"{patient_id}_{interview_type}")
            # Store detailed information
            interview_details.append({
                'patient_id': patient_id,
                'therapist_id': therapist_id,
                'interview_type': interview_type,
                'therapist_blri': therapist_blri,
                'client_blri': client_blri,
                'difference': diff,
                'therapist_openface': type_data.get('therapist_openface', 'N/A'),
                'patient_openface': type_data.get('patient_openface', 'N/A'),
                'transcript': type_data.get('transcript', 'N/A')
            })
    
    print(f"✅ Extracted {len(blri_differences)} BLRI difference scores")
    return blri_differences, labels, interview_details


def plot_blri_histogram(
    blri_differences: List[float],
    output_dir: Path,
    bins: int = 20,
    figsize: Tuple[int, int] = (12, 7)
):
    """Create and save histogram of BLRI differences.
    
    Args:
        blri_differences: List of BLRI difference scores
        output_dir: Directory to save plot
        bins: Number of histogram bins
        figsize: Figure size (width, height)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    n, bins_edges, patches = ax.hist(
        blri_differences,
        bins=bins,
        color='steelblue',
        alpha=0.7,
        edgecolor='black',
        linewidth=1.2
    )
    
    # Color bars based on positive/negative
    for i, patch in enumerate(patches):
        bin_center = (bins_edges[i] + bins_edges[i+1]) / 2
        if bin_center > 0:
            patch.set_facecolor('#2E86AB')  # Blue for positive (therapist > client)
        elif bin_center < 0:
            patch.set_facecolor('#A23B72')  # Purple for negative (client > therapist)
        else:
            patch.set_facecolor('#6A994E')  # Green for zero
    
    # Add vertical line at zero
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero difference')
    
    # Add mean line
    mean_diff = np.mean(blri_differences)
    ax.axvline(x=mean_diff, color='orange', linestyle='-', linewidth=2, alpha=0.8, label=f'Mean: {mean_diff:.2f}')
    
    # Labels and title
    ax.set_xlabel('BLRI Difference (Therapist - Client)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency (Number of Interviews)', fontsize=13, fontweight='bold')
    ax.set_title(
        'Distribution of BLRI Empathy Score Differences\n'
        'Positive = Therapist finds Client more empathic | Negative = Client finds Therapist more empathic',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Add statistics text box
    stats_text = (
        f'N = {len(blri_differences)}\n'
        f'Mean = {np.mean(blri_differences):.2f}\n'
        f'Median = {np.median(blri_differences):.2f}\n'
        f'SD = {np.std(blri_differences):.2f}\n'
        f'Min = {np.min(blri_differences):.2f}\n'
        f'Max = {np.max(blri_differences):.2f}'
    )
    ax.text(
        0.02, 0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    # Legend
    ax.legend(loc='upper right', fontsize=11)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'blri_difference_histogram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved histogram to {output_path}")
    
    # Also save as PDF for publications
    output_path_pdf = output_dir / 'blri_difference_histogram.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"✅ Saved PDF to {output_path_pdf}")
    
    plt.close()


def print_summary_statistics(blri_differences: List[float]):
    """Print summary statistics of BLRI differences."""
    print("\n" + "="*60)
    print("BLRI Difference Summary Statistics")
    print("="*60)
    print(f"Total interviews: {len(blri_differences)}")
    print(f"Mean difference: {np.mean(blri_differences):.2f}")
    print(f"Median difference: {np.median(blri_differences):.2f}")
    print(f"Standard deviation: {np.std(blri_differences):.2f}")
    print(f"Min difference: {np.min(blri_differences):.2f}")
    print(f"Max difference: {np.max(blri_differences):.2f}")
    print(f"\nPositive differences (therapist > client): {sum(1 for d in blri_differences if d > 0)} ({100*sum(1 for d in blri_differences if d > 0)/len(blri_differences):.1f}%)")
    print(f"Negative differences (client > therapist): {sum(1 for d in blri_differences if d < 0)} ({100*sum(1 for d in blri_differences if d < 0)/len(blri_differences):.1f}%)")
    print(f"Zero differences: {sum(1 for d in blri_differences if d == 0)}")
    print("="*60 + "\n")


def print_category_distribution(blri_differences: List[float]):
    """Print distribution of binary empathy categories."""
    categories = [discretize_blri_difference(diff) for diff in blri_differences]
    
    equally_empathic = sum(1 for c in categories if c == "equally empathic")
    discrepancy = sum(1 for c in categories if c == "discrepancy")
    total = len(categories)
    
    print("\n" + "="*60)
    print("Binary Category Distribution")
    print("="*60)
    print(f"Total interviews: {total}")
    print(f"\nEqually empathic (|diff| ≤ 8): {equally_empathic} ({100*equally_empathic/total:.1f}%)")
    print(f"Discrepancy (|diff| > 8):      {discrepancy} ({100*discrepancy/total:.1f}%)")
    print("="*60 + "\n")


def plot_category_distribution(
    blri_differences: List[float],
    output_dir: Path,
    figsize: Tuple[int, int] = (10, 6)
):
    """Create and save bar chart of binary category distribution.
    
    Args:
        blri_differences: List of BLRI difference scores
        output_dir: Directory to save plot
        figsize: Figure size (width, height)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate category counts
    categories = [discretize_blri_difference(diff) for diff in blri_differences]
    equally_empathic = sum(1 for c in categories if c == "equally empathic")
    discrepancy = sum(1 for c in categories if c == "discrepancy")
    total = len(categories)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Data for bar chart
    category_names = ["Equally Empathic\n(|diff| ≤ 8)", "Discrepancy\n(|diff| > 8)"]
    counts = [equally_empathic, discrepancy]
    percentages = [100*equally_empathic/total, 100*discrepancy/total]
    colors = ['#6A994E', '#BC4749']
    
    # Create bars
    bars = ax.bar(category_names, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{count}\n({pct:.1f}%)',
            ha='center',
            va='bottom',
            fontsize=14,
            fontweight='bold'
        )
    
    # Labels and title
    ax.set_ylabel('Number of Interviews', fontsize=13, fontweight='bold')
    ax.set_title(
        'Distribution of Empathy Categories (Binary Classification)\n'
        f'Total N = {total} interviews',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    
    # Y-axis starts at 0
    ax.set_ylim(bottom=0, top=max(counts) * 1.15)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'blri_category_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved category distribution to {output_path}")
    
    # Also save as PDF
    output_path_pdf = output_dir / 'blri_category_distribution.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"✅ Saved PDF to {output_path_pdf}")
    
    plt.close()


def print_extreme_differences(interview_details: List[dict], threshold: float = 15.0):
    """Print details of interviews with extreme BLRI differences.
    
    Args:
        interview_details: List of interview metadata dicts
        threshold: Absolute difference threshold (default: 15.0)
    """
    # Filter interviews with |difference| > threshold
    extreme_interviews = [
        detail for detail in interview_details
        if abs(detail['difference']) > threshold
    ]
    
    if not extreme_interviews:
        print(f"ℹ️ No interviews found with |difference| > {threshold}")
        return
    
    # Sort by difference in descending order (most positive first)
    extreme_interviews.sort(key=lambda x: x['difference'], reverse=True)
    
    print("\n" + "="*100)
    print(f"🔍 INTERVIEWS WITH EXTREME BLRI DIFFERENCES (|diff| > {threshold})")
    print("="*100)
    print(f"Found {len(extreme_interviews)} interview(s)\n")
    
    for i, detail in enumerate(extreme_interviews, 1):
        print(f"\n{'─'*100}")
        print(f"#{i} | Patient: {detail['patient_id']} | Therapist: {detail['therapist_id']} | Type: {detail['interview_type']}")
        print(f"{'─'*100}")
        print(f"  Therapist BLRI: {detail['therapist_blri']:.2f}")
        print(f"  Client BLRI:    {detail['client_blri']:.2f}")
        print(f"  Difference:     {detail['difference']:.2f} {'(Therapist >> Client)' if detail['difference'] > 0 else '(Client >> Therapist)'}")
        print(f"\n  📁 File Paths:")
        print(f"     Therapist OpenFace: {detail['therapist_openface']}")
        print(f"     Patient OpenFace:   {detail['patient_openface']}")
        print(f"     Transcript:         {detail['transcript']}")
    
    print("\n" + "="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot histogram of BLRI score differences (therapist - client)"
    )
    parser.add_argument(
        "--data_model",
        type=Path,
        default=Path("C:/Users/User/Desktop/martins/synchrony/data_model.yaml"),
        help="Path to data_model.yaml"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("C:/Users/User/Desktop/martins/plots"),
        help="Output directory for plots"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of histogram bins (default: 20)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=15.0,
        help="Threshold for printing extreme differences (default: 15.0)"
    )
    
    args = parser.parse_args()
    
    print("📊 Creating BLRI difference histogram and category distribution")
    print("=" * 60)
    print(f"Data model: {args.data_model}")
    print(f"Output dir: {args.output_dir}")
    print(f"Bins: {args.bins}")
    print(f"Extreme threshold: {args.threshold}")
    print()
    
    # Load data
    data_model = load_data_model(args.data_model)
    
    # Extract BLRI differences
    blri_differences, labels, interview_details = extract_blri_differences(data_model)
    
    if not blri_differences:
        print("❌ No BLRI scores found in data model")
        return
    
    # Print statistics
    print_summary_statistics(blri_differences)
    
    # Print binary category distribution
    print_category_distribution(blri_differences)
    
    # Print extreme differences
    print_extreme_differences(interview_details, threshold=args.threshold)
    
    # Create and save histogram
    plot_blri_histogram(blri_differences, args.output_dir, bins=args.bins)
    
    # Create and save category distribution bar chart
    plot_category_distribution(blri_differences, args.output_dir)
    
    print(f"\n✅ Complete! Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
