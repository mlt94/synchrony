"""
Combine BLRI scores from .sav file with labels_cleaned.csv.

This script:
1. Reads the new BLRI scores from .sav file (data_kommunikationpsych_2025-11-17_IRF_BLRI_items.sav)
2. For each client (ID_Proband), extracts 6 rows (3 for patient, 3 for therapist across 3 interview types)
3. Sums BLRI_E + BLRI_R + BLRI_C for each row
4. Updates the corresponding columns in labels_cleaned.csv:
   - T3_BLRI_ges_Pr (patient at T3)
   - T3_BLRI_ges_In (therapist at T3)
   - T5_BLRI_ges_Pr (patient at T5)
   - T5_BLRI_ges_In (therapist at T5)
   - T7_BLRI_ges_Pr (patient at T7)
   - T7_BLRI_ges_In (therapist at T7)

Data structure in .sav file:
- Each client has 6 rows (3 measurement points × 2 people)
- Rows where ID_Interviewerin is NaN/empty → patient scores
- Rows where ID_Interviewerin is populated → therapist scores
- Measurement_time_point indicates T3, T5, or T7
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def load_blri_scores(sav_path: Path) -> pd.DataFrame:
    """Load BLRI scores from .sav file and compute total scores.
    
    Returns:
        DataFrame with columns: ID_Proband, ID_Interviewerin, Measurement_time_point, BLRI_total
    """
    print(f"Loading BLRI data from {sav_path}...")
    df = pd.read_spss(sav_path)
    
    # Compute total BLRI score (sum of E, R, C)
    df['BLRI_total'] = df['BLRI_E'] + df['BLRI_R'] + df['BLRI_C']
    
    # Keep only relevant columns
    df_clean = df[['ID_Proband', 'ID_Interviewerin', 'Measurement_time_point', 'BLRI_total']].copy()
    
    print(f"✅ Loaded {len(df_clean)} BLRI score records")
    print(f"   Unique clients: {df_clean['ID_Proband'].nunique()}")
    print(f"   Time points: {sorted(df_clean['Measurement_time_point'].unique())}")
    
    return df_clean


def update_labels_with_blri(labels_df: pd.DataFrame, blri_df: pd.DataFrame) -> pd.DataFrame:
    """Update labels DataFrame with new BLRI scores.
    
    Args:
        labels_df: Original labels_cleaned.csv DataFrame
        blri_df: BLRI scores DataFrame from .sav file
    
    Returns:
        Updated labels DataFrame
    """
    print("\nUpdating BLRI scores in labels DataFrame...")
    
    # Create a copy to avoid modifying original
    labels_updated = labels_df.copy()
    
    # Track statistics
    updates_count = {col: 0 for col in ['T3_BLRI_ges_Pr', 'T3_BLRI_ges_In', 
                                          'T5_BLRI_ges_Pr', 'T5_BLRI_ges_In',
                                          'T7_BLRI_ges_Pr', 'T7_BLRI_ges_In']}
    missing_count = 0
    
    # Iterate through each row in labels
    for idx, row in labels_df.iterrows():
        patient_id = row['ID_Proband']
        therapist_id = row['ID_Interviewerin']
        
        # Get all BLRI records for this patient
        patient_records = blri_df[blri_df['ID_Proband'] == patient_id]
        
        if patient_records.empty:
            missing_count += 1
            continue
        
        # Process each time point (T3, T5, T7)
        for time_point in ['T3', 'T5', 'T7']:
            time_records = patient_records[patient_records['Measurement_time_point'] == time_point]
            
            if time_records.empty:
                continue
            
            # Patient score (where ID_Interviewerin is NaN/empty string)
            patient_score_records = time_records[
                (time_records['ID_Interviewerin'].isna()) | 
                (time_records['ID_Interviewerin'] == '') |
                (time_records['ID_Interviewerin'].str.strip() == '')
            ]
            if not patient_score_records.empty:
                # Take mean if multiple records (should typically be 1)
                patient_score = patient_score_records['BLRI_total'].mean()
                col_name = f'{time_point}_BLRI_ges_Pr'
                labels_updated.at[idx, col_name] = patient_score
                updates_count[col_name] += 1
            
            # Therapist score (where ID_Interviewerin matches therapist_id)
            therapist_score_records = time_records[time_records['ID_Interviewerin'] == therapist_id]
            if not therapist_score_records.empty:
                # Take mean if multiple records (should typically be 1)
                therapist_score = therapist_score_records['BLRI_total'].mean()
                col_name = f'{time_point}_BLRI_ges_In'
                labels_updated.at[idx, col_name] = therapist_score
                updates_count[col_name] += 1
    
    # Print statistics
    print("\n📊 Update Statistics:")
    for col, count in updates_count.items():
        print(f"   {col}: {count} values updated")
    if missing_count > 0:
        print(f"\n⚠️  Warning: {missing_count} clients in labels_cleaned.csv not found in BLRI data")
    
    return labels_updated


def main():
    parser = argparse.ArgumentParser(
        description="Combine BLRI scores from .sav file with labels_cleaned.csv"
    )
    parser.add_argument(
        "--sav_file",
        type=Path,
        default=Path("C:/Users/mlut/OneDrive - ITU/Desktop/msb/data_kommunikationpsych_2025-11-17_IRF_BLRI_items.sav"),
        help="Path to .sav file with BLRI scores"
    )
    parser.add_argument(
        "--labels_csv",
        type=Path,
        default=Path("C:/Users/mlut/OneDrive - ITU/Desktop/msb/labels_cleaned.csv"),
        help="Path to labels_cleaned.csv"
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Output path for updated labels (default: overwrites input labels_csv)"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of original labels_cleaned.csv before overwriting"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not args.sav_file.exists():
        print(f"❌ Error: .sav file not found: {args.sav_file}")
        return 1
    if not args.labels_csv.exists():
        print(f"❌ Error: labels_cleaned.csv not found: {args.labels_csv}")
        return 1
    
    print("🚀 Starting BLRI score update process")
    print("=" * 80)
    
    # Load data
    blri_df = load_blri_scores(args.sav_file)
    
    print(f"\nLoading labels from {args.labels_csv}...")
    labels_df = pd.read_csv(args.labels_csv)
    print(f"✅ Loaded labels: {labels_df.shape[0]} rows, {labels_df.shape[1]} columns")
    
    # Check for BLRI columns
    blri_cols = [c for c in labels_df.columns if 'BLRI' in c]
    print(f"   Found BLRI columns: {blri_cols}")
    
    # Update labels
    labels_updated = update_labels_with_blri(labels_df, blri_df)
    
    # Determine output path
    output_path = args.output_csv if args.output_csv else args.labels_csv
    
    # Create backup if requested
    if args.backup and output_path == args.labels_csv:
        backup_path = args.labels_csv.parent / f"{args.labels_csv.stem}_backup{args.labels_csv.suffix}"
        print(f"\n💾 Creating backup: {backup_path}")
        labels_df.to_csv(backup_path, index=False)
    
    # Save updated labels
    print(f"\n💾 Saving updated labels to {output_path}...")
    labels_updated.to_csv(output_path, index=False)
    
    print("\n" + "=" * 80)
    print("✅ BLRI scores successfully updated!")
    print(f"   Output saved to: {output_path}")
    
    # Show sample of changes
    print("\n📋 Sample of updated BLRI scores (first 5 rows):")
    blri_cols = [c for c in labels_updated.columns if 'BLRI' in c]
    print(labels_updated[['ID_Proband', 'ID_Interviewerin'] + blri_cols].head())
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
