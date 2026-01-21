"""
Shared utility functions for the Medical VAE Clustering project.

This module consolidates commonly used functions to avoid code duplication.

Usage:
    from utils import load_cluster_data, find_score
"""

import os
import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from config import Config


def load_cluster_data(
    cluster_table_path: str | Path,
    metadata_path: Optional[str | Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load cluster data from WandB export and metadata CSV.
    
    Supports both frame-level data (with 'image_path' column) and 
    video-level data (with 'video_id' column).
    
    Args:
        cluster_table_path: Path to WandB exported cluster table (JSON or CSV)
        metadata_path: Path to metadata CSV. Defaults to Config.METADATA_PATH
    
    Returns:
        Tuple of (cluster_df, df_metadata)
    
    Raises:
        ValueError: If cluster_table_path has unsupported file extension
    """
    print("Loading data...")
    
    # Use default metadata path if not provided
    if metadata_path is None:
        metadata_path = Config.METADATA_PATH
    
    # 1. Load Metadata
    df_metadata = pd.read_csv(metadata_path)
    df_metadata['video_id'] = df_metadata['File Path'].apply(lambda p: Path(str(p)).stem)
    
    # 2. Load WandB Cluster Table (JSON or CSV)
    cluster_table_path = Path(cluster_table_path)
    file_ext = cluster_table_path.suffix.lower()
    
    if file_ext == '.json':
        print(f"Loading cluster data from JSON: {cluster_table_path}")
        with open(cluster_table_path, 'r') as f:
            data = json.load(f)
        cluster_df = pd.DataFrame(data['data'], columns=data['columns'])
    elif file_ext == '.csv':
        print(f"Loading cluster data from CSV: {cluster_table_path}")
        cluster_df = pd.read_csv(cluster_table_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Expected .json or .csv")
    
    # Detect data level (frame vs video) and normalize column names
    has_video_id = 'video_id' in cluster_df.columns
    has_image_path = 'image_path' in cluster_df.columns
    
    if has_video_id and not has_image_path:
        # Video-level data: use video_id as the primary identifier
        print("Detected VIDEO-LEVEL clustering data")
        cluster_df['image_path'] = cluster_df['video_id']  # Alias for compatibility
        cluster_df['_data_level'] = 'video'
    elif has_image_path:
        # Frame-level data: extract video_id from image_path
        print("Detected FRAME-LEVEL clustering data")
        cluster_df['video_id'] = cluster_df['image_path'].apply(
            lambda p: Path(str(p)).stem.split('_selected_frame')[0] 
            if '_selected_frame' in str(p) else Path(str(p)).stem
        )
        cluster_df['_data_level'] = 'frame'
    else:
        raise ValueError("Cluster table must contain either 'video_id' or 'image_path' column")

    # Ensure numerical columns are actually numbers
    if 'tsne_x' in cluster_df.columns:
        cluster_df['tsne_x'] = pd.to_numeric(cluster_df['tsne_x'])
    if 'tsne_y' in cluster_df.columns:
        cluster_df['tsne_y'] = pd.to_numeric(cluster_df['tsne_y'])
    
    return cluster_df, df_metadata


def find_score(image_path: str, df_meta: pd.DataFrame) -> int | float:
    """
    Match an image path to its LUS score in the metadata CSV.
    
    Handles different filename formats for each hospital:
    - JCUH/MFT: ID_ScanNo_ScanLabel_...
    - UHW: video_id_selected_frame...
    
    Args:
        image_path: Full path to the image file
        df_meta: Metadata DataFrame with Score column
    
    Returns:
        Integer score if found, float('nan') otherwise
    """
    # Precompute video_id column if it is missing
    if 'video_id' not in df_meta.columns:
        df_meta = df_meta.copy()
        df_meta['video_id'] = df_meta['File Path'].apply(lambda p: Path(str(p)).stem)

    score = float('nan')

    try:
        # First attempt: match on video_id derived from the provided string
        basename = Path(str(image_path)).stem
        candidate_ids = [basename]
        
        # Handle _selected_frame suffix
        if '_selected_frame' in basename:
            candidate_ids.append(basename.split('_selected_frame')[0])
        
        # Handle hospital prefix (new format: HOSPITAL_baseId)
        for h in ['JCUH', 'MFT', 'UHW']:
            if basename.startswith(f"{h}_"):
                # Strip hospital prefix for metadata matching
                base_without_hospital = basename[len(h)+1:]
                candidate_ids.append(base_without_hospital)
                # Also try without frame suffix
                if '_selected_frame' in base_without_hospital:
                    candidate_ids.append(base_without_hospital.split('_selected_frame')[0])
                break
        
        # Handle trailing digits
        parts = basename.split('_')
        if parts and parts[-1].isdigit():
            candidate_ids.append('_'.join(parts[:-1]))

        for vid in candidate_ids:
            match = df_meta[df_meta['video_id'] == vid]
            if not match.empty:
                return int(float(match.iloc[0]['Score']))

        # Fallback to legacy hospital-based parsing for frame-level paths
        hospital = ""
        if 'JCUH' in image_path:
            hospital = 'JCUH'
        elif 'MFT' in image_path:
            hospital = 'MFT'
        elif 'UHW' in image_path:
            hospital = 'UHW'

        if hospital in ('JCUH', 'MFT'):
            parts = image_path.split('/')[-1].split('_')
            
            # Check if first part is the hospital prefix (new format: HOSPITAL_ID_...)
            # If so, skip it to get the actual patient ID
            start_idx = 0
            if parts[0] in ['JCUH', 'MFT', 'UHW']:
                start_idx = 1
            
            # Ensure we have enough parts after skipping hospital prefix
            if len(parts) < start_idx + 4:
                print(f'DEBUG: Invalid format for {image_path} - not enough parts')
            else:
                patient_id = int(float(parts[start_idx]))
                scan_no = f"{parts[start_idx + 1]}_{parts[start_idx + 2]}"
                scan_label = parts[start_idx + 3]

                row = df_meta[
                    (df_meta['Hospital'] == hospital) &
                    (df_meta['Patient ID'] == int(patient_id)) &
                    (df_meta['Scan No'].astype(str) == str(scan_no)) &
                    (df_meta['Scan Label'] == scan_label)
                ]

                if not row.empty:
                    score = int(float(row['Score'].values[0]))
                else:
                    print(f'DEBUG: No match found for {image_path}')
                    print(f"  Hospital: {hospital}, Patient ID: {patient_id}, "
                          f"Scan No: {scan_no}, Scan Label: {scan_label}")

        elif hospital == 'UHW':
            filename = image_path.split('/')[-1]
            video_id = filename.split('_selected_frame')[0]

            matching_rows = df_meta[df_meta['File Path'].str.contains(video_id, na=False)]
            if not matching_rows.empty:
                score = int(float(matching_rows['Score'].values[0]))
            else:
                print(f"DEBUG: No UHW match found for video_id: {video_id}")
        else:
            print(f"DEBUG: No hospital detected in path: {image_path}")

    except Exception as e:
        print(f"DEBUG: Exception for {image_path}: {e}")

    return score


def get_hospital_from_path(image_path: str) -> Optional[str]:
    """
    Extract hospital identifier from image path.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Hospital code ('JCUH', 'MFT', 'UHW') or None if not detected
    """
    for hospital in Config.HOSPITALS:
        if hospital in image_path:
            return hospital
    return None
