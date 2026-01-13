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
    
    # Ensure numerical columns are actually numbers
    cluster_df['tsne_x'] = pd.to_numeric(cluster_df['tsne_x'])
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
    # Detect hospital from path
    hospital = ""
    if 'JCUH' in image_path:
        hospital = 'JCUH'
    elif 'MFT' in image_path:
        hospital = 'MFT'
    elif 'UHW' in image_path:
        hospital = 'UHW'
    
    score = float('nan')
    
    try:
        if hospital in ('JCUH', 'MFT'):
            # Parse filename: ID_ScanNo_ScanLabel_...
            parts = image_path.split('/')[-1].split('_')
            patient_id = int(float(parts[0]))
            scan_no = f"{parts[1]}_{parts[2]}"
            scan_label = parts[3]
            
            # Filter metadata
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
            # Parse filename for UHW
            filename = image_path.split('/')[-1]
            video_id = filename.split('_selected_frame')[0]
            
            # Search for video ID in path column
            matching_rows = df_meta[df_meta['File Path'].str.contains(video_id, na=False)]
            if not matching_rows.empty:
                score = int(float(matching_rows['Score'].values[0]))
            else:
                print(f"DEBUG: No UHW match found for video_id: {video_id}")
        else:
            print(f"DEBUG: No hospital detected in path: {image_path}")
    
    except Exception as e:
        # If parsing fails, just return NaN
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
