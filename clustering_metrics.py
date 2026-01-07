from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import mutual_info_score, adjusted_rand_score
import os
import pandas as pd
import json


CLUSTER_TABLE_PATH = "/cosma/home/durham/dc-fras4/code/wandb_export_2026-01-05T18_00_39.610+00_00.csv"

# Path to your metadata CSV
METADATA_PATH = "/cosma/home/durham/dc-fras4/code/data_preprocessing/data_tables/all_data.csv"

def load_data():
    print("Loading data...")
    # 1. Load Metadata
    df_metadata = pd.read_csv(METADATA_PATH)
    

    # 2. Load WandB Cluster Table (JSON or CSV)
    file_ext = os.path.splitext(CLUSTER_TABLE_PATH)[1].lower()
    
    if file_ext == '.json':
        print(f"Loading cluster data from JSON: {CLUSTER_TABLE_PATH}")
        with open(CLUSTER_TABLE_PATH, 'r') as f:
            data = json.load(f)
        cluster_df = pd.DataFrame(data['data'], columns=data['columns'])
    elif file_ext == '.csv':
        print(f"Loading cluster data from CSV: {CLUSTER_TABLE_PATH}")
        cluster_df = pd.read_csv(CLUSTER_TABLE_PATH)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Expected .json or .csv")
    

    # Ensure numerical columns are actually numbers
    cluster_df['tsne_x'] = pd.to_numeric(cluster_df['tsne_x'])
    cluster_df['tsne_y'] = pd.to_numeric(cluster_df['tsne_y'])
    
    return cluster_df, df_metadata

def find_score(image_path, df_meta):
    """
    Matches an image path to its score in the metadata CSV.
    """
    hospital = ""
    if 'JCUH' in image_path: hospital = 'JCUH' 
    elif 'MFT' in image_path: hospital = 'MFT'
    elif 'UHW' in image_path: hospital = 'UHW'
    
    score = float('nan')

    try:
        if hospital == 'JCUH' or hospital == 'MFT':  
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
                print(f"  Hospital: {hospital}, Patient ID: {patient_id}, Scan No: {scan_no}, Scan Label: {scan_label}")

        
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

def main():
    cluster_df, df_metadata = load_data()
    
    # Map image paths to scores
    print("Mapping image paths to scores...")
    cluster_df['Score'] = cluster_df['image_path'].apply(lambda path: find_score(path, df_metadata))
    
    # Drop rows with NaN scores
    valid_clusters = cluster_df.dropna(subset=['Score'])
    
    # Extract necessary columns
    X = valid_clusters[['tsne_x', 'tsne_y']].values
    labels = valid_clusters['cluster_label'].values
    scores = valid_clusters['Score'].values
    
    print("Calculating clustering metrics...")
    silhouette_avg = silhouette_score(X, labels)
    davies_bouldin_avg = davies_bouldin_score(X, labels)
    calinski_harabasz_avg = calinski_harabasz_score(X, labels)
    mi_score = mutual_info_score(labels, scores)
    ari_score = adjusted_rand_score(labels, scores)
    
    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin_avg}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz_avg}")
    print(f"Mutual Information Score: {mi_score}")
    print(f"Adjusted Rand Index: {ari_score}")

if __name__ == "__main__":
    main()
