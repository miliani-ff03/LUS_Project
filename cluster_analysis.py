import os
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================= CONFIGURATION =================
# Path to your downloaded WandB table JSON
# Update this to the specific JSON file you want to analyze
CLUSTER_TABLE_PATH = "/cosma/home/durham/dc-fras4/code/wandb/offline-run-20251203_172743-d85bnu5d/files/media/table/cluster_labels_10_f65bf656f57b62fb5312.table.json"

# Path to your metadata CSV
METADATA_PATH = "/cosma/home/durham/dc-fras4/code/data_preprocessing/data_tables/all_data.csv"

# Output filename for the interactive plot
OUTPUT_HTML = "results/interactive_tsne.html"
# =================================================

def load_data():
    print("Loading data...")
    # 1. Load Metadata
    df_metadata = pd.read_csv(METADATA_PATH)
    

    # 2. Load WandB Cluster Table
    with open(CLUSTER_TABLE_PATH, 'r') as f:
        data = json.load(f)
    
    # Reconstruct DataFrame from WandB JSON format
    cluster_df = pd.DataFrame(data['data'], columns=data['columns'])
    
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
    # 1. Load Data
    cluster_df, df_metadata = load_data()
    print(f"Loaded {len(cluster_df)} points from cluster table.")

    # 2. Match Scores
    print("Matching scores to images...")
    cluster_df['Score'] = cluster_df['image_path'].apply(lambda x: find_score(x, df_metadata))
    cluster_df['Score'] = cluster_df['Score'].astype('Int64')  # Int64 handles NaN values

    # Clean up image path for display (just show filename and directory before for hospital)

    cluster_df['filename'] = cluster_df['image_path'].apply(lambda x: os.path.join(os.path.basename(os.path.dirname(x)), os.path.basename(x)))
    
    # Ensure Cluster is treated as a category (string) for better coloring
    cluster_df['cluster_label'] = cluster_df['cluster_label'].astype(str)

    # 3. Generate Interactive Plot with Plotly
    print("Generating interactive HTML plot...")
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Interactive t-SNE Analysis", "Score Distribution by Cluster"),
        specs=[[{"type": "scatter"}, {"type": "bar"}]],
        column_widths=[0.6, 0.4]
    )
    scatter = px.scatter(
        cluster_df, 
        x='tsne_x', 
        y='tsne_y', 
        color='cluster_label',
        
        # What shows up when you hover:
        hover_data={
            'tsne_x': False,    # Hide coordinates to keep it clean
            'tsne_y': False,
            'cluster_label': True,
            'Score': True,
            'filename': True
        },
        template="plotly_dark", # Optional: looks cool, remove if you prefer white background
    )

    # Customize the layout
    for trace in scatter.data:
        fig.add_trace(trace, row=1, col=1)
    
        # Add bar chart to second subplot
    score_counts = cluster_df.groupby(['Score', 'cluster_label']).size().reset_index(name='Count')

    
    for score in sorted(score_counts['Score'].unique()):
        score_data = score_counts[score_counts['Score'] == score]
        fig.add_trace(go.Bar(
            x=score_data['cluster_label'],
            y=score_data['Count'],
            name=f'Score {score}',
            hovertemplate='<b>Cluster: %{x}</b><br>Count: %{y}<extra></extra>'
        ), row=1, col=2)
    
    # Update layout
    fig.update_xaxes(title_text="t-SNE Dimension 1", row=1, col=1)
    fig.update_yaxes(title_text="t-SNE Dimension 2", row=1, col=1)
    fig.update_xaxes(title_text="Cluster", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    fig.update_layout(
        height=600,
        width=1400,
        title_text="Cluster Analysis Dashboard",
        template="plotly_dark",
        barmode='group'
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8), selector=dict(mode='markers'))
    # 4. Save to HTML
    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    fig.write_html(OUTPUT_HTML)
    
    print(f"Success! Plot saved to: {OUTPUT_HTML}")
    print("Download this file to your local machine and open it in Chrome/Safari/Edge.")

if __name__ == "__main__":
    main()