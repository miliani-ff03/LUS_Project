"""
Cluster Analysis with Interactive t-SNE Visualization.

Generates interactive Plotly dashboards showing t-SNE embeddings
colored by cluster labels and LUS scores.

Usage:
    python cluster_analysis.py
    
    Or with custom paths:
    python cluster_analysis.py --cluster-table path/to/export.csv --output path/to/output.html
"""

import os
import argparse

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import Config
from utils import load_cluster_data, find_score


# ================= DEFAULT CONFIGURATION =================
# These can be overridden via command line arguments
DEFAULT_CLUSTER_TABLE = "/cosma/home/durham/dc-fras4/code/wandb_export_2026-01-20T11_18_50.297+00_00.csv"
DEFAULT_OUTPUT_HTML = "results/interactive_tsne_ld_23_beta_2_crop_10_k_4.html"
# =========================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate interactive t-SNE cluster visualization")
    parser.add_argument(
        "--cluster-table", "-c",
        default=DEFAULT_CLUSTER_TABLE,
        help="Path to WandB exported cluster table (CSV or JSON)"
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_HTML,
        help="Output path for HTML visualization"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 1. Load Data using shared utility
    cluster_df, df_metadata = load_cluster_data(args.cluster_table)
    print(f"Loaded {len(cluster_df)} points from cluster table.")

    # 2. Match Scores using shared utility
    print("Matching scores to images...")
    cluster_df['Score'] = cluster_df['image_path'].apply(lambda x: find_score(x, df_metadata))
    cluster_df['Score'] = cluster_df['Score'].astype('Int64')  # Int64 handles NaN values

    # Clean up image path for display (just show filename and directory before for hospital)
    cluster_df['filename'] = cluster_df['image_path'].apply(
        lambda x: os.path.join(os.path.basename(os.path.dirname(x)), os.path.basename(x))
    )
    
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
    
    # Diverse color palette for clusters
    diverse_colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#e377c2', '#bcbd22']
    cluster_color_map = {
        str(cluster): diverse_colors[i % len(diverse_colors)] 
        for i, cluster in enumerate(sorted(cluster_df['cluster_label'].unique()))
    }
    
    scatter = px.scatter(
        cluster_df, 
        x='tsne_x', 
        y='tsne_y', 
        color='cluster_label',
        color_discrete_map=cluster_color_map,
        hover_data={
            'tsne_x': False,
            'tsne_y': False,
            'cluster_label': True,
            'Score': True,
            'filename': True
        }
    )

    # Customize the layout
    for trace in scatter.data:
        fig.add_trace(trace, row=1, col=1)
    
    # Add bar chart to second subplot
    score_counts = cluster_df.groupby(['Score', 'cluster_label']).size().reset_index(name='Count')

    # Blue color palette for bars
    blue_colors = ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7']
    
    for idx, score in enumerate(sorted(score_counts['Score'].unique())):
        score_data = score_counts[score_counts['Score'] == score]
        fig.add_trace(go.Bar(
            x=score_data['cluster_label'],
            y=score_data['Count'],
            name=f'Score {score}',
            marker=dict(color=blue_colors[idx % len(blue_colors)]),
            hovertemplate='<b>Cluster: %{x}</b><br>Count: %{y}<extra></extra>'
        ), row=1, col=2)
    
    # Update layout
    fig.update_xaxes(title_text="t-SNE Dimension 1", title_font=dict(size=18), tickfont=dict(size=14), row=1, col=1)
    fig.update_yaxes(title_text="t-SNE Dimension 2", title_font=dict(size=18), tickfont=dict(size=14), row=1, col=1)
    fig.update_xaxes(title_text="Cluster", title_font=dict(size=18), tickfont=dict(size=14), row=1, col=2)
    fig.update_yaxes(title_text="Count", title_font=dict(size=18), tickfont=dict(size=14), row=1, col=2)

    fig.update_layout(
        height=600,
        width=1400,
        title_text=f"Cluster Analysis Dashboard ({os.path.splitext(os.path.basename(args.output))[0].split('_', 1)[-1].replace('_', ' ')})",
        barmode='group',
        legend=dict(font=dict(size=16))
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8), selector=dict(mode='markers'))
    
    # 4. Save to HTML
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.write_html(args.output)
    
    print(f"Success! Plot saved to: {args.output}")
    print("Download this file to your local machine and open it in Chrome/Safari/Edge.")


if __name__ == "__main__":
    main()
