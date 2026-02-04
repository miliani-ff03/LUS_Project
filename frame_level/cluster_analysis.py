"""
Frame-Level Cluster Analysis with Interactive t-SNE Visualization.

Generates interactive Plotly dashboards showing t-SNE embeddings
colored by cluster labels and LUS scores at the frame level.

Usage:
    python -m frame_level.cluster_analysis
    python frame_level/cluster_analysis.py --cluster-table path/to/table.json --output results/plot.html
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.config import Config
from shared.utils import load_cluster_data, find_score

# ==========================================
# CONFIGURATION
# ==========================================

MODULE_DIR = Path(__file__).parent
RESULTS_DIR = MODULE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Default paths
DEFAULT_CLUSTER_TABLE = str(MODULE_DIR.parent / "wandb" / "latest-run" / "files" / "cluster_table.json")
DEFAULT_OUTPUT_HTML = str(RESULTS_DIR / "interactive_tsne_frame_level.html")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate interactive frame-level t-SNE visualization")
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
    
    # Load data using shared utility
    cluster_df, df_metadata = load_cluster_data(args.cluster_table)
    print(f"Loaded {len(cluster_df)} points from cluster table.")
    
    # Detect data level
    data_level = cluster_df.get('_data_level', 'frame').iloc[0] if '_data_level' in cluster_df.columns else 'frame'
    print(f"Data level: {data_level.upper()}")

    # Match scores
    print("Matching scores to data points...")
    cluster_df['Score'] = cluster_df['image_path'].apply(lambda x: find_score(x, df_metadata))
    cluster_df['Score'] = cluster_df['Score'].astype('Int64')
    
    # Match hospital from metadata
    def get_hospital(identifier, df_meta):
        """Extract hospital from image path."""
        for h in ['JCUH', 'MFT', 'UHW']:
            if h in str(identifier):
                return h
        return 'Unknown'
    
    cluster_df['Hospital'] = cluster_df['image_path'].apply(lambda x: get_hospital(x, df_metadata))

    # Create display identifier (directory/filename)
    cluster_df['display_id'] = cluster_df['image_path'].apply(
        lambda x: os.path.join(os.path.basename(os.path.dirname(x)), os.path.basename(x))
    )
    
    # Ensure cluster is string for coloring
    cluster_df['cluster_label'] = cluster_df['cluster_label'].astype(str)

    # Generate interactive plot
    print("Generating interactive HTML plot...")
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Frame-Level t-SNE Clustering", "Score Distribution by Cluster"),
        specs=[[{"type": "scatter"}, {"type": "bar"}]],
        column_widths=[0.6, 0.4]
    )
    
    # Color palette for clusters
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
            'Hospital': True,
            'display_id': True
        }
    )

    for trace in scatter.data:
        fig.add_trace(trace, row=1, col=1)
    
    # Bar chart for score distribution
    score_counts = cluster_df.groupby(['Score', 'cluster_label']).size().reset_index(name='Count')
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
    fig.update_xaxes(title_text="t-SNE Dim 1", title_font=dict(size=18), tickfont=dict(size=14), row=1, col=1)
    fig.update_yaxes(title_text="t-SNE Dim 2", title_font=dict(size=18), tickfont=dict(size=14), row=1, col=1)
    fig.update_xaxes(title_text="Cluster", title_font=dict(size=18), tickfont=dict(size=14), row=1, col=2)
    fig.update_yaxes(title_text="Count", title_font=dict(size=18), tickfont=dict(size=14), row=1, col=2)

    fig.update_layout(
        height=600,
        width=1400,
        title_text="Frame-Level Cluster Analysis Dashboard",
        barmode='group',
        legend=dict(font=dict(size=16))
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8), selector=dict(mode='markers'))
    
    # Save to HTML
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.write_html(args.output)
    
    print(f"Success! Plot saved to: {args.output}")
    print("Download this file to your local machine and open it in a browser.")


if __name__ == "__main__":
    main()
