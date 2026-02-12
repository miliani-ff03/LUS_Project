"""
Frame-Level Cluster Analysis with Interactive t-SNE Visualization.

Generates interactive Plotly dashboards showing t-SNE embeddings
colored by cluster labels and LUS scores at the frame level.

Usage:
    python -m frame_level.cluster_analysis
    python frame_level/cluster_analysis.py --cluster-table results/supervised/clustering_plots/cluster_table_ld32_beta2.0_crop10_k3.csv
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.config import Config
from shared.utils import find_score

# ==========================================
# CONFIGURATION
# ==========================================

MODULE_DIR = Path(__file__).parent.resolve()  # ✅ CHANGED: Added .resolve()
RESULTS_DIR = MODULE_DIR / "results"
CLUSTERING_PLOTS_DIR = RESULTS_DIR / "supervised" / "clustering_plots"

# ✅ CHANGED: Check what files actually exist
print(f"DEBUG: MODULE_DIR = {MODULE_DIR}")
print(f"DEBUG: CLUSTERING_PLOTS_DIR = {CLUSTERING_PLOTS_DIR}")
if CLUSTERING_PLOTS_DIR.exists():
    print(f"DEBUG: Available files in clustering_plots/:")
    for f in sorted(CLUSTERING_PLOTS_DIR.glob("*")):
        print(f"  - {f.name}")
else:
    print(f"DEBUG: Directory does not exist: {CLUSTERING_PLOTS_DIR}")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Default paths - use first available CSV file
DEFAULT_CLUSTER_TABLE = None
if CLUSTERING_PLOTS_DIR.exists():
    csv_files = list(CLUSTERING_PLOTS_DIR.glob("*.csv"))
    if csv_files:
        DEFAULT_CLUSTER_TABLE = str(csv_files[0])  # ✅ CHANGED: Use first found CSV
        print(f"DEBUG: Using default table: {DEFAULT_CLUSTER_TABLE}")
    else:
        print("WARNING: No CSV files found in clustering_plots/")
        DEFAULT_CLUSTER_TABLE = str(CLUSTERING_PLOTS_DIR / "cluster_table_ld32_beta2.0_crop10_k3.csv")
else:
    DEFAULT_CLUSTER_TABLE = str(CLUSTERING_PLOTS_DIR / "cluster_table_ld32_beta2.0_crop10_k3.csv")

DEFAULT_OUTPUT_HTML = str(RESULTS_DIR / "interactive_tsne_frame_level.html")

def load_cluster_data_local(table_path):
    """
    Load cluster table from CSV or JSON format.
    Handles both WandB exported tables and locally saved tables.
    
    Args:
        table_path: Path to cluster table file (relative to frame_level/ or absolute)
        
    Returns:
        tuple: (cluster_df, metadata_df)
    """
    table_path = Path(table_path)
    
    # ✅ CHANGED: Better path resolution with debug output
    if not table_path.is_absolute():
        table_path = MODULE_DIR / table_path
        print(f"DEBUG: Resolved relative path to: {table_path}")
    
    if not table_path.exists():
        # ✅ IMPROVED: More helpful error message
        parent_dir = table_path.parent
        available = list(parent_dir.glob("*.csv")) + list(parent_dir.glob("*.json")) if parent_dir.exists() else []
        
        error_msg = f"Cluster table not found: {table_path}\n"
        error_msg += f"  Searched in: {parent_dir}\n"
        if available:
            error_msg += f"  Available files:\n"
            for f in available:
                error_msg += f"    - {f.name}\n"
        else:
            error_msg += f"  Directory exists: {parent_dir.exists()}\n"
        
        raise FileNotFoundError(error_msg)
    
    # Load based on extension
    print(f"Loading {table_path}...")
    if table_path.suffix == '.json':
        cluster_df = pd.read_json(table_path)
    elif table_path.suffix == '.csv':
        cluster_df = pd.read_csv(table_path)
    else:
        raise ValueError(f"Unsupported file format: {table_path.suffix}. Use .csv or .json")
    
    print(f"✓ Loaded {len(cluster_df)} rows from {table_path.name}")
    
    # Validate required columns
    required_cols = ['tsne_x', 'tsne_y', 'cluster_label', 'image_path']
    missing_cols = [col for col in required_cols if col not in cluster_df.columns]
    if missing_cols:
        print(f"Available columns: {list(cluster_df.columns)}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Load metadata
    metadata_path = Path(__file__).parent.parent / "data_preprocessing" / "data_tables" / "all_data.csv"
    if not metadata_path.exists():
        print(f"Warning: Metadata not found at {metadata_path}")
        df_metadata = pd.DataFrame()
    else:
        df_metadata = pd.read_csv(metadata_path)
        print(f"✓ Loaded metadata with {len(df_metadata)} entries")
    
    return cluster_df, df_metadata


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate interactive frame-level t-SNE visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default (first found CSV in clustering_plots/)
  python frame_level/cluster_analysis.py
  
  # Specify table with relative path
  python frame_level/cluster_analysis.py -c results/supervised/clustering_plots/cluster_table_ld32_beta2.0_crop10_k4.csv
  
  # Auto-generate output filename
  python frame_level/cluster_analysis.py -c results/supervised/clustering_plots/cluster_table_ld32_beta2.0_crop10_k2.csv --auto-output
        """
    )
    parser.add_argument(
        "--cluster-table", "-c",
        default=DEFAULT_CLUSTER_TABLE,
        help="Path to cluster table (CSV or JSON) - relative to frame_level/ or absolute"
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_HTML,
        help="Output path for HTML visualization - relative to frame_level/ or absolute"
    )
    parser.add_argument(
        "--auto-output", "-a",
        action="store_true",
        help="Auto-generate output filename based on input table name"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # ✅ CHANGED: Check if default is None
    if args.cluster_table is None:
        print("\nERROR: No cluster table specified and no default found.")
        print(f"Please create CSV files in: {CLUSTERING_PLOTS_DIR}")
        print("\nOr specify a table with: -c path/to/table.csv")
        sys.exit(1)
    
    # Load data using local function (handles relative paths)
    cluster_df, df_metadata = load_cluster_data_local(args.cluster_table)
    
    # Auto-generate output filename if requested
    if args.auto_output:
        input_stem = Path(args.cluster_table).stem
        args.output = str(RESULTS_DIR / f"interactive_{input_stem}.html")
        print(f"Auto-generated output path: {args.output}")
    
    # Ensure output is in frame_level directory
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = MODULE_DIR / output_path

    # Detect data level
    data_level = cluster_df.get('_data_level', 'frame').iloc[0] if '_data_level' in cluster_df.columns else 'frame'
    print(f"Data level: {data_level.upper()}")

    # Match scores
    print("Matching scores to data points...")
    if not df_metadata.empty:
        cluster_df['Score'] = cluster_df['image_path'].apply(lambda x: find_score(x, df_metadata))
        cluster_df['Score'] = cluster_df['Score'].astype('Int64')
    else:
        print("Warning: No metadata available, scores will be missing")
        cluster_df['Score'] = pd.NA
    
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
    
    # Bar chart for score distribution (only if scores exist)
    if not cluster_df['Score'].isna().all():
        score_counts = cluster_df.groupby(['Score', 'cluster_label']).size().reset_index(name='Count')
        blue_colors = ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7']
        
        for idx, score in enumerate(sorted(score_counts['Score'].unique())):
            if pd.isna(score):
                continue
            score_data = score_counts[score_counts['Score'] == score]
            fig.add_trace(go.Bar(
                x=score_data['cluster_label'],
                y=score_data['Count'],
                name=f'Score {int(score)}',
                marker=dict(color=blue_colors[idx % len(blue_colors)]),
                hovertemplate='<b>Cluster: %{x}</b><br>Count: %{y}<extra></extra>'
            ), row=1, col=2)
    
    # Update layout
    fig.update_xaxes(title_text="t-SNE Dim 1", title_font=dict(size=18), tickfont=dict(size=14), row=1, col=1)
    fig.update_yaxes(title_text="t-SNE Dim 2", title_font=dict(size=18), tickfont=dict(size=14), row=1, col=1)
    fig.update_xaxes(title_text="Cluster", title_font=dict(size=18), tickfont=dict(size=14), row=1, col=2)
    fig.update_yaxes(title_text="Count", title_font=dict(size=18), tickfont=dict(size=14), row=1, col=2)

    input_name = Path(args.cluster_table).stem
    fig.update_layout(
        height=600,
        width=1400,
        title_text=f"Frame-Level Cluster Analysis: {input_name}",
        barmode='group',
        legend=dict(font=dict(size=16))
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8), selector=dict(mode='markers'))
    
    # Save to HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    
    print(f"\n✓ Success! Plot saved to: {output_path}")
    print(f"  Relative to frame_level/: {output_path.relative_to(MODULE_DIR)}")
    print("\nDownload this file to your local machine and open it in a browser.")


if __name__ == "__main__":
    main()