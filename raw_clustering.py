# Perform K-Means clustering on a given dataset and visualize the results.


import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from plotly.subplots import make_subplots

DATA_PATH = '/cosma5/data/durham/dc-fras4/ultrasound/output_frames/for_vae/all_images'
OUTPUT_HTML = "results/interactive_kmeans_clustering.html"

def load_images(data_path):
    


