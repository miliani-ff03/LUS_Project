# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
df_kl = pd.read_csv('kl_loss_points.csv')
df_loss = pd.read_csv('loss_points.csv')
df_recon = pd.read_csv('recon_loss_points.csv')

# 2. Helper function to clean and reshape the data for Seaborn
def process_df(df, value_name):
    # Remove columns that contain 'MIN' or 'MAX'
    cols_to_remove = [col for col in df.columns if 'MAX' in col or 'MIN' in col]
    df_clean = df.drop(columns=cols_to_remove)
    
    # Rename columns to Beta values (mapping from your script)
    rename_map = {}
    for col in df_clean.columns:
        if 'clear-butterfly-13' in col:
            rename_map[col] = 'Beta = 5'
        elif 'ruby-dream-12' in col:
            rename_map[col] = 'Beta = 2'
        elif 'fanciful-haze-11' in col:
            rename_map[col] = 'Beta = 1'
        elif 'eager-dream-10' in col:
            rename_map[col] = 'Beta = 0.1'
    
    # Apply renaming
    df_renamed = df_clean.rename(columns=rename_map)
    
    # "Melt" the dataframe to long format for Seaborn
    # (keeps 'Step' as is, and stacks all Beta columns into one 'Beta' column)
    value_vars = [c for c in df_renamed.columns if c != 'Step']
    df_long = df_renamed.melt(id_vars=['Step'], value_vars=value_vars, 
                              var_name='Beta', value_name=value_name)
    return df_long

# 3. Process all three dataframes
df_kl_long = process_df(df_kl, 'KL Loss')
df_loss_long = process_df(df_loss, 'Total Loss')
df_recon_long = process_df(df_recon, 'Reconstruction Loss')
# %%
# 4. Create the plot
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(3, 1, figsize=(16, 18))

# Consistent order and color palette
hue_order = ['Beta = 5', 'Beta = 2', 'Beta = 1', 'Beta = 0.1']
palette = "viridis" 

# make fonts larger
plt.rcParams.update({'font.size': 20,'axes.labelsize': 22,   # axis labels
    'xtick.labelsize': 18,  # tick labels
    'ytick.labelsize': 18,'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Palatino']})

# Subplot 1: KL Loss
sns.lineplot(data=df_kl_long, x='Step', y='KL Loss', hue='Beta', 
             ax=axes[2], hue_order=hue_order, palette=palette, linewidth=3.0)
axes[2].set_xlabel('Step', fontsize=22)
axes[2].set_ylabel('KL Loss', fontsize=22)


# Subplot 2: Total Loss
sns.lineplot(data=df_loss_long, x='Step', y='Total Loss', hue='Beta', 
             ax=axes[0], hue_order=hue_order, palette=palette, linewidth=3.0)
axes[0].set_xlabel('')
axes[0].set_ylabel('Total Loss', fontsize=22)


# Subplot 3: Reconstruction Loss
sns.lineplot(data=df_recon_long, x='Step', y='Reconstruction Loss', hue='Beta', 
             ax=axes[1], hue_order=hue_order, palette=palette,linewidth=3.0)
axes[1].set_xlabel('')
axes[1].set_ylabel('Reconstruction Loss', fontsize=22)

for ax in axes:
    ax.tick_params(axis='both', labelsize=18)
# remove legends from individual plots
for ax in axes:
    ax.legend_.remove()

plt.tight_layout()
plt.savefig('combined_loss_plot.png', dpi=300)
plt.show()
# %%
