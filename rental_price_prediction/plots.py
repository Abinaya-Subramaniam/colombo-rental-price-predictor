import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from .config import *

def setup_plot_style():
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)

def plot_feature_distributions(df, numerical_features, categorical_features):
    setup_plot_style()
    
    df[numerical_features].hist(figsize=(15, 10), bins=20, edgecolor='black')
    plt.suptitle("Distribution of Numerical Features")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "figures/numerical_distributions.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    for col in categorical_features:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f"Distribution of {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / f"figures/{col}_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()

def plot_correlation_matrix(df, numerical_features):
    setup_plot_style()
    
    corr = df[numerical_features + ['Rental_Price']].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", center=0)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "figures/correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_model_comparison(results):
    setup_plot_style()
    
    model_names = list(results.keys())
    r2_scores = [results[name]['r2'] for name in model_names]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].bar(model_names, r2_scores, color='skyblue')
    axes[0].set_title('R² Scores by Model')
    axes[0].set_ylabel('R² Score')
    axes[0].tick_params(axis='x', rotation=45)
    
    mae_scores = [results[name]['mae'] for name in model_names]
    axes[1].bar(model_names, mae_scores, color='lightcoral')
    axes[1].set_title('MAE by Model')
    axes[1].set_ylabel('MAE')
    axes[1].tick_params(axis='x', rotation=45)
    
    rmse_scores = [results[name]['rmse'] for name in model_names]
    axes[2].bar(model_names, rmse_scores, color='lightgreen')
    axes[2].set_title('RMSE by Model')
    axes[2].set_ylabel('RMSE')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "figures/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()