import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_shap_summary(model, X_sample, feature_names, output_path):
    """
    Generate SHAP summary plot showing feature importance.
    
    Args:
        model: Trained LightGBM model
        X_sample: Sample of features (use last 1000 predictions)
        feature_names: List of feature column names
        output_path: Where to save the plot
    """
    print(f"  Calculating SHAP values for {len(X_sample)} samples...")

    # Create SHAP explainer for tree-based models
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values (how much each feature contributed to predictions)
    shap_values = explainer.shap_values(X_sample)
    
    # Convert to DataFrame for easier handling
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    
    print(f"SHAP values calculated")

    # Create summary plot (shows feature importance + impact direction)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, 
                      feature_names=feature_names,
                      show=False,
                      max_display=12)  # Show top 12 features
    
    plt.title('SHAP Feature Importance Summary', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved SHAP summary plot to {output_path}")

def generate_feature_importance_chart(model, feature_names, output_path):
    """
    Generate bar chart of raw feature importance from LightGBM.
    Complements SHAP with model's internal importance scores.
    """
    # Get feature importance from trained model
    importance = model.feature_importances_
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    # Plot horizontal bar chart
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'], importance_df['importance'], color='#06A77D', alpha=0.8)
    plt.xlabel('Feature Importance (Gain)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('LightGBM Feature Importance (Gain)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved feature importance chart to {output_path}")


