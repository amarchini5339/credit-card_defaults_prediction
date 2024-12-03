import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# Helper function for SHAP summary and rankings
def shap_summary(model, features):
    # Create a SHAP explainer
    explainer = shap.Explainer(model, features)

    # Compute SHAP values
    shap_values = explainer(features)

    # Combine SHAP feature importance dataframe
    shap_importance = pd.DataFrame({
        'Feature': features.columns,
        'SHAP Importance': np.abs(shap_values.values).mean(axis=0),
    }).sort_values(by='SHAP Importance', ascending=False)

    # Display the shap importance
    print(shap_importance)

    # SHAP summary plot
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, features)
    
    return shap_importance

def shap_interactions(model, features):
    explainer = shap.TreeExplainer(model)
    shap_interaction_values = explainer.shap_interaction_values(features)
    shap.summary_plot(shap_interaction_values, features, plot_type="dot")
    return shap_interaction_values

def shap_dependence_plot(features_tuple, shap_interaction_values, features):
    shap.dependence_plot(features_tuple, shap_interaction_values, features)
