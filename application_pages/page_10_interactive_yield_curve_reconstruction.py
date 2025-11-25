"""Page 10: Interactive Yield Curve Reconstruction"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100

def main():
    st.title("Interactive Yield Curve Reconstruction")

    st.markdown("""
    ### Interactive Yield Curve Reconstruction

    To further explore the impact of each principal component, we will create an interactive tool that allows you to reconstruct a yield curve using a chosen number of principal components (1, 2, or 3). This interactive feature will dynamically update the reconstructed curve, illustrating how adding each component refines the approximation of the original curve and captures more of its variance. You will observe how the 'level' component forms a basic curve, 'slope' adjusts its steepness, and 'curvature' fine-tunes its convexity.
    """)

    if ('yield_data' not in st.session_state or 
        'transformed_yield_data' not in st.session_state or 
        'sorted_eigenvectors' not in st.session_state or 
        'mean_yields' not in st.session_state or
        'maturities' not in st.session_state or
        'reconstruct_yield_curve' not in st.session_state):
        st.error("Required data for interactive reconstruction not found in session state. Please go through previous steps.")
        return
    
    yield_data = st.session_state.yield_data
    transformed_yield_data = st.session_state.transformed_yield_data
    sorted_eigenvectors = st.session_state.sorted_eigenvectors
    mean_yields = st.session_state.mean_yields
    maturities = st.session_state.maturities
    reconstruct_yield_curve = st.session_state.reconstruct_yield_curve

    # Sidebar control for interactive reconstruction day
    day_options = yield_data.index.strftime('%Y-%m-%d').tolist()
    if 'selected_reconstruction_day_idx' not in st.session_state:
        st.session_state.selected_reconstruction_day_idx = 0 # Default to the first day

    selected_day_str = st.sidebar.selectbox(
        "Select Day for Interactive Reconstruction:",
        options=day_options,
        index=st.session_state.selected_reconstruction_day_idx,
        key="interactive_day_selector"
    )
    day_to_reconstruct_index = day_options.index(selected_day_str)
    st.session_state.selected_reconstruction_day_idx = day_to_reconstruct_index

    original_curve_selected_day = yield_data.iloc[day_to_reconstruct_index]

    num_components_selected = st.slider(
        'Select Number of Principal Components for Reconstruction:',
        min_value=1, max_value=3, value=3, step=1,
        key="num_pcs_slider" # Ensure a unique key for the slider
    )

    reconstructed_curve_full_df = reconstruct_yield_curve(
        transformed_yield_data, sorted_eigenvectors, mean_yields, num_components_selected
    )
    reconstructed_curve_selected_day = reconstructed_curve_full_df.iloc[day_to_reconstruct_index]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(maturities, original_curve_selected_day, marker='o', label='Original Yield Curve', color='blue')
    ax.plot(maturities, reconstructed_curve_selected_day, marker='x', linestyle='--', color='red',
             label=f'Reconstructed ({num_components_selected} PCs)')
    ax.set_title(f'Yield Curve Reconstruction (Using {num_components_selected} Principal Components) for {original_data.index[day_to_reconstruct_index].strftime('%Y-%m-%d')}')
    ax.set_xlabel('Maturity (Years)')
    ax.set_ylabel('Yield (%)')
    ax.set_xticks(maturities)
    ax.grid(True)
    ax.legend()
    ax.set_ylim(yield_data.min().min() - 0.5, yield_data.max().max() + 0.5) # Consistent y-axis
    st.pyplot(fig)

    st.markdown("""
    Interact with the slider above to see how adding more principal components (Level, then Slope, then Curvature) progressively improves the reconstruction of the yield curve. Observe how:
    *   With **1 PC (Level)**, you get a basic, parallel shifted curve.
    *   With **2 PCs (Level + Slope)**, the steepness or flatness of the curve is accurately captured.
    *   With **3 PCs (Level + Slope + Curvature)**, the overall shape, including the convexity, is closely matched to the original curve, demonstrating the comprehensive explanatory power of these components.
    This interactive visualization provides a clear, intuitive understanding of how these primary factors collectively determine the shape and movements of yield curves.
    """)