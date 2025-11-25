"""Page 9: Yield Curve Reconstruction (Static)"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100

def main():
    st.title("Yield Curve Reconstruction (Static)")

    st.markdown(r"""
    ### Yield Curve Reconstruction

    One of the powerful aspects of PCA is the ability to reconstruct the original data using a subset of the principal components. This demonstrates how much information about the original yield curve shape is captured by the selected components. By using only the top principal components, we can essentially denoise the data and isolate the most significant movements.

    The reconstruction process is the inverse of the transformation. If $Y_k$ are the scores from the top $k$ principal components, and $W_k$ are the corresponding $k$ eigenvectors, then the reconstructed centered data $\hat{X}_{\text{centered}}$ is:

    $$ \hat{X}_{\text{centered}} = Y_k W_k^T $$

    To get the reconstructed yield curve $\hat{X}$, we add back the original mean yields:

    $$ \hat{X} = \hat{X}_{\text{centered}} + \text{mean}(X) $$
    """)

    if ('yield_data' not in st.session_state or 
        'transformed_yield_data' not in st.session_state or 
        'sorted_eigenvectors' not in st.session_state or 
        'mean_yields' not in st.session_state or
        'maturities' not in st.session_state or
        'reconstruct_yield_curve' not in st.session_state):
        st.error("Required data for reconstruction not found in session state. Please go through previous steps.")
        return
    
    yield_data = st.session_state.yield_data
    transformed_yield_data = st.session_state.transformed_yield_data
    sorted_eigenvectors = st.session_state.sorted_eigenvectors
    mean_yields = st.session_state.mean_yields
    maturities = st.session_state.maturities
    reconstruct_yield_curve = st.session_state.reconstruct_yield_curve

    # Select a specific day (e.g., the last day) to illustrate reconstruction
    day_index_static = yield_data.index[-1]
    original_curve_day_static = yield_data.loc[day_index_static]

    # Reconstruct the yield curve using 3 principal components
    reconstructed_curve_3pc_static = reconstruct_yield_curve(
        transformed_yield_data, sorted_eigenvectors, mean_yields, num_components=3
    )
    reconstructed_curve_day_3pc_static = reconstructed_curve_3pc_static.loc[day_index_static]

    st.subheader(f"Original Yield Curve for Day {day_index_static.strftime('%Y-%m-%d')}:")
    st.write(original_curve_day_static)
    st.subheader(f"Reconstructed Yield Curve (using 3 PCs) for Day {day_index_static.strftime('%Y-%m-%d')}:")
    st.write(reconstructed_curve_day_3pc_static)

    fig_static, ax_static = plt.subplots(figsize=(10, 6))
    ax_static.plot(maturities, original_curve_day_static, marker='o', label='Original Yield Curve')
    ax_static.plot(maturities, reconstructed_curve_day_3pc_static, marker='x', linestyle='--', label='Reconstructed (3 PCs)')
    ax_static.set_title(f'Yield Curve Reconstruction for Day {day_index_static.strftime('%Y-%m-%d')} (Using 3 Principal Components)')
    ax_static.set_xlabel('Maturity (Years)')
    ax_static.set_ylabel('Yield (%)')
    ax_static.set_xticks(maturities)
    ax_static.grid(True)
    ax_static.legend()
    st.pyplot(fig_static)

    st.markdown("""
    The comparison between the original and reconstructed yield curve for a specific day visually demonstrates the effectiveness of PCA. Using just three principal components, the reconstructed curve closely approximates the original, confirming that these components capture the essential shape and movements of the yield curve. The slight differences highlight the small amount of variance not captured by the top components.
    """)