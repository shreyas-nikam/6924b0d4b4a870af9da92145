"""Page 8: PCA Step 6: Transforming Data to Principal Component Space"""

import streamlit as st
import pandas as pd

def main():
    st.title("PCA Step 6: Transforming Data to Principal Component Space")

    st.markdown(r"""
    ### Step 6: Transforming Data to Principal Component Space

    Once the principal components (eigenvectors) are identified, the original centered data can be projected onto these new orthogonal axes. This transformation creates a new dataset, often called the "scores" or "transformed data," where each column corresponds to a principal component. These scores represent the daily values or "magnitudes" of the level, slope, and curvature movements affecting the yield curve.

    The transformed data $Y$ is obtained by multiplying the centered data $X_{\text{centered}}$ by the matrix of selected principal components $W$:

    $$ Y = X_{\text{centered}} W $$

    Here, $W$ would typically contain the top $k$ eigenvectors as its columns. In our case, we'll transform the data using all components for completeness, but we can later select a subset for reconstruction.
    """)

    if 'transformed_yield_data' not in st.session_state:
        st.error("Transformed yield data not found in session state. Please go through previous steps.")
        return

    transformed_yield_data = st.session_state.transformed_yield_data

    st.subheader("Transformed Yield Data (first 5 days, PCA scores):")
    st.dataframe(transformed_yield_data.head())

    st.markdown("""
    The table above displays the first few rows of the transformed data. Each column now represents the "score" for a specific principal component on a given day. For example, the 'PC 1' column indicates the daily magnitude of the 'level' movement in the yield curve, 'PC 2' for the 'slope', and so on. These scores are uncorrelated and capture the underlying daily movements in a more compact and interpretable form.
    """)