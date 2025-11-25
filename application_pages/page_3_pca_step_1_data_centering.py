"""Page 3: PCA Step 1: Data Centering"""

import streamlit as st
import pandas as pd

def main():
    st.title("PCA Step 1: Data Centering")

    st.markdown(r"""
    ### Step 1: Data Centering

    The first crucial step in PCA is to center the data. This involves subtracting the mean of each feature (in our case, each maturity) from its respective observations. Centering ensures that the first principal component explains the maximum variance, as it will pass through the origin in the transformed feature space. Without centering, the first principal component might simply correspond to the mean of the data rather than the direction of maximum variance.

    Mathematically, if $X$ is our data matrix (where each column is a maturity and each row is a day), and $\text{mean}(X)$ is a row vector of the means of each column, then the centered data $X_{\text{centered}}$ is calculated as:

    $$ X_{\text{centered}} = X - \text{mean}(X) $$
    """)

    if 'centered_yield_data' not in st.session_state or 'mean_yields' not in st.session_state:
        st.error("Centered data or mean yields not found in session state. Please go through previous steps.")
        return

    centered_yield_data = st.session_state.centered_yield_data
    mean_yields = st.session_state.mean_yields

    st.subheader("Centered Yield Curve Data (first 5 days):")
    st.dataframe(centered_yield_data.head())

    st.subheader("Mean of Centered Data Columns:")
    st.write(centered_yield_data.mean())

    st.markdown("""
    The table above shows the first five rows of the centered yield curve data. As expected, the mean of each maturity column in the `centered_yield_data` DataFrame is now very close to zero, confirming that the data has been correctly centered around its origin. This prepares our data for the next step: covariance matrix computation.
    """)