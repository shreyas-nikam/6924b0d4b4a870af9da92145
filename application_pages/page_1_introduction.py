"""Page 1: Introduction"""

import streamlit as st

def main():
    st.title("Yield Curve Decomposition with Principal Component Analysis (PCA)")

    st.markdown("""
    ### Understanding Yield Curve Decomposition with PCA

    The yield curve is a fundamental indicator of economic health and market expectations, representing the relationship between the yield on bonds of the same credit quality but different maturities. Analyzing its movements is crucial for financial professionals. However, yield curves are high-dimensional objects, making direct analysis challenging.

    **Principal Component Analysis (PCA)** is a powerful dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving the maximum possible variance [7]. In finance, PCA is extensively used to decompose complex structures, such as yield curves, into a few interpretable underlying factors. This lab will guide Financial Data Engineers through the application of PCA to yield curve data, revealing its dominant drivers: level, slope, and curvature [8].

    The core idea is to identify orthogonal (uncorrelated) components, known as principal components, that capture the most significant variations in the data. For yield curves, these components correspond to economically meaningful movements. We will follow the PCA algorithm steps as outlined in Figure 4 [7].
    """)