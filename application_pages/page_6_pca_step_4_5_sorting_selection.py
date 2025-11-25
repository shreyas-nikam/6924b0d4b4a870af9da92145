"""Page 6: PCA Step 4 & 5: Sorting and Selecting Principal Components"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100

def main():
    st.title("PCA Step 4 & 5: Sorting and Selecting Principal Components")

    st.markdown(r"""
    ### Step 4 & 5: Sorting and Selecting Principal Components

    To identify the most significant principal components, we need to sort the eigenvalues in descending order. The eigenvector corresponding to the largest eigenvalue is the first principal component, capturing the most variance in the data. Subsequent eigenvectors, ordered by their eigenvalues, capture progressively less variance.

    After sorting, we select the top $k$ principal components (eigenvectors) that collectively explain a significant portion of the total variance. For yield curve decomposition, it is well-established that the first three principal components typically explain **95% to 99%** of the total yield curve movements [8].
    """)

    if ('sorted_eigenvalues' not in st.session_state or 
        'explained_variance_ratio' not in st.session_state or 
        'cumulative_explained_variance_ratio' not in st.session_state):
        st.error("Sorted eigen components or explained variance ratios not found in session state. Please go through previous steps.")
        return

    sorted_eigenvalues = st.session_state.sorted_eigenvalues
    explained_variance_ratio = st.session_state.explained_variance_ratio
    cumulative_explained_variance_ratio = st.session_state.cumulative_explained_variance_ratio

    st.subheader("Sorted Eigenvalues:")
    st.write(sorted_eigenvalues)
    st.subheader("Explained Variance Ratio:")
    st.write(explained_variance_ratio)
    st.subheader("Cumulative Explained Variance Ratio:")
    st.write(cumulative_explained_variance_ratio)

    st.markdown("""
    The sorted eigenvalues and their corresponding explained variance ratios clearly show the dominance of the first few principal components. The cumulative explained variance ratio provides insight into how much of the total data variability is captured by a subset of these components. This confirms our expectation that a small number of components can explain a large proportion of yield curve movements.
    """)

    st.markdown("""
    ### Visualizing Explained Variance

    Visualizing the explained variance ratio helps us determine the optimal number of principal components to retain. A 