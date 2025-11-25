
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plots for better aesthetics globally
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100

def plot_explained_variance(explained_variance_ratio: np.ndarray, cumulative_explained_variance_ratio: np.ndarray):
    num_components = len(explained_variance_ratio)
    components_idx = np.arange(1, num_components + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(components_idx, explained_variance_ratio, alpha=0.7, label='Individual Explained Variance')
    ax.plot(components_idx, cumulative_explained_variance_ratio, marker='o', linestyle='--', color='red', label='Cumulative Explained Variance')
    ax.axhline(y=0.95, color='gray', linestyle=':', label='95% Threshold')
    ax.axhline(y=0.99, color='darkgray', linestyle=':', label='99% Threshold')

    ax.set_title('Explained Variance by Principal Component')
    ax.set_xlabel('Principal Component Index')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_xticks(components_idx)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def main():
    maturities = st.session_state["maturities"]
    centered_yield_data = st.session_state["centered_yield_data"]
    covariance_matrix = st.session_state["covariance_matrix"]
    eigenvalues = st.session_state["eigenvalues"]
    eigenvectors = st.session_state["eigenvectors"]
    sorted_eigenvalues = st.session_state["sorted_eigenvalues"]
    explained_variance_ratio = st.session_state["explained_variance_ratio"]
    cumulative_explained_variance_ratio = st.session_state["cumulative_explained_variance_ratio"]

    st.header("PCA Steps 1-5")

    # Markdown Cell: Step 1: Data Centering
    st.markdown(r"""
    ### Step 1: Data Centering

    The first crucial step in PCA is to center the data. This involves subtracting the mean of each feature (in our case, each maturity) from its respective observations. Centering ensures that the first principal component explains the maximum variance, as it will pass through the origin in the transformed feature space. Without centering, the first principal component might simply correspond to the mean of the data rather than the direction of maximum variance.

    Mathematically, if $X$ is our data matrix (where each column is a maturity and each row is a day), and $\text{mean}(X)$ is a row vector of the means of each column, then the centered data $X_{\text{centered}}$ is calculated as:

    $$ X_{\text{centered}} = X - \text{mean}(X) $$
    """)

    st.subheader("Centered Yield Curve Data (first 5 days):")
    st.dataframe(centered_yield_data.head())

    st.subheader("Mean of Centered Data Columns:")
    st.write(centered_yield_data.mean())

    # Markdown Cell: The table above shows...
    st.markdown("""
    The table above shows the first five rows of the centered yield curve data. As expected, the mean of each maturity column in the `centered_yield_data` DataFrame is now very close to zero, confirming that the data has been correctly centered around its origin. This prepares our data for the next step: covariance matrix computation.
    """)

    # Markdown Cell: Step 2: Computing the Covariance Matrix
    st.markdown(r"""
    ### Step 2: Computing the Covariance Matrix

    After centering the data, the next step is to compute the covariance matrix. The covariance matrix quantifies the degree to which each pair of maturities varies together. A positive covariance indicates that two maturities tend to move in the same direction, while a negative covariance suggests they move in opposite directions. The diagonal elements of the covariance matrix represent the variance of each individual maturity.

    For a centered data matrix $X_{\text{centered}}$ with $n$ observations (days) and $m$ features (maturities), the covariance matrix $C$ is computed as:

    $$ C = \frac{1}{n-1} X_{\text{centered}}^T X_{\text{centered}} $$

    Here, $X_{\text{centered}}^T$ is the transpose of the centered data matrix. This matrix will be square, with dimensions $m \times m$.
    """)

    st.subheader("Covariance Matrix:")
    st.dataframe(pd.DataFrame(covariance_matrix, index=maturities, columns=maturities))

    # Markdown Cell: The covariance matrix above...
    st.markdown("""
    The covariance matrix above shows the relationships between different maturities. The diagonal elements represent the variance of each maturity, indicating its volatility. The off-diagonal elements show the covariance between pairs of maturities. As anticipated, we observe positive covariances, indicating that yields across different maturities generally move in the same direction, reflecting the correlated nature of yield curve movements.
    """)

    # Markdown Cell: Step 3: Performing Eigendecomposition
    st.markdown(r"""
    ### Step 3: Performing Eigendecomposition

    Eigendecomposition is the core mathematical operation in PCA. It decomposes the covariance matrix into a set of eigenvectors and eigenvalues.
    *   **Eigenvectors** represent the principal components. These are orthogonal (uncorrelated) directions in the original feature space along which the data varies the most. They define the new coordinate system.
    *   **Eigenvalues** quantify the amount of variance explained by each corresponding eigenvector. A larger eigenvalue means its associated eigenvector captures more of the data's variance.

    For a symmetric matrix like our covariance matrix $C$, eigendecomposition finds a matrix of eigenvectors $V$ and a diagonal matrix of eigenvalues $\Lambda$ such that:

    $$ C = V \Lambda V^T $$

    Where $V$ contains the eigenvectors as its columns, and $\Lambda$ is a diagonal matrix where the diagonal entries are the eigenvalues.
    """)

    st.subheader("Eigenvalues (Unsorted):")
    st.write(eigenvalues)
    st.subheader("Eigenvectors (Unsorted, columns are eigenvectors):")
    st.write(eigenvectors)

    # Markdown Cell: The unsorted eigenvalues...
    st.markdown("""
    The unsorted eigenvalues and eigenvectors obtained from the eigendecomposition are displayed above. The eigenvalues represent the variance explained by each principal component, and the eigenvectors define the directions of these components in the original maturity space. In the next step, we will sort these components by the magnitude of their eigenvalues to identify the most significant principal components.
    """)

    # Markdown Cell: Step 4 & 5: Sorting and Selecting Principal Components
    st.markdown(r"""
    ### Step 4 & 5: Sorting and Selecting Principal Components

    To identify the most significant principal components, we need to sort the eigenvalues in descending order. The eigenvector corresponding to the largest eigenvalue is the first principal component, capturing the most variance in the data. Subsequent eigenvectors, ordered by their eigenvalues, capture progressively less variance.

    After sorting, we select the top $k$ principal components (eigenvectors) that collectively explain a significant portion of the total variance. For yield curve decomposition, it is well-established that the first three principal components typically explain **95% to 99%** of the total yield curve movements [8].
    """)

    st.subheader("Sorted Eigenvalues:")
    st.write(sorted_eigenvalues)
    st.subheader("Explained Variance Ratio:")
    st.write(explained_variance_ratio)
    st.subheader("Cumulative Explained Variance Ratio:")
    st.write(cumulative_explained_variance_ratio)

    # Markdown Cell: The sorted eigenvalues...
    st.markdown("""
    The sorted eigenvalues and their corresponding explained variance ratios clearly show the dominance of the first few principal components. The cumulative explained variance ratio provides insight into how much of the total data variability is captured by a subset of these components. This confirms our expectation that a small number of components can explain a large proportion of yield curve movements.
    """)

    # Markdown Cell: Visualizing Explained Variance
    st.markdown("""
    ### Visualizing Explained Variance

    Visualizing the explained variance ratio helps us determine the optimal number of principal components to retain. A "scree plot" (bar plot for individual variance and line plot for cumulative variance) is commonly used for this purpose. We expect to see that the first three components explain the vast majority of the variance, aligning with financial literature [8].
    """)

    # Plot the explained variance
    plot_explained_variance(explained_variance_ratio, cumulative_explained_variance_ratio)

    # Markdown Cell: The explained variance plot...
    st.markdown("""
    The explained variance plot clearly demonstrates that the first three principal components indeed capture a substantial portion of the total variance in the yield curve data, typically between 95% and 99%. This reinforces the idea that yield curve movements can be effectively summarized by a small number of underlying factors, significantly reducing the dimensionality of the problem while retaining most of the important information.
    """)
