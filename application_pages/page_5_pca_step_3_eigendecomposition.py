"""Page 5: PCA Step 3: Performing Eigendecomposition"""

import streamlit as st
import numpy as np

def main():
    st.title("PCA Step 3: Performing Eigendecomposition")

    st.markdown(r"""
    ### Step 3: Performing Eigendecomposition

    Eigendecomposition is the core mathematical operation in PCA. It decomposes the covariance matrix into a set of eigenvectors and eigenvalues.
    *   **Eigenvectors** represent the principal components. These are orthogonal (uncorrelated) directions in the original feature space along which the data varies the most. They define the new coordinate system.
    *   **Eigenvalues** quantify the amount of variance explained by each corresponding eigenvector. A larger eigenvalue means its associated eigenvector captures more of the data's variance.

    For a symmetric matrix like our covariance matrix $C$, eigendecomposition finds a matrix of eigenvectors $V$ and a diagonal matrix of eigenvalues $\Lambda$ such that:

    $$ C = V \Lambda V^T $$

    Where $V$ contains the eigenvectors as its columns, and $\Lambda$ is a diagonal matrix where the diagonal entries are the eigenvalues.
    """)

    if 'eigenvalues' not in st.session_state or 'eigenvectors' not in st.session_state:
        st.error("Eigenvalues or eigenvectors not found in session state. Please go through previous steps.")
        return

    eigenvalues = st.session_state.eigenvalues
    eigenvectors = st.session_state.eigenvectors

    st.subheader("Eigenvalues (Unsorted):")
    st.write(eigenvalues)
    st.subheader("Eigenvectors (Unsorted, columns are eigenvectors):")
    st.write(eigenvectors)

    st.markdown("""
    The unsorted eigenvalues and eigenvectors obtained from the eigendecomposition are displayed above. The eigenvalues represent the variance explained by each principal component, and the eigenvectors define the directions of these components in the original maturity space. In the next step, we will sort these components by the magnitude of their eigenvalues to identify the most significant principal components.
    """)