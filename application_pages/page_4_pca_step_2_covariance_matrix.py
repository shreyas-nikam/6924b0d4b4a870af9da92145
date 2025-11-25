"""Page 4: PCA Step 2: Covariance Matrix Computation"""

import streamlit as st
import pandas as pd

def main():
    st.title("PCA Step 2: Covariance Matrix Computation")

    st.markdown(r"""
    ### Step 2: Computing the Covariance Matrix

    After centering the data, the next step is to compute the covariance matrix. The covariance matrix quantifies the degree to which each pair of maturities varies together. A positive covariance indicates that two maturities tend to move in the same direction, while a negative covariance suggests they move in opposite directions. The diagonal elements of the covariance matrix represent the variance of each individual maturity.

    For a centered data matrix $X_{\text{centered}}$ with $n$ observations (days) and $m$ features (maturities), the covariance matrix $C$ is computed as:

    $$ C = \frac{1}{n-1} X_{\text{centered}}^T X_{\text{centered}} $$

    Here, $X_{\text{centered}}^T$ is the transpose of the centered data matrix. This matrix will be square, with dimensions $m \times m$.
    """)

    if 'covariance_matrix' not in st.session_state or 'maturities' not in st.session_state:
        st.error("Covariance matrix or maturities not found in session state. Please go through previous steps.")
        return

    covariance_matrix = st.session_state.covariance_matrix
    maturities = st.session_state.maturities

    st.subheader("Covariance Matrix:")
    st.dataframe(pd.DataFrame(covariance_matrix, index=maturities, columns=maturities))

    st.markdown("""
    The covariance matrix above shows the relationships between different maturities. The diagonal elements represent the variance of each maturity, indicating its volatility. The off-diagonal elements show the covariance between pairs of maturities. As anticipated, we observe positive covariances, indicating that yields across different maturities generally move in the same direction, reflecting the correlated nature of yield curve movements.
    """)