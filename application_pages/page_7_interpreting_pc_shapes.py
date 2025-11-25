"""Page 7: Interpreting Principal Component Shapes"""

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100

def main():
    st.title("Interpreting Principal Component Shapes: Level, Slope, and Curvature")

    st.markdown("""
    ### Interpreting Principal Component Shapes: Level, Slope, and Curvature

    The eigenvectors, when plotted against maturities, reveal the characteristic shapes of the principal components. For yield curves, the first three principal components have well-established financial interpretations [8]:

    1.  **First Principal Component (Level):**
        *   Typically accounts for 80%-90% of the variance.
        *   Represents parallel shifts in the yield curve, where all yields tend to move up or down together by similar amounts.
        *   Financially, this reflects broad monetary policy changes, inflation expectations, or general economic conditions that affect all maturities uniformly.

    2.  **Second Principal Component (Slope):**
        *   Explains roughly 5%-15% of the variance.
        *   Captures the steepening or flattening of the yield curve, reflecting changes in the spread between long-term and short-term rates.
        *   Financially, this is related to expectations about future monetary policy, economic growth, or recessionary fears. A steeper curve often signals expectations of future growth and inflation, while a flatter or inverted curve may signal economic slowdowns.

    3.  **Third Principal Component (Curvature):**
        *   Accounts for 1%-5% of the variance.
        *   Captures changes in the curve's convexity or "bow" shape, reflecting relative movements in medium-term rates compared with short- and long-term rates.
        *   Financially, this component is often related to market expectations about intermediate-term economic conditions or supply/demand imbalances at specific maturities.
    """)

    if 'sorted_eigenvectors' not in st.session_state or 'maturities' not in st.session_state:
        st.error("Sorted eigenvectors or maturities not found in session state. Please go through previous steps.")
        return

    sorted_eigenvectors = st.session_state.sorted_eigenvectors
    maturities = st.session_state.maturities

    fig, ax = plt.subplots(figsize=(10, 6))
    component_names = ['Level', 'Slope', 'Curvature']
    num_to_plot = min(3, sorted_eigenvectors.shape[1])
    for i in range(num_to_plot):
        ax.plot(maturities, sorted_eigenvectors[:, i], marker='o', label=f'PC {i+1}: {component_names[i] if i < len(component_names) else ""}')
    ax.set_title('Principal Component Shapes (Eigenvectors)')
    ax.set_xlabel('Maturity (Years)')
    ax.set_ylabel('Component Weight')
    ax.set_xticks(maturities)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.markdown("""
    The plot clearly visualizes the shapes of the first three principal components.
    *   **PC1 (Level)** shows weights that are all positive and relatively uniform across maturities, signifying a parallel shift in the yield curve.
    *   **PC2 (Slope)** exhibits positive weights for short maturities and negative weights for long maturities, indicating that it captures the steepening or flattening behavior.
    *   **PC3 (Curvature)** displays weights that are positive at short and long ends but negative in the middle, representing changes in the convexity or "bow" of the yield curve.
    These shapes perfectly align with their financial interpretations, providing an intuitive understanding of the fundamental drivers of yield curve movements.
    """)