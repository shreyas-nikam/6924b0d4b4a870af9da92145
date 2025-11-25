
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plots for better aesthetics globally
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100

def plot_yield_curves_timeseries(data: pd.DataFrame, maturities: list):
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, maturity in enumerate(maturities):
        ax.plot(data.index, data.iloc[:, i], label=f'{maturity}-year')
    ax.set_title('Simulated Historical Yield Curves Over Time')
    ax.set_xlabel('Date Index')
    ax.set_ylabel('Yield (%)')
    ax.legend(title='Maturity')
    ax.grid(True)
    st.pyplot(fig)

def main():
    maturities = st.session_state["maturities"]
    yield_data = st.session_state["yield_data"]

    st.header("Introduction & Data")

    # Markdown Cell: Understanding Yield Curve Decomposition with PCA
    st.markdown("""
    ### Understanding Yield Curve Decomposition with PCA

    The yield curve is a fundamental indicator of economic health and market expectations, representing the relationship between the yield on bonds of the same credit quality but different maturities. Analyzing its movements is crucial for financial professionals. However, yield curves are high-dimensional objects, making direct analysis challenging.

    **Principal Component Analysis (PCA)** is a powerful dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving the maximum possible variance [7]. In finance, PCA is extensively used to decompose complex structures, such as yield curves, into a few interpretable underlying factors. This notebook will guide Financial Data Engineers through the application of PCA to yield curve data, revealing its dominant drivers: level, slope, and curvature [8].

    The core idea is to identify orthogonal (uncorrelated) components, known as principal components, that capture the most significant variations in the data. For yield curves, these components correspond to economically meaningful movements. We will follow the PCA algorithm steps as outlined in Figure 4 [7].
    """)

    # Markdown Cell: Generating Synthetic Yield Curve Data
    st.markdown("""
    ### Generating Synthetic Yield Curve Data

    To demonstrate PCA on yield curves, we will first generate synthetic historical yield curve data. This simulated dataset will include yields for various maturities, such as 3-month, 1-year, 5-year, 10-year, and 30-year bonds. The generation process aims to mimic realistic yield curve dynamics, ensuring that the underlying factors (level, slope, curvature) can be clearly identified through PCA. We will simulate daily data for a period equivalent to one trading year (252 days).
    """)

    st.subheader("Simulated Yield Curve Data (first 5 days):")
    st.dataframe(yield_data.head())

    st.subheader("Descriptive Statistics of Simulated Yields:")
    st.dataframe(yield_data.describe())

    # Markdown Cell: The synthetic yield curve data...
    st.markdown("""
    The synthetic yield curve data, displayed above, provides a realistic representation of how yield rates might evolve across different maturities over time. Notice the range and variability in yields, which are crucial for PCA to extract meaningful components. Each column represents a different maturity, and each row corresponds to a specific day.
    """)

    # Markdown Cell: Visualizing Historical Yield Curve Movements
    st.markdown("""
    ### Visualizing Historical Yield Curve Movements

    Before applying PCA, it's beneficial to visualize the raw historical yield curve data. This allows us to observe the general trends, volatilities, and correlations between different maturities over the simulated period. We expect to see parallel shifts, steepening/flattening, and changes in convexity, which PCA will later decompose into distinct components.
    """)

    # Plot the simulated yield curve data
    plot_yield_curves_timeseries(yield_data, maturities)

    # Markdown Cell: The plot above illustrates...
    st.markdown("""
    The plot above illustrates the time series of yields for each maturity. We can observe the typical behavior of yield curves, including periods where all rates move in the same direction (parallel shifts) and periods where the spreads between short and long-term rates change (steepening or flattening). These visual patterns confirm the presence of systematic movements that PCA aims to capture.
    """)
