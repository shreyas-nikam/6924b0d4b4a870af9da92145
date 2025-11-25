"""Page 2: Data Generation & Visualization"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100

def main():
    st.title("Data Generation & Visualization")

    st.markdown("""
    ### Generating Synthetic Yield Curve Data

    To demonstrate PCA on yield curves, we will first generate synthetic historical yield curve data. This simulated dataset will include yields for various maturities, such as 3-month, 1-year, 5-year, 10-year, and 30-year bonds. The generation process aims to mimic realistic yield curve dynamics, ensuring that the underlying factors (level, slope, curvature) can be clearly identified through PCA. We will simulate daily data for a period equivalent to one trading year (252 days).
    """)

    if 'yield_data' not in st.session_state:
        st.error("Yield data not found in session state. Please go back to the Introduction page.")
        return

    yield_data = st.session_state.yield_data
    maturities = st.session_state.maturities

    st.subheader("Simulated Yield Curve Data (first 5 days):")
    st.dataframe(yield_data.head())

    st.subheader("Descriptive Statistics of Simulated Yields:")
    st.dataframe(yield_data.describe())

    st.markdown("""
    The synthetic yield curve data, displayed above, provides a realistic representation of how yield rates might evolve across different maturities over time. Notice the range and variability in yields, which are crucial for PCA to extract meaningful components. Each column represents a different maturity, and each row corresponds to a specific day.
    """)

    st.markdown("""
    ### Visualizing Historical Yield Curve Movements

    Before applying PCA, it's beneficial to visualize the raw historical yield curve data. This allows us to observe the general trends, volatilities, and correlations between different maturities over the simulated period. We expect to see parallel shifts, steepening/flattening, and changes in convexity, which PCA will later decompose into distinct components.
    """)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, maturity in enumerate(maturities):
        ax.plot(yield_data.index, yield_data.iloc[:, i], label=f'{maturity}-year')
    ax.set_title('Simulated Historical Yield Curves Over Time')
    ax.set_xlabel('Date Index')
    ax.set_ylabel('Yield (%)')
    ax.legend(title='Maturity')
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("""
    The plot above illustrates the time series of yields for each maturity. We can observe the typical behavior of yield curves, including periods where all rates move in the same direction (parallel shifts) and periods where the spreads between short and long-term rates change (steepening or flattening). These visual patterns confirm the presence of systematic movements that PCA aims to capture.
    """)