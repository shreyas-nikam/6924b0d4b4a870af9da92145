id: 6924b0d4b4a870af9da92145_documentation
summary: Yield Curve Decomposer: PCA in Action Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Yield Curve Decomposition with Principal Component Analysis (PCA) using Streamlit

## Introduction to Yield Curve Decomposition and PCA
Duration: 0:05

The yield curve is a fundamental indicator in finance, representing the relationship between the yield and maturity of fixed-income securities. Its shape and movements provide critical insights into market expectations regarding inflation, economic growth, and future interest rates. For financial professionals, particularly Financial Data Engineers, understanding and analyzing these complex movements is paramount for risk management, hedging, and strategic decision-making.

However, yield curves are high-dimensional objects, involving yields across many maturities that often move together in correlated ways. Directly analyzing these high-dimensional, correlated movements can be challenging. This is where **Principal Component Analysis (PCA)** becomes an invaluable tool. PCA is a powerful dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving the maximum possible variance.

<aside class="positive">
<b>Why PCA for Yield Curves?</b> PCA helps us decompose the complex, correlated movements of the yield curve into a few independent, interpretable factors. In the context of yield curves, these factors typically correspond to economically meaningful movements known as **Level, Slope, and Curvature**. This allows Financial Data Engineers to simplify complex market data, making it more tractable for analysis, modeling, and interpretation.
</aside>

This codelab will guide you through a Streamlit application that demonstrates the application of PCA to yield curve data. You will learn:

*   The theoretical foundations and practical application of PCA in a financial context.
*   How to execute each step of the PCA algorithm, from data centering to eigendecomposition.
*   To interpret the financial significance of the principal components as Level, Slope, and Curvature.
*   To visualize and understand the contribution of each principal component to the overall yield curve shape.
*   How PCA enhances risk management, hedging, and economic analysis capabilities.

The application follows the standard PCA algorithm steps, which can be conceptually visualized as:

1.  **Data Preparation:** Gather and normalize (center) the yield curve data.
2.  **Covariance Matrix Calculation:** Compute the covariance matrix to understand relationships between maturities.
3.  **Eigendecomposition:** Extract eigenvalues and eigenvectors from the covariance matrix.
4.  **Component Selection:** Sort eigenvectors by their corresponding eigenvalues and select the most significant ones.
5.  **Data Transformation:** Project the original data onto the selected principal components.
6.  **Interpretation & Reconstruction:** Analyze the principal component shapes and reconstruct the yield curve using a subset of components.

Let's dive into setting up and exploring the Streamlit application.

## Setting up the Streamlit Application Environment
Duration: 0:10

To begin, you need to set up the Streamlit application locally. This involves getting the Python code and running it.

First, ensure you have Python installed (version 3.8 or higher is recommended). Then, install the necessary Python packages:

```console
pip install streamlit numpy pandas matplotlib seaborn scipy
```

Next, create the following file structure and save the provided code snippets:

```
your_project_folder/
├── app.py
└── application_pages/
    └── pca_yield_curve_decomposition.py
```

**`app.py`:**
This is the main entry point for the Streamlit application. It sets up the page configuration and navigates to the core PCA application.

```python
import streamlit as st
import os

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()

st.markdown("""
In this lab, we delve into the world of **Yield Curve Decomposition with Principal Component Analysis (PCA)**. 
This application is designed for Financial Data Engineers to interactively explore and understand how PCA can be applied to complex yield curve data. 
We will walk through each step of the PCA algorithm, from data generation and visualization to the interpretation of key principal components like Level, Slope, and Curvature, and finally, the interactive reconstruction of yield curves.

**Learning Objectives:**
- Understand the theoretical foundations and practical applications of PCA in finance.
- Execute the core PCA steps programmatically.
- Interpret the financial significance of the Level, Slope, and Curvature components.
- Analyze and visualize the contribution of each principal component to yield curve shapes.
- Gain insights into PCA's role in risk management, hedging, and economic analysis.

Use the sidebar to navigate through the different sections of the lab.
"""))

# Ensure the application_pages directory exists
if not os.path.exists("application_pages"):
    os.makedirs("application_pages")

page = st.sidebar.selectbox(label="Navigation", options=["Yield Curve Decomposition with PCA"])

if page == "Yield Curve Decomposition with PCA":
    from application_pages.pca_yield_curve_decomposition import main
    main()
```

**`application_pages/pca_yield_curve_decomposition.py`:**
This file contains the core logic for simulating data, performing PCA, and visualizing the results.

```python
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eig

def main():
    # Configure plots for better aesthetics globally
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 100

    #  Application Sidebar (Configuration) 
    st.sidebar.header("Configuration")

    st.sidebar.subheader("Data Simulation Parameters")
    num_days_input = st.sidebar.number_input(
        "Number of Business Days (e.g., 252 for a year)",
        value=252, min_value=50, max_value=500, step=1,
        key="num_days_input"
    )
    maturities_input_str = st.sidebar.text_input(
        "Maturities (comma-separated years, e.g., 0.25, 1, 5, 10, 30)",
        value="0.25, 1, 5, 10, 30",
        key="maturities_input_str"
    )
    try:
        maturities = [float(m.strip()) for m in maturities_input_str.split(',')]
        if not maturities:
            st.sidebar.error("Maturities list cannot be empty.")
            st.stop()
    except ValueError:
        st.sidebar.error("Invalid maturities format. Please use comma-separated numbers.")
        st.stop()

    # Placeholder for custom data upload
    # st.sidebar.subheader("Upload Custom Data (Optional)")
    # uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    # if uploaded_file:
    #     # Logic to process uploaded file (not in current scope, placeholder)
    #     st.sidebar.info("Custom data upload functionality is a future enhancement.")


    #  Functions from Jupyter Notebook (Adapted for Streamlit Caching) 

    # Define function to simulate yield curve data
    @st.cache_data
    def simulate_yield_curve_data(num_days: int, maturities: list) -> pd.DataFrame:
        np.random.seed(42) # For reproducibility

        base_rate = np.cumsum(np.random.normal(0, 0.05, num_days)) + 2.0
        slope_factor = np.cumsum(np.random.normal(0, 0.02, num_days)) - 0.5
        curvature_factor = np.cumsum(np.random.normal(0, 0.01, num_days)) + 0.1

        yield_curves = []
        for i in range(num_days):
            daily_yields = []
            for m in maturities:
                y = base_rate[i] + slope_factor[i] * (np.exp(-m/5) - 0.5) + curvature_factor[i] * (4 * m * np.exp(-m/10) - 1.0)
                daily_yields.append(max(0.01, y)) # Ensure non-negative yields
            yield_curves.append(daily_yields)

        df = pd.DataFrame(yield_curves, columns=[f'{m}-year' for m in maturities])
        df.index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_days, freq='B')) # Business days
        return df

    # Define function to plot yield curve time series
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

    # Define function to center the data
    @st.cache_data
    def center_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        mean_yields = data.mean()
        centered_data = data - mean_yields
        return centered_data, mean_yields

    # Define function to compute the covariance matrix
    @st.cache_data
    def compute_covariance_matrix(centered_data: pd.DataFrame) -> np.ndarray:
        covariance_matrix = np.cov(centered_data, rowvar=False)
        return covariance_matrix

    # Define function to perform eigendecomposition
    @st.cache_data
    def perform_eigendecomposition(covariance_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        return eigenvalues, eigenvectors

    # Define function to sort eigenvalues and eigenvectors
    @st.cache_data
    def sort_eigen_components(eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        return sorted_eigenvalues, sorted_eigenvectors

    # Define function to calculate explained variance
    @st.cache_data
    def calculate_explained_variance(sorted_eigenvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        total_variance = np.sum(sorted_eigenvalues)
        explained_variance_ratio = sorted_eigenvalues / total_variance
        cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
        return explained_variance_ratio, cumulative_explained_variance_ratio

    # Define function to plot explained variance
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

    # Define function to plot principal component shapes
    def plot_principal_component_shapes(principal_components: np.ndarray, maturities: list, num_to_plot: int = 3):
        fig, ax = plt.subplots(figsize=(10, 6))
        component_names = ['Level', 'Slope', 'Curvature']
        for i in range(min(num_to_plot, principal_components.shape[1])):
            ax.plot(maturities, principal_components[:, i], marker='o', label=f'PC {i+1}: {component_names[i] if i < len(component_names) else ""}')
        ax.set_title('Principal Component Shapes (Eigenvectors)')
        ax.set_xlabel('Maturity (Years)')
        ax.set_ylabel('Component Weight')
        ax.set_xticks(maturities)
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    # Define function to transform data
    @st.cache_data
    def transform_data(centered_data: pd.DataFrame, principal_components: np.ndarray) -> pd.DataFrame:
        transformed_data = centered_data @ principal_components
        return pd.DataFrame(transformed_data, index=centered_data.index,
                            columns=[f'PC {i+1}' for i in range(principal_components.shape[1])])

    # Define function to reconstruct yield curve
    @st.cache_data
    def reconstruct_yield_curve(transformed_data: pd.DataFrame, principal_components: np.ndarray,
                                mean_yields: pd.Series, num_components: int) -> pd.DataFrame:
        selected_transformed_data = transformed_data.iloc[:, :num_components]
        selected_principal_components = principal_components[:, :num_components]
        reconstructed_centered_data = selected_transformed_data @ selected_principal_components.T
        reconstructed_curve = reconstructed_centered_data + mean_yields
        return reconstructed_curve

    #  Streamlit Application Main Content 

    st.title("Yield Curve Decomposition with Principal Component Analysis (PCA)")

    # Markdown Cell: Understanding Yield Curve Decomposition with PCA
    st.markdown(r"""
    ### Understanding Yield Curve Decomposition with PCA

    The yield curve is a fundamental indicator of economic health and market expectations, representing the relationship between the yield on bonds of the same credit quality but different maturities. Analyzing its movements is crucial for financial professionals. However, yield curves are high-dimensional objects, making direct analysis challenging.

    **Principal Component Analysis (PCA)** is a powerful dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving the maximum possible variance [7]. In finance, PCA is extensively used to decompose complex structures, such as yield curves, into a few interpretable underlying factors. This notebook will guide Financial Data Engineers through the application of PCA to yield curve data, revealing its dominant drivers: level, slope, and curvature [8].

    The core idea is to identify orthogonal (uncorrelated) components, known as principal components, that capture the most significant variations in the data. For yield curves, these components correspond to economically meaningful movements. We will follow the PCA algorithm steps as outlined in Figure 4 [7].
    """)

    # Markdown Cell: Generating Synthetic Yield Curve Data
    st.markdown(r"""
    ### Generating Synthetic Yield Curve Data

    To demonstrate PCA on yield curves, we will first generate synthetic historical yield curve data. This simulated dataset will include yields for various maturities, such as 3-month, 1-year, 5-year, 10-year, and 30-year bonds. The generation process aims to mimic realistic yield curve dynamics, ensuring that the underlying factors (level, slope, curvature) can be clearly identified through PCA. We will simulate daily data for a period equivalent to one trading year (252 days).
    """)

    # Generate the simulated yield curve data
    yield_data = simulate_yield_curve_data(num_days_input, maturities)

    st.subheader("Simulated Yield Curve Data (first 5 days):")
    st.dataframe(yield_data.head())

    st.subheader("Descriptive Statistics of Simulated Yields:")
    st.dataframe(yield_data.describe())

    # Markdown Cell: The synthetic yield curve data...
    st.markdown(r"""
    The synthetic yield curve data, displayed above, provides a realistic representation of how yield rates might evolve across different maturities over time. Notice the range and variability in yields, which are crucial for PCA to extract meaningful components. Each column represents a different maturity, and each row corresponds to a specific day.
    """)

    # Markdown Cell: Visualizing Historical Yield Curve Movements
    st.markdown(r"""
    ### Visualizing Historical Yield Curve Movements

    Before applying PCA, it's beneficial to visualize the raw historical yield curve data. This allows us to observe the general trends, volatilities, and correlations between different maturities over the simulated period. We expect to see parallel shifts, steepening/flattening, and changes in convexity, which PCA will later decompose into distinct components.
    """)

    # Plot the simulated yield curve data
    plot_yield_curves_timeseries(yield_data, maturities)

    # Markdown Cell: The plot above illustrates...
    st.markdown(r"""
    The plot above illustrates the time series of yields for each maturity. We can observe the typical behavior of yield curves, including periods where all rates move in the same direction (parallel shifts) and periods where the spreads between short and long-term rates change (steepening or flattening). These visual patterns confirm the presence of systematic movements that PCA aims to capture.
    """)

    # Markdown Cell: Step 1: Data Centering
    st.markdown(r"""
    ### Step 1: Data Centering

    The first crucial step in PCA is to center the data. This involves subtracting the mean of each feature (in our case, each maturity) from its respective observations. Centering ensures that the first principal component explains the maximum variance, as it will pass through the origin in the transformed feature space. Without centering, the first principal component might simply correspond to the mean of the data rather than the direction of maximum variance.

    Mathematically, if $X$ is our data matrix (where each column is a maturity and each row is a day), and $\text{mean}(X)$ is a row vector of the means of each column, then the centered data $X_{\text{centered}}$ is calculated as:

    $$ X_{\text{centered}} = X - \text{mean}(X) $$
    """)

    # Center the yield data
    centered_yield_data, mean_yields = center_data(yield_data)

    st.subheader("Centered Yield Curve Data (first 5 days):")
    st.dataframe(centered_yield_data.head())

    st.subheader("Mean of Centered Data Columns:")
    st.write(centered_yield_data.mean())

    # Markdown Cell: The table above shows...
    st.markdown(r"""
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

    # Compute the covariance matrix
    covariance_matrix = compute_covariance_matrix(centered_yield_data)

    st.subheader("Covariance Matrix:")
    st.dataframe(pd.DataFrame(covariance_matrix, index=maturities, columns=maturities))

    # Markdown Cell: The covariance matrix above...
    st.markdown(r"""
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

    # Perform eigendecomposition on the covariance matrix
    eigenvalues, eigenvectors = perform_eigendecomposition(covariance_matrix)

    st.subheader("Eigenvalues (Unsorted):")
    st.write(eigenvalues)
    st.subheader("Eigenvectors (Unsorted, columns are eigenvectors):")
    st.write(eigenvectors)

    # Markdown Cell: The unsorted eigenvalues...
    st.markdown(r"""
    The unsorted eigenvalues and eigenvectors obtained from the eigendecomposition are displayed above. The eigenvalues represent the variance explained by each principal component, and the eigenvectors define the directions of these components in the original maturity space. In the next step, we will sort these components by the magnitude of their eigenvalues to identify the most significant principal components.
    """)

    # Markdown Cell: Step 4 & 5: Sorting and Selecting Principal Components
    st.markdown(r"""
    ### Step 4 & 5: Sorting and Selecting Principal Components

    To identify the most significant principal components, we need to sort the eigenvalues in descending order. The eigenvector corresponding to the largest eigenvalue is the first principal component, capturing the most variance in the data. Subsequent eigenvectors, ordered by their eigenvalues, capture progressively less variance.

    After sorting, we select the top $k$ principal components (eigenvectors) that collectively explain a significant portion of the total variance. For yield curve decomposition, it is well-established that the first three principal components typically explain **95% to 99%** of the total yield curve movements.
    """)

    # Sort the eigen components
    sorted_eigenvalues, sorted_eigenvectors = sort_eigen_components(eigenvalues, eigenvectors)

    # Calculate explained variance ratios
    explained_variance_ratio, cumulative_explained_variance_ratio = \
        calculate_explained_variance(sorted_eigenvalues)

    st.subheader("Sorted Eigenvalues:")
    st.write(sorted_eigenvalues)
    st.subheader("Explained Variance Ratio:")
    st.write(explained_variance_ratio)
    st.subheader("Cumulative Explained Variance Ratio:")
    st.write(cumulative_explained_variance_ratio)

    # Markdown Cell: The sorted eigenvalues...
    st.markdown(r"""
    The sorted eigenvalues and their corresponding explained variance ratios clearly show the dominance of the first few principal components. The cumulative explained variance ratio provides insight into how much of the total data variability is captured by a subset of these components. This confirms our expectation that a small number of components can explain a large proportion of yield curve movements.
    """)

    # Markdown Cell: Visualizing Explained Variance
    st.markdown(r"""
    ### Visualizing Explained Variance

    Visualizing the explained variance ratio helps us determine the optimal number of principal components to retain. A "scree plot" (bar plot for individual variance and line plot for cumulative variance) is commonly used for this purpose. We expect to see that the first three components explain the vast majority of the variance, aligning with financial literature.
    """)

    # Plot the explained variance
    plot_explained_variance(explained_variance_ratio, cumulative_explained_variance_ratio)

    # Markdown Cell: The explained variance plot...
    st.markdown(r"""
    The explained variance plot clearly demonstrates that the first three principal components indeed capture a substantial portion of the total variance in the yield curve data, typically between 95% and 99%. This reinforces the idea that yield curve movements can be effectively summarized by a small number of underlying factors, significantly reducing the dimensionality of the problem while retaining most of the important information.
    """)

    # Markdown Cell: Interpreting Principal Component Shapes: Level, Slope, and Curvature
    st.markdown(r"""
    ### Interpreting Principal Component Shapes: Level, Slope, and Curvature

    The eigenvectors, when plotted against maturities, reveal the characteristic shapes of the principal components. For yield curves, the first three principal components have well-established financial interpretations:

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

    # Plot the first three principal component shapes
    plot_principal_component_shapes(sorted_eigenvectors, maturities, num_to_plot=3)

    # Markdown Cell: The plot clearly visualizes...
    st.markdown(r"""
    The plot clearly visualizes the shapes of the first three principal components.
    *   **PC1 (Level)** shows weights that are all positive and relatively uniform across maturities, signifying a parallel shift in the yield curve.
    *   **PC2 (Slope)** exhibits positive weights for short maturities and negative weights for long maturities, indicating that it captures the steepening or flattening behavior.
    *   **PC3 (Curvature)** displays weights that are positive at short and long ends but negative in the middle, representing changes in the convexity or "bow" of the yield curve.
    These shapes perfectly align with their financial interpretations, providing an intuitive understanding of the fundamental drivers of yield curve movements.
    """)

    # Markdown Cell: Step 6: Transforming Data to Principal Component Space
    st.markdown(r"""
    ### Step 6: Transforming Data to Principal Component Space

    Once the principal components (eigenvectors) are identified, the original centered data can be projected onto these new orthogonal axes. This transformation creates a new dataset, often called the "scores" or "transformed data," where each column corresponds to a principal component. These scores represent the daily values or "magnitudes" of the level, slope, and curvature movements affecting the yield curve.

    The transformed data $Y$ is obtained by multiplying the centered data $X_{\text{centered}}$ by the matrix of selected principal components $W$:

    $$ Y = X_{\text{centered}} W $$

    Here, $W$ would typically contain the top $k$ eigenvectors as its columns. In our case, we'll transform the data using all components for completeness, but we can later select a subset for reconstruction.
    """)

    # Transform the centered data using all sorted principal components
    transformed_yield_data = transform_data(centered_yield_data, sorted_eigenvectors)

    st.subheader("Transformed Yield Data (first 5 days, PCA scores):")
    st.dataframe(transformed_yield_data.head())

    # Markdown Cell: The table above displays...
    st.markdown(r"""
    The table above displays the first few rows of the transformed data. Each column now represents the "score" for a specific principal component on a given day. For example, the 'PC 1' column indicates the daily magnitude of the 'level' movement in the yield curve, 'PC 2' for the 'slope', and so on. These scores are uncorrelated and capture the underlying daily movements in a more compact and interpretable form.
    """)

    # Markdown Cell: Yield Curve Reconstruction
    st.markdown(r"""
    ### Yield Curve Reconstruction

    One of the powerful aspects of PCA is the ability to reconstruct the original data using a subset of the principal components. This demonstrates how much information about the original yield curve shape is captured by the selected components. By using only the top principal components, we can essentially denoise the data and isolate the most significant movements.

    The reconstruction process is the inverse of the transformation. If $Y_k$ are the scores from the top $k$ principal components, and $W_k$ are the corresponding $k$ eigenvectors, then the reconstructed centered data $\hat{X}_{\text{centered}}$ is:

    $$ \hat{X}_{\text{centered}} = Y_k W_k^T $$

    To get the reconstructed yield curve $\hat{X}$, we add back the original mean yields:

    $$ \hat{X} = \hat{X}_{\text{centered}} + \text{mean}(X) $$
    """)

    # Select a specific day (e.g., the last day) to illustrate reconstruction
    day_index_static = yield_data.index[-1]
    original_curve_day_static = yield_data.loc[day_index_static]

    # Reconstruct the yield curve using 3 principal components
    reconstructed_curve_3pc_static = reconstruct_yield_curve(
        transformed_yield_data, sorted_eigenvectors, mean_yields, num_components=3
    )
    reconstructed_curve_day_3pc_static = reconstructed_curve_3pc_static.loc[day_index_static]

    st.subheader(f"Original Yield Curve for Day {day_index_static.strftime('%Y-%m-%d')}:")
    st.write(original_curve_day_static)
    st.subheader(f"Reconstructed Yield Curve (using 3 PCs) for Day {day_index_static.strftime('%Y-%m-%d')}:")
    st.write(reconstructed_curve_day_3pc_static)

    # Plot for visual comparison
    fig_static, ax_static = plt.subplots(figsize=(10, 6))
    ax_static.plot(maturities, original_curve_day_static, marker='o', label='Original Yield Curve')
    ax_static.plot(maturities, reconstructed_curve_day_3pc_static, marker='x', linestyle='--', label='Reconstructed (3 PCs)')
    ax_static.set_title(f"Yield Curve Reconstruction for Day {day_index_static.strftime('%Y-%m-%d')} (Using 3 Principal Components)")
    ax_static.set_xlabel('Maturity (Years)')
    ax_static.set_ylabel('Yield (%)')
    ax_static.set_xticks(maturities)
    ax_static.grid(True)
    ax_static.legend()
    st.pyplot(fig_static)

    # Markdown Cell: The comparison between...
    st.markdown(r"""
    The comparison between the original and reconstructed yield curve for a specific day visually demonstrates the effectiveness of PCA. Using just three principal components, the reconstructed curve closely approximates the original, confirming that these components capture the essential shape and movements of the yield curve. The slight differences highlight the small amount of variance not captured by the top components.
    """)

    # Markdown Cell: Interactive Yield Curve Reconstruction
    st.markdown(r"""
    ### Interactive Yield Curve Reconstruction

    To further explore the impact of each principal component, we will create an interactive tool that allows you to reconstruct a yield curve using a chosen number of principal components (1, 2, or 3). This interactive feature will dynamically update the reconstructed curve, illustrating how adding each component refines the approximation of the original curve and captures more of its variance. You will observe how the 'level' component forms a basic curve, 'slope' adjusts its steepness, and 'curvature' fine-tunes its convexity.
    """)

    # Sidebar control for interactive reconstruction day
    day_options = yield_data.index.strftime('%Y-%m-%d').tolist()
    if 'selected_reconstruction_day_idx' not in st.session_state:
        st.session_state.selected_reconstruction_day_idx = 0 # Default to the first day

    selected_day_str = st.sidebar.selectbox(
        "Select Day for Interactive Reconstruction:",
        options=day_options,
        index=st.session_state.selected_reconstruction_day_idx,
        key="interactive_day_selector"
    )
    day_to_reconstruct_index = day_options.index(selected_day_str)
    st.session_state.selected_reconstruction_day_idx = day_to_reconstruct_index

    # Interactive function for reconstruction (adapted for Streamlit)
    def create_interactive_reconstruction_streamlit(
        original_data: pd.DataFrame,
        transformed_data: pd.DataFrame,
        principal_components: np.ndarray,
        mean_yields: pd.Series,
        maturities: list,
        day_to_reconstruct_idx: int
    ):
        original_curve_selected_day = original_data.iloc[day_to_reconstruct_idx]

        num_components_selected = st.slider(
            'Select Number of Principal Components for Reconstruction:',
            min_value=1, max_value=3, value=3, step=1,
            key="num_pcs_slider" # Ensure a unique key for the slider
        )

        reconstructed_curve_full_df = reconstruct_yield_curve(
            transformed_data, principal_components, mean_yields, num_components_selected
        )
        reconstructed_curve_selected_day = reconstructed_curve_full_df.iloc[day_to_reconstruct_idx]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(maturities, original_curve_selected_day, marker='o', label='Original Yield Curve', color='blue')
        ax.plot(maturities, reconstructed_curve_selected_day, marker='x', linestyle='--', color='red',
                 label=f'Reconstructed ({num_components_selected} PCs)')
        ax.set_title(f"Yield Curve Reconstruction (Using {num_components_selected} Principal Components) for {original_data.index[day_to_reconstruct_idx].strftime('%Y-%m-%d')}")
        ax.set_xlabel('Maturity (Years)')
        ax.set_ylabel('Yield (%)')
        ax.set_xticks(maturities)
        ax.grid(True)
        ax.legend()
        ax.set_ylim(original_data.min().min() - 0.5, original_data.max().max() + 0.5) # Consistent y-axis
        st.pyplot(fig)

    # Call the interactive reconstruction widget
    create_interactive_reconstruction_streamlit(
        yield_data,
        transformed_yield_data,
        sorted_eigenvectors,
        mean_yields,
        maturities,
        day_to_reconstruct_index
    )

    # Markdown Cell: Interact with the slider above...
    st.markdown(r"""
    Interact with the slider above to see how adding more principal components (Level, then Slope, then Curvature) progressively improves the reconstruction of the yield curve. Observe how:
    *   With **1 PC (Level)**, you get a basic, parallel shifted curve.
    *   With **2 PCs (Level + Slope)**, the steepness or flatness of the curve is accurately captured.
    *   With **3 PCs (Level + Slope + Curvature)**, the overall shape, including the convexity, is closely matched to the original curve, demonstrating the comprehensive explanatory power of these components.
    This interactive visualization provides a clear, intuitive understanding of how these primary factors collectively determine the shape and movements of yield curves.
    """)

    # Markdown Cell: Conclusion and Financial Implications
    st.markdown(r"""
    ### Conclusion and Financial Implications

    This notebook has provided a step-by-step guide to applying Principal Component Analysis (PCA) for the decomposition of yield curves, a critical task for Financial Data Engineers. We have demonstrated how to:

    1.  Generate realistic synthetic yield curve data.
    2.  Perform the core PCA algorithm steps: data centering, covariance matrix computation, eigendecomposition, and component sorting.
    3.  Visually interpret the explained variance, confirming that a small number of principal components capture the vast majority of yield curve movements (typically 95-99%).
    4.  Identify and financially interpret the first three principal components as **Level, Slope, and Curvature**, relating them to broad monetary policy, economic growth expectations, and market dynamics.
    5.  Reconstruct yield curves using a selected number of principal components, highlighting their contribution to the overall curve shape.

    The ability to decompose yield curves into these fundamental, orthogonal factors is invaluable for Financial Data Engineers. It simplifies complex yield curve dynamics, enabling more robust:
    *   **Risk Management:** Understanding how portfolio sensitivity to level, slope, and curvature changes.
    *   **Hedging Strategies:** Constructing hedges against specific types of yield curve movements.
    *   **Scenario Analysis:** Modeling the impact of changes in these fundamental factors on bond portfolios.
    *   **Economic Interpretation:** Gaining deeper insights into market expectations embedded within the yield curve.

    By using PCA, we transform high-dimensional, correlated yield data into a low-dimensional, uncorrelated set of factors, making analysis and interpretation significantly more tractable and intuitive for decision-making in financial markets.
    """)
```

To run the application, navigate to the `your_project_folder` directory in your terminal and execute:

```console
streamlit run app.py
```

This command will open the Streamlit application in your web browser, typically at `http://localhost:8501`.

The application's sidebar provides configuration options:
*   **Number of Business Days:** Controls the length of the simulated historical data.
*   **Maturities:** Allows you to define the bond maturities (e.g., 0.25, 1, 5, 10, 30 years) for which yield data will be simulated.

You'll also notice the use of the `@st.cache_data` decorator on several functions in `pca_yield_curve_decomposition.py`. This is a powerful Streamlit feature that caches the output of a function, improving performance by preventing re-execution if the input parameters haven't changed. This is particularly useful for computationally intensive steps like data simulation and PCA calculations.

## Simulating and Visualizing Yield Curve Data
Duration: 0:15

Before we dive into PCA, we need yield curve data. For this codelab, we'll simulate synthetic historical yield curve data to ensure reproducible results and highlight the underlying factors. The application uses a model that generates base rate, slope, and curvature components, which are then combined to produce daily yield curves across specified maturities.

The `simulate_yield_curve_data` function is responsible for this:

```python
@st.cache_data
def simulate_yield_curve_data(num_days: int, maturities: list) -> pd.DataFrame:
    np.random.seed(42) # For reproducibility

    # Simulate underlying factors
    base_rate = np.cumsum(np.random.normal(0, 0.05, num_days)) + 2.0
    slope_factor = np.cumsum(np.random.normal(0, 0.02, num_days)) - 0.5
    curvature_factor = np.cumsum(np.random.normal(0, 0.01, num_days)) + 0.1

    yield_curves = []
    for i in range(num_days):
        daily_yields = []
        for m in maturities:
            # Formula mimicking Nelson-Siegel or Svensson-like factors
            y = base_rate[i] + slope_factor[i] * (np.exp(-m/5) - 0.5) + curvature_factor[i] * (4 * m * np.exp(-m/10) - 1.0)
            daily_yields.append(max(0.01, y)) # Ensure non-negative yields
        yield_curves.append(daily_yields)

    df = pd.DataFrame(yield_curves, columns=[f'{m}-year' for m in maturities])
    df.index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_days, freq='B')) # Business days
    return df
```

In the main application, after calling this function, you will see the first few rows of the simulated data and its descriptive statistics:

```python
st.subheader("Simulated Yield Curve Data (first 5 days):")
st.dataframe(yield_data.head())

st.subheader("Descriptive Statistics of Simulated Yields:")
st.dataframe(yield_data.describe())
```

The data frame will show daily yield rates for each specified maturity (e.g., 0.25-year, 1-year, 5-year, etc.). The descriptive statistics provide an overview of the range, mean, and standard deviation of yields for each maturity, indicating their historical variability.

Next, it's crucial to visualize the historical movements of these simulated yield curves. The `plot_yield_curves_timeseries` function generates a time series plot:

```python
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
```

Observe the "Simulated Historical Yield Curves Over Time" plot in the Streamlit application. You should see multiple lines, each representing a different maturity, fluctuating over time.
<aside class="positive">
<b>Observation:</b> This plot helps us visually confirm that yield curves don't move independently. You'll likely notice periods of **parallel shifts** (all lines moving up or down together) and **steepening/flattening** (short-term and long-term rates moving in opposite directions or at different magnitudes). These are the systematic movements that PCA aims to isolate and quantify.
</aside>

## PCA Step 1: Data Centering
Duration: 0:10

The first fundamental step in PCA is **data centering**. This involves adjusting the data so that each feature (in our case, each maturity's yield) has a mean of zero. Centering is essential because PCA works by identifying directions of maximum variance. If the data is not centered, the first principal component might simply represent the mean of the data rather than the true direction of maximum variance.

Mathematically, if $X$ is your data matrix (where each column represents a maturity and each row represents an observation day), and $\text{mean}(X)$ is a row vector containing the mean of each column, then the centered data $X_{\text{centered}}$ is calculated as:

$$ X_{\text{centered}} = X - \text{mean}(X) $$

The application implements this with the `center_data` function:

```python
@st.cache_data
def center_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    mean_yields = data.mean()
    centered_data = data - mean_yields
    return centered_data, mean_yields
```

After centering, the application displays the first few rows of the `centered_yield_data` and confirms that the mean of each column is now approximately zero.

```python
st.subheader("Centered Yield Curve Data (first 5 days):")
st.dataframe(centered_yield_data.head())

st.subheader("Mean of Centered Data Columns:")
st.write(centered_yield_data.mean())
```
<aside class="positive">
<b>Verification:</b> In the Streamlit app, verify that the `Mean of Centered Data Columns` output shows values very close to zero for all maturities. This confirms successful data centering.
</aside>

## PCA Step 2: Computing the Covariance Matrix
Duration: 0:10

Once the data is centered, the next step is to compute the **covariance matrix**. This matrix is crucial because it summarizes how much each pair of maturities varies together.
*   A **positive covariance** between two maturities indicates that they tend to move in the same direction (e.g., if one yield increases, the other also tends to increase).
*   A **negative covariance** suggests they move in opposite directions.
*   The **diagonal elements** of the covariance matrix represent the variance of each individual maturity, indicating its absolute volatility.

For a centered data matrix $X_{\text{centered}}$ with $n$ observations (days) and $m$ features (maturities), the covariance matrix $C$ is computed as:

$$ C = \frac{1}{n-1} X_{\text{centered}}^T X_{\text{centered}} $$

Here, $X_{\text{centered}}^T$ is the transpose of the centered data matrix. This results in a square matrix of dimensions $m \times m$, where $m$ is the number of maturities.

The application uses the `compute_covariance_matrix` function:

```python
@st.cache_data
def compute_covariance_matrix(centered_data: pd.DataFrame) -> np.ndarray:
    covariance_matrix = np.cov(centered_data, rowvar=False)
    return covariance_matrix
```

The resulting covariance matrix is then displayed:

```python
st.subheader("Covariance Matrix:")
st.dataframe(pd.DataFrame(covariance_matrix, index=maturities, columns=maturities))
```
<aside class="positive">
<b>Observation:</b> In the Streamlit app, examine the covariance matrix. You should see relatively large positive values for off-diagonal elements, especially between adjacent maturities, reinforcing the idea that yield curves move in a highly correlated fashion. The diagonal values will represent the variance of each maturity.
</aside>

## PCA Step 3: Performing Eigendecomposition
Duration: 0:10

**Eigendecomposition** is the mathematical heart of PCA. It's the process of decomposing the covariance matrix into a set of **eigenvectors** and **eigenvalues**.

*   **Eigenvectors:** These are the principal components themselves. They represent orthogonal (uncorrelated) directions in the original feature space along which the data exhibits the most variance. Each eigenvector points in a direction that captures a certain pattern of movement across all maturities.
*   **Eigenvalues:** Each eigenvector has a corresponding eigenvalue. The eigenvalue quantifies the amount of variance in the data that is captured by its associated eigenvector. A larger eigenvalue means that its eigenvector explains more of the total variance in the dataset.

For a symmetric matrix like our covariance matrix $C$, eigendecomposition finds a matrix of eigenvectors $V$ and a diagonal matrix of eigenvalues $\Lambda$ such that:

$$ C = V \Lambda V^T $$

Where $V$ contains the eigenvectors as its columns, and $\Lambda$ is a diagonal matrix where the diagonal entries are the eigenvalues.

The application performs this step using the `perform_eigendecomposition` function:

```python
@st.cache_data
def perform_eigendecomposition(covariance_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    return eigenvalues, eigenvectors
```

The unsorted eigenvalues and eigenvectors are then displayed:

```python
st.subheader("Eigenvalues (Unsorted):")
st.write(eigenvalues)
st.subheader("Eigenvectors (Unsorted, columns are eigenvectors):")
st.write(eigenvectors)
```
<aside class="negative">
<b>Important Note:</b> The output of `np.linalg.eig` might not return eigenvectors in any particular order of significance. The next step is crucial for ordering them by their explanatory power.
</aside>

## PCA Steps 4 & 5: Sorting Components and Explained Variance
Duration: 0:15

After obtaining the eigenvalues and eigenvectors, the next critical steps in PCA are to **sort them by significance** and then **calculate the explained variance**.

First, we sort the eigenvalues in descending order. The eigenvector corresponding to the largest eigenvalue is the **first principal component**, capturing the most variance in the data. Subsequent eigenvectors, ordered by their eigenvalues, capture progressively less variance.

The `sort_eigen_components` function handles this:

```python
@st.cache_data
def sort_eigen_components(eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    return sorted_eigenvalues, sorted_eigenvectors
```

Once sorted, we can quantify how much variance each principal component explains, both individually and cumulatively. The `calculate_explained_variance` function computes these ratios:

```python
@st.cache_data
def calculate_explained_variance(sorted_eigenvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total_variance = np.sum(sorted_eigenvalues)
    explained_variance_ratio = sorted_eigenvalues / total_variance
    cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
    return explained_variance_ratio, cumulative_explained_variance_ratio
```

The application will display the sorted eigenvalues, individual explained variance ratios, and cumulative explained variance ratios.
<aside class="positive">
<b>Key Insight:</b> For yield curve decomposition, it is a well-established empirical finding that the first three principal components typically explain an overwhelming portion (often **95% to 99%**) of the total yield curve movements. This is a powerful demonstration of PCA's ability to simplify complex data.
</aside>

To visualize this, the `plot_explained_variance` function creates a "scree plot," which is a combination of a bar chart (individual explained variance) and a line chart (cumulative explained variance):

```python
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
```
Observe the "Explained Variance by Principal Component" plot. You should clearly see the first component dominating, followed by the second and third. The cumulative line should quickly rise and cross the 95% and 99% thresholds with just a few components, confirming their collective explanatory power.

## Interpreting Principal Component Shapes: Level, Slope, and Curvature
Duration: 0:15

One of the most valuable aspects of PCA in finance is the interpretability of the principal components. For yield curves, the first three principal components have widely recognized financial meanings: **Level, Slope, and Curvature**. When we plot the eigenvectors (principal components) against the maturities, their shapes reveal these interpretations.

The `plot_principal_component_shapes` function visualizes these shapes:

```python
def plot_principal_component_shapes(principal_components: np.ndarray, maturities: list, num_to_plot: int = 3):
    fig, ax = plt.subplots(figsize=(10, 6))
    component_names = ['Level', 'Slope', 'Curvature']
    for i in range(min(num_to_plot, principal_components.shape[1])):
        ax.plot(maturities, principal_components[:, i], marker='o', label=f'PC {i+1}: {component_names[i] if i < len(component_names) else ""}')
    ax.set_title('Principal Component Shapes (Eigenvectors)')
    ax.set_xlabel('Maturity (Years)')
    ax.set_ylabel('Component Weight')
    ax.set_xticks(maturities)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
```
Look at the "Principal Component Shapes (Eigenvectors)" plot in the Streamlit app. You will see three distinct curves:

1.  **First Principal Component (PC 1) - Level:** This eigenvector typically shows weights that are all positive and relatively uniform across all maturities. This shape signifies a **parallel shift** in the yield curve, where all yields move up or down by similar amounts. Financially, this reflects broad market sentiment, general interest rate changes, or inflation expectations impacting all maturities. It accounts for the largest share of variance (80-90%).

2.  **Second Principal Component (PC 2) - Slope:** This eigenvector typically shows positive weights for short maturities and negative weights for long maturities, or vice-versa. This shape captures the **steepening or flattening** of the yield curve, reflecting changes in the spread between short-term and long-term rates. Financially, it's often linked to expectations about future monetary policy, economic growth, or recessionary pressures. It accounts for 5-15% of the variance.

3.  **Third Principal Component (PC 3) - Curvature:** This eigenvector typically displays weights that are positive at the short and long ends but negative in the middle (or the opposite pattern). This "bow" shape represents changes in the **convexity** or "butterfly" movement of the yield curve, where medium-term rates move differently relative to short- and long-term rates. Financially, it can be related to market expectations about intermediate-term economic conditions or supply/demand dynamics at specific maturities. It accounts for 1-5% of the variance.

<aside class="positive">
<b>Connect the Dots:</b> These three components provide a powerful, low-dimensional representation of almost all yield curve movements, making it much easier to analyze and model. This is a cornerstone for many quantitative finance applications.
</aside>

## PCA Step 6: Transforming Data to Principal Component Space
Duration: 0:10

Now that we have identified the principal components (the sorted eigenvectors), we can transform our original centered data into this new, lower-dimensional space. This process is called **projection**, and the resulting data points are known as "scores" or "transformed data." Each column in this new dataset corresponds to a principal component, and each value represents the daily "magnitude" or "exposure" to that specific component's movement (Level, Slope, or Curvature).

The transformed data $Y$ is obtained by multiplying the centered data $X_{\text{centered}}$ by the matrix of selected principal components $W$:

$$ Y = X_{\text{centered}} W $$

Here, $W$ is a matrix where each column is one of the sorted principal components (eigenvectors).

The application uses the `transform_data` function for this:

```python
@st.cache_data
def transform_data(centered_data: pd.DataFrame, principal_components: np.ndarray) -> pd.DataFrame:
    transformed_data = centered_data @ principal_components
    return pd.DataFrame(transformed_data, index=centered_data.index,
                        columns=[f'PC {i+1}' for i in range(principal_components.shape[1])])
```

The first few rows of the transformed data are displayed:

```python
st.subheader("Transformed Yield Data (first 5 days, PCA scores):")
st.dataframe(transformed_yield_data.head())
```
<aside class="positive">
<b>Observation:</b> The `Transformed Yield Data` table now shows daily scores for each PC. For example, a large positive value in the 'PC 1' column for a given day would indicate a significant upward parallel shift (increase in 'Level') of the yield curve on that day. These scores are uncorrelated, meaning that the level, slope, and curvature movements are now separated and can be analyzed independently.
</aside>

## Yield Curve Reconstruction (Static and Interactive)
Duration: 0:20

One of the powerful features of PCA is the ability to **reconstruct** the original data using a subset of the principal components. This demonstrates how much information about the original yield curve shape is captured by the selected components. By using only the top principal components, we can effectively denoise the data and isolate the most significant, systematic movements.

The reconstruction process is the inverse of the transformation. If $Y_k$ are the scores from the top $k$ principal components, and $W_k$ are the corresponding $k$ eigenvectors, then the reconstructed *centered* data $\hat{X}_{\text{centered}}$ is:

$$ \hat{X}_{\text{centered}} = Y_k W_k^T $$

To get the full reconstructed yield curve $\hat{X}$, we must add back the original mean yields (which were subtracted during the centering step):

$$ \hat{X} = \hat{X}_{\text{centered}} + \text{mean}(X) $$

The `reconstruct_yield_curve` function implements this:

```python
@st.cache_data
def reconstruct_yield_curve(transformed_data: pd.DataFrame, principal_components: np.ndarray,
                            mean_yields: pd.Series, num_components: int) -> pd.DataFrame:
    selected_transformed_data = transformed_data.iloc[:, :num_components]
    selected_principal_components = principal_components[:, :num_components]
    reconstructed_centered_data = selected_transformed_data @ selected_principal_components.T
    reconstructed_curve = reconstructed_centered_data + mean_yields
    return reconstructed_curve
```

The application first provides a static reconstruction for a chosen day (e.g., the last day in the simulated data), comparing the original curve with one reconstructed using 3 principal components.

```python
st.subheader(f"Original Yield Curve for Day {day_index_static.strftime('%Y-%m-%d')}:")
st.write(original_curve_day_static)
st.subheader(f"Reconstructed Yield Curve (using 3 PCs) for Day {day_index_static.strftime('%Y-%m-%d')}:")
st.write(reconstructed_curve_day_3pc_static)

# Plot for visual comparison
fig_static, ax_static = plt.subplots(figsize=(10, 6))
ax_static.plot(maturities, original_curve_day_static, marker='o', label='Original Yield Curve')
ax_static.plot(maturities, reconstructed_curve_day_3pc_static, marker='x', linestyle='--', label='Reconstructed (3 PCs)')
ax_static.set_title(f"Yield Curve Reconstruction for Day {day_index_static.strftime('%Y-%m-%d')} (Using 3 Principal Components)")
ax_static.set_xlabel('Maturity (Years)')
ax_static.set_ylabel('Yield (%)')
ax_static.set_xticks(maturities)
ax_static.grid(True)
ax_static.legend()
st.pyplot(fig_static)
```
<aside class="positive">
<b>Visual Confirmation:</b> The plot "Yield Curve Reconstruction (Using 3 Principal Components)" visually confirms that even with just three components, the reconstructed curve closely matches the original. This highlights that Level, Slope, and Curvature effectively capture the essential shape and movements of the yield curve.
</aside>

### Interactive Yield Curve Reconstruction

To truly appreciate the contribution of each principal component, the application offers an interactive reconstruction tool. This feature allows you to select a specific day and then adjust the number of principal components (1, 2, or 3) used for reconstruction using a slider.

The Streamlit sidebar provides a dropdown to select the day for reconstruction, and a slider on the main page controls the number of PCs.

```python
# Sidebar control for interactive reconstruction day
day_options = yield_data.index.strftime('%Y-%m-%d').tolist()
# ... (session state and selectbox logic)

# Interactive function for reconstruction (adapted for Streamlit)
def create_interactive_reconstruction_streamlit(
    original_data: pd.DataFrame,
    transformed_data: pd.DataFrame,
    principal_components: np.ndarray,
    mean_yields: pd.Series,
    maturities: list,
    day_to_reconstruct_idx: int
):
    original_curve_selected_day = original_data.iloc[day_to_reconstruct_idx]

    num_components_selected = st.slider(
        'Select Number of Principal Components for Reconstruction:',
        min_value=1, max_value=3, value=3, step=1,
        key="num_pcs_slider"
    )

    reconstructed_curve_full_df = reconstruct_yield_curve(
        transformed_data, principal_components, mean_yields, num_components_selected
    )
    reconstructed_curve_selected_day = reconstructed_curve_full_df.iloc[day_to_reconstruct_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(maturities, original_curve_selected_day, marker='o', label='Original Yield Curve', color='blue')
    ax.plot(maturities, reconstructed_curve_selected_day, marker='x', linestyle='--', color='red',
             label=f'Reconstructed ({num_components_selected} PCs)')
    ax.set_title(f"Yield Curve Reconstruction (Using {num_components_selected} Principal Components) for {original_data.index[day_to_reconstruct_idx].strftime('%Y-%m-%d')}")
    ax.set_xlabel('Maturity (Years)')
    ax.set_ylabel('Yield (%)')
    ax.set_xticks(maturities)
    ax.grid(True)
    ax.legend()
    ax.set_ylim(original_data.min().min() - 0.5, original_data.max().max() + 0.5) # Consistent y-axis
    st.pyplot(fig)

# Call the interactive reconstruction widget
create_interactive_reconstruction_streamlit(
    yield_data,
    transformed_yield_data,
    sorted_eigenvectors,
    mean_yields,
    maturities,
    day_to_reconstruct_index
)
```
Interact with the `Select Number of Principal Components for Reconstruction` slider in the Streamlit application.
*   With **1 PC (Level)**: The reconstructed curve will appear as a largely parallel shift of the mean curve, capturing the overall height but lacking steepness or curvature.
*   With **2 PCs (Level + Slope)**: The reconstructed curve will now accurately reflect the general steepness or flatness of the original curve.
*   With **3 PCs (Level + Slope + Curvature)**: The reconstructed curve will closely approximate the original curve's exact shape, including its convexity or "bow."

This interactive experience provides an intuitive understanding of how the Level, Slope, and Curvature components incrementally contribute to forming the complex shape of the yield curve.

## Conclusion and Financial Implications
Duration: 0:05

This codelab has provided a comprehensive guide to applying Principal Component Analysis (PCA) for the decomposition of yield curves using a Streamlit application. We've walked through each essential step of the PCA algorithm, from simulating financial data to interpreting the results with direct financial implications.

Here's a summary of what we've covered:

1.  **Data Generation & Visualization:** We started by creating synthetic yield curve data and visualizing its historical movements to observe the inherent correlations and patterns.
2.  **Core PCA Algorithm:** We meticulously applied the steps of PCA:
    *   **Data Centering:** Removing the mean from each maturity's yield.
    *   **Covariance Matrix Computation:** Quantifying the relationships between different maturities.
    *   **Eigendecomposition:** Extracting eigenvalues and eigenvectors from the covariance matrix.
    *   **Sorting & Explained Variance:** Ordering components by their explanatory power and demonstrating that a few components capture most of the variance.
3.  **Interpretation of Components:** We identified and financially interpreted the first three principal components as **Level, Slope, and Curvature**, which are the fundamental drivers of yield curve movements.
4.  **Transformation & Reconstruction:** We transformed the data into the principal component space (PCA scores) and demonstrated how to reconstruct yield curves using a chosen number of components, showcasing their collective power in representing the original curve.

For **Financial Data Engineers**, the ability to decompose yield curves into these orthogonal, interpretable factors is invaluable. It transforms a high-dimensional, complex problem into a lower-dimensional, more manageable one, leading to more robust and insightful financial analysis.

The direct financial implications include:

*   **Enhanced Risk Management:** Portfolios can be hedged more effectively against specific types of yield curve movements (e.g., insulate a portfolio from parallel shifts or changes in slope). This allows for a granular understanding of interest rate risk.
*   **Targeted Hedging Strategies:** Instead of hedging each maturity individually, one can construct hedges against Level, Slope, and Curvature risk, significantly simplifying complex hedging strategies.
*   **Sophisticated Scenario Analysis:** Financial models can simulate the impact of specific changes in Level, Slope, or Curvature on asset valuations and portfolio performance, providing deeper insights than simple parallel shifts.
*   **Improved Economic Interpretation:** Changes in the scores of Level, Slope, and Curvature can be directly linked to broader economic factors, monetary policy shifts, and market expectations, offering a clearer picture of market sentiment.

By mastering PCA for yield curve decomposition, Financial Data Engineers gain a powerful tool to demystify complex market dynamics, enabling more informed decision-making and the development of sophisticated quantitative strategies in fixed-income markets.
