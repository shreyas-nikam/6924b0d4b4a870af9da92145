
# Streamlit Application Specification: Yield Curve Decomposition with PCA

This document outlines the design and functional requirements for a Streamlit application focused on Principal Component Analysis (PCA) for yield curve decomposition, tailored for Financial Data Engineers.

## 1. Application Overview

This application will serve as an interactive educational and analytical tool for Financial Data Engineers to understand and apply PCA to yield curve data. It programmatically demonstrates each step of the PCA algorithm, from data generation to yield curve reconstruction, providing clear financial interpretations of the derived components.

**Learning Goals:**
*   Understand the theoretical and practical application of Principal Component Analysis (PCA) for dimensionality reduction in financial data.
*   Execute the fundamental steps of the PCA algorithm programmatically: data centering, covariance matrix computation, eigendecomposition, sorting, and data transformation.
*   Interpret the financial significance of the first three principal components of yield curve movements: Level, Slope, and Curvature.
*   Analyze the contribution of each principal component to the overall yield curve shape through interactive reconstruction.
*   Gain insights into how PCA can simplify complex yield curve dynamics for applications in risk management, hedging, and economic interpretation.

## 2. User Interface Requirements

The application will feature a clear, step-by-step layout guiding the user through the PCA process, with interactive elements for data input and visualization.

### Layout and Navigation Structure
*   **Sidebar:**
    *   **Data Configuration:**
        *   Input fields for `Number of Business Days` (for synthetic data generation).
        *   Input field for `Maturities` (comma-separated list of years).
        *   (Optional, placeholder for future development) File uploader for `Upload Custom Yield Data`.
    *   **Interactive Reconstruction Controls:**
        *   Dropdown/Selector for `Select Day for Interactive Reconstruction` (to pick a specific day from the generated/uploaded data for live plotting).
*   **Main Content Area:** Arranged into sequential sections, corresponding to the PCA workflow:
    1.  **Introduction:** Overview of yield curve decomposition with PCA.
    2.  **Data Generation & Visualization:** Display of simulated data and its time series plot.
    3.  **PCA Step 1: Data Centering:** Explanation and output of centered data.
    4.  **PCA Step 2: Covariance Matrix Computation:** Explanation and output of the covariance matrix.
    5.  **PCA Step 3: Performing Eigendecomposition:** Explanation and output of unsorted eigenvalues/eigenvectors.
    6.  **PCA Step 4 & 5: Sorting and Selecting Principal Components:** Explanation, sorted eigenvalues/explained variance, and a Scree Plot.
    7.  **Interpreting Principal Component Shapes:** Detailed financial interpretation for Level, Slope, and Curvature, accompanied by their characteristic plots.
    8.  **PCA Step 6: Transforming Data to Principal Component Space:** Explanation and output of transformed data (PCA scores).
    9.  **Yield Curve Reconstruction (Static):** Explanation and a static comparison plot of original vs. 3-PC reconstructed curve for a selected day.
    10. **Interactive Yield Curve Reconstruction:** Explanation and an interactive plot allowing real-time reconstruction using 1, 2, or 3 principal components.
    11. **Conclusion:** Summary of findings and financial implications.

### Input Widgets and Controls
*   **Number of Business Days (`num_days`):** `st.number_input` in sidebar. Default: 252, Min: 50, Max: 500.
*   **Maturities (`maturities`):** `st.text_input` in sidebar, accepting comma-separated float values. Default: "0.25, 1, 5, 10, 30".
*   **Upload Custom Data:** `st.file_uploader` in sidebar (optional/placeholder).
*   **Select Day for Interactive Reconstruction (`day_to_reconstruct_idx`):** `st.selectbox` in sidebar, populated with dates from the generated/uploaded data. Default: First day.
*   **Number of Principal Components (`num_components_selected`):** `st.slider` in main content area for the interactive reconstruction plot. Range: 1 to 3, Step: 1, Default: 3.

### Visualization Components
*   **Plot 1: Simulated Historical Yield Curves Over Time**
    *   **Type:** Line plot (matplotlib figure displayed with `st.pyplot`).
    *   **Description:** Illustrates the evolution of yields for various maturities over the simulated period.
    *   **Elements:** X-axis: 'Date Index', Y-axis: 'Yield (%)'. Multiple lines, each representing a maturity, with legends.
    *   **Title:** 'Simulated Historical Yield Curves Over Time'.
*   **Plot 2: Explained Variance by Principal Component (Scree Plot)**
    *   **Type:** Combination of bar chart (individual explained variance) and line chart (cumulative explained variance) (matplotlib figure).
    *   **Description:** Shows the proportion of total variance explained by each principal component and the cumulative sum.
    *   **Elements:** X-axis: 'Principal Component Index', Y-axis: 'Explained Variance Ratio'. Reference lines at 95% and 99% cumulative variance. Legends for individual and cumulative variance.
    *   **Title:** 'Explained Variance by Principal Component'.
*   **Plot 3: Principal Component Shapes (Eigenvectors)**
    *   **Type:** Line plot (matplotlib figure).
    *   **Description:** Visualizes the weights of the first three eigenvectors across different maturities, corresponding to Level, Slope, and Curvature.
    *   **Elements:** X-axis: 'Maturity (Years)', Y-axis: 'Component Weight'. Three distinct lines for PC1, PC2, PC3, labeled as 'Level', 'Slope', 'Curvature'.
    *   **Title:** 'Principal Component Shapes (Eigenvectors)'.
*   **Plot 4: Yield Curve Reconstruction (Static Comparison)**
    *   **Type:** Line plot (matplotlib figure).
    *   **Description:** Compares the original yield curve with its reconstruction using three principal components for a specific selected day.
    *   **Elements:** X-axis: 'Maturity (Years)', Y-axis: 'Yield (%)'. Two lines: 'Original Yield Curve' and 'Reconstructed (3 PCs)'.
    *   **Title:** 'Yield Curve Reconstruction for Day [Selected Day] (Using 3 Principal Components)'.
*   **Plot 5: Interactive Yield Curve Reconstruction**
    *   **Type:** Line plot (matplotlib figure), dynamically updated.
    *   **Description:** Allows users to select 1, 2, or 3 principal components and observe how the reconstructed yield curve approximates the original curve in real-time.
    *   **Elements:** X-axis: 'Maturity (Years)', Y-axis: 'Yield (%)'. Two lines: 'Original Yield Curve' and 'Reconstructed ([Num PCs Selected] PCs)'. Consistent Y-axis limits.
    *   **Title:** 'Yield Curve Reconstruction (Using [Num PCs Selected] Principal Components)'.

### Interactive Elements and Feedback Mechanisms
*   The `st.slider` for `num_components_selected` will trigger re-rendering of Plot 5, providing immediate visual feedback on the impact of adding more principal components.
*   All plots will include clear titles, axis labels, and legends for comprehensive understanding.
*   Descriptive text (markdown) will accompany each section, explaining the concepts and the significance of the outputs.

## 3. Additional Requirements

### Annotation and Tooltip Specifications
*   **Mathematical Explanations:** All equations from the notebook will be rendered using LaTeX within `st.markdown`, ensuring correct display ($...$ for inline, $$...$$ for display equations).
*   **PCA Step Explanations:** Each PCA step will be introduced with detailed markdown explanations from the notebook.
*   **Financial Interpretations:** The 'Level', 'Slope', and 'Curvature' components will be extensively explained in a dedicated markdown section and labeled clearly in Plot 3.
*   **Plot Annotations:** All plots will have descriptive titles, axis labels, and legends as specified in Section 2. Reference lines for 95% and 99% cumulative variance will be present on the Scree Plot.

### Save the States of the Fields Properly
*   Streamlit's `st.session_state` will be utilized to maintain the values of input widgets (e.g., `num_days`, `maturities`, `selected_day_for_reconstruction`) across user interactions and re-renders, preventing loss of input data.
*   Functions performing data generation and core PCA calculations will be decorated with `@st.cache_data` to cache results, improving performance by avoiding re-computation unless input parameters change.

## 4. Notebook Content and Code Requirements

This section extracts the essential markdown and Python code from the Jupyter notebook, outlining how it will be integrated into the Streamlit application. `st.pyplot(fig)` will replace `plt.show()` for all matplotlib figures. `@st.cache_data` will be applied to functions where appropriate to optimize performance.

```python
# Core imports for the Streamlit application
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eig

# Configure plots for better aesthetics globally
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100

# --- Application Sidebar (Configuration) ---
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


# --- Functions from Jupyter Notebook (Adapted for Streamlit Caching) ---

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

# --- Streamlit Application Main Content ---

st.title("Yield Curve Decomposition with Principal Component Analysis (PCA)")

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

# Generate the simulated yield curve data
yield_data = simulate_yield_curve_data(num_days_input, maturities)

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

# Compute the covariance matrix
covariance_matrix = compute_covariance_matrix(centered_yield_data)

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

# Perform eigendecomposition on the covariance matrix
eigenvalues, eigenvectors = perform_eigendecomposition(covariance_matrix)

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

# Markdown Cell: Interpreting Principal Component Shapes: Level, Slope, and Curvature
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

# Plot the first three principal component shapes
plot_principal_component_shapes(sorted_eigenvectors, maturities, num_to_plot=3)

# Markdown Cell: The plot clearly visualizes...
st.markdown("""
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
st.markdown("""
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
ax_static.set_title(f'Yield Curve Reconstruction for Day {day_index_static.strftime('%Y-%m-%d')} (Using 3 Principal Components)')
ax_static.set_xlabel('Maturity (Years)')
ax_static.set_ylabel('Yield (%)')
ax_static.set_xticks(maturities)
ax_static.grid(True)
ax_static.legend()
st.pyplot(fig_static)

# Markdown Cell: The comparison between...
st.markdown("""
The comparison between the original and reconstructed yield curve for a specific day visually demonstrates the effectiveness of PCA. Using just three principal components, the reconstructed curve closely approximates the original, confirming that these components capture the essential shape and movements of the yield curve. The slight differences highlight the small amount of variance not captured by the top components.
""")

# Markdown Cell: Interactive Yield Curve Reconstruction
st.markdown("""
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
    ax.set_title(f'Yield Curve Reconstruction (Using {num_components_selected} Principal Components) for {original_data.index[day_to_reconstruct_idx].strftime('%Y-%m-%d')}')
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
st.markdown("""
Interact with the slider above to see how adding more principal components (Level, then Slope, then Curvature) progressively improves the reconstruction of the yield curve. Observe how:
*   With **1 PC (Level)**, you get a basic, parallel shifted curve.
*   With **2 PCs (Level + Slope)**, the steepness or flatness of the curve is accurately captured.
*   With **3 PCs (Level + Slope + Curvature)**, the overall shape, including the convexity, is closely matched to the original curve, demonstrating the comprehensive explanatory power of these components.
This interactive visualization provides a clear, intuitive understanding of how these primary factors collectively determine the shape and movements of yield curves.
""")

# Markdown Cell: Conclusion and Financial Implications
st.markdown("""
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

