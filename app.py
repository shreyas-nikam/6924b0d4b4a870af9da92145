"""Main Streamlit application file"""

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

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()

# Your code starts here
st.markdown("""
In this lab, we will explore the powerful application of Principal Component Analysis (PCA) to decompose yield curve movements into interpretable factors such as Level, Slope, and Curvature. This interactive tool is designed for Financial Data Engineers to gain a deeper understanding of yield curve dynamics and PCA in action.
""")

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

# Store maturities in session state
st.session_state.maturities = maturities

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

# Store functions in session state for access in pages
st.session_state.reconstruct_yield_curve = reconstruct_yield_curve


# Perform all data processing and store in session state
if 'yield_data' not in st.session_state or st.session_state.get('num_days_input_cached') != num_days_input or st.session_state.get('maturities_input_str_cached') != maturities_input_str:
    st.session_state.yield_data = simulate_yield_curve_data(num_days_input, maturities)
    st.session_state.centered_yield_data, st.session_state.mean_yields = center_data(st.session_state.yield_data)
    st.session_state.covariance_matrix = compute_covariance_matrix(st.session_state.centered_yield_data)
    st.session_state.eigenvalues, st.session_state.eigenvectors = perform_eigendecomposition(st.session_state.covariance_matrix)
    st.session_state.sorted_eigenvalues, st.session_state.sorted_eigenvectors = sort_eigen_components(st.session_state.eigenvalues, st.session_state.eigenvectors)
    st.session_state.explained_variance_ratio, st.session_state.cumulative_explained_variance_ratio = calculate_explained_variance(st.session_state.sorted_eigenvalues)
    st.session_state.transformed_yield_data = transform_data(st.session_state.centered_yield_data, st.session_state.sorted_eigenvectors)
    
    # Cache input values to detect changes
    st.session_state.num_days_input_cached = num_days_input
    st.session_state.maturities_input_str_cached = maturities_input_str


# Page navigation
page_names = [
    "Introduction",
    "Data Generation & Visualization",
    "PCA Step 1: Data Centering",
    "PCA Step 2: Covariance Matrix Computation",
    "PCA Step 3: Performing Eigendecomposition",
    "PCA Step 4 & 5: Sorting and Selecting Principal Components",
    "Interpreting Principal Component Shapes: Level, Slope, and Curvature",
    "PCA Step 6: Transforming Data to Principal Component Space",
    "Yield Curve Reconstruction (Static)",
    "Interactive Yield Curve Reconstruction",
    "Conclusion and Financial Implications"
]

page = st.sidebar.selectbox(label="Navigation", options=page_names)

if page == "Introduction":
    from application_pages.page_1_introduction import main
    main()
elif page == "Data Generation & Visualization":
    from application_pages.page_2_data_generation_visualization import main
    main()
elif page == "PCA Step 1: Data Centering":
    from application_pages.page_3_pca_step_1_data_centering import main
    main()
elif page == "PCA Step 2: Covariance Matrix Computation":
    from application_pages.page_4_pca_step_2_covariance_matrix import main
    main()
elif page == "PCA Step 3: Performing Eigendecomposition":
    from application_pages.page_5_pca_step_3_eigendecomposition import main
    main()
elif page == "PCA Step 4 & 5: Sorting and Selecting Principal Components":
    from application_pages.page_6_pca_step_4_5_sorting_selection import main
    main()
elif page == "Interpreting Principal Component Shapes: Level, Slope, and Curvature":
    from application_pages.page_7_interpreting_pc_shapes import main
    main()
elif page == "PCA Step 6: Transforming Data to Principal Component Space":
    from application_pages.page_8_pca_step_6_transforming_data import main
    main()
elif page == "Yield Curve Reconstruction (Static)":
    from application_pages.page_9_yield_curve_reconstruction_static import main
    main()
elif page == "Interactive Yield Curve Reconstruction":
    from application_pages.page_10_interactive_yield_curve_reconstruction import main
    main()
elif page == "Conclusion and Financial Implications":
    from application_pages.page_11_conclusion import main
    main()
# Your code ends here
