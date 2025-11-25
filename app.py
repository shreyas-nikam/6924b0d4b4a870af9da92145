
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
st.title("QuLab: Yield Curve Decomposition with PCA")
st.divider()

# Your code starts here

st.markdown("""
In this lab, we explore the powerful application of Principal Component Analysis (PCA) to decompose yield curve data. Financial Data Engineers can leverage PCA to transform complex, high-dimensional yield curve movements into a few interpretable underlying factors: Level, Slope, and Curvature. This interactive tool guides you through each step of the PCA algorithm, from data simulation to yield curve reconstruction, providing clear financial insights into the dynamics of bond markets.
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
st.session_state["maturities"] = maturities


# Placeholder for custom data upload
# st.sidebar.subheader("Upload Custom Data (Optional)")
# uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
# if uploaded_file:
#     # Logic to process uploaded file (not in current scope, placeholder)
#     st.sidebar.info("Custom data upload functionality is a future enhancement.")


# --- Functions from Jupyter Notebook (Adapted for Streamlit Caching) ---

# Define function to simulate yield curve data
@st.cache_data
def simulate_yield_curve_data_cached(num_days: int, maturities: list) -> pd.DataFrame:
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
def center_data_cached(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    mean_yields = data.mean()
    centered_data = data - mean_yields
    return centered_data, mean_yields

# Define function to compute the covariance matrix
@st.cache_data
def compute_covariance_matrix_cached(centered_data: pd.DataFrame) -> np.ndarray:
    covariance_matrix = np.cov(centered_data, rowvar=False)
    return covariance_matrix

# Define function to perform eigendecomposition
@st.cache_data
def perform_eigendecomposition_cached(covariance_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    return eigenvalues, eigenvectors

# Define function to sort eigenvalues and eigenvectors
@st.cache_data
def sort_eigen_components_cached(eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    return sorted_eigenvalues, sorted_eigenvectors

# Define function to calculate explained variance
@st.cache_data
def calculate_explained_variance_cached(sorted_eigenvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total_variance = np.sum(sorted_eigenvalues)
    explained_variance_ratio = sorted_eigenvalues / total_variance
    cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
    return explained_variance_ratio, cumulative_explained_variance_ratio

# Define function to transform data
@st.cache_data
def transform_data_cached(centered_data: pd.DataFrame, principal_components: np.ndarray) -> pd.DataFrame:
    transformed_data = centered_data @ principal_components
    return pd.DataFrame(transformed_data, index=centered_data.index,
                        columns=[f'PC {i+1}' for i in range(principal_components.shape[1])])

# Perform all core calculations and store in session state
if "yield_data" not in st.session_state or \
   st.session_state["num_days_input"] != num_days_input or \
   st.session_state["maturities_input_str"] != maturities_input_str:

    st.session_state["num_days_input"] = num_days_input
    st.session_state["maturities_input_str"] = maturities_input_str

    st.session_state["yield_data"] = simulate_yield_curve_data_cached(num_days_input, maturities)
    st.session_state["centered_yield_data"], st.session_state["mean_yields"] = \
        center_data_cached(st.session_state["yield_data"])
    st.session_state["covariance_matrix"] = \
        compute_covariance_matrix_cached(st.session_state["centered_yield_data"])
    st.session_state["eigenvalues"], st.session_state["eigenvectors"] = \
        perform_eigendecomposition_cached(st.session_state["covariance_matrix"])
    st.session_state["sorted_eigenvalues"], st.session_state["sorted_eigenvectors"] = \
        sort_eigen_components_cached(st.session_state["eigenvalues"], st.session_state["eigenvectors"])
    st.session_state["explained_variance_ratio"], st.session_state["cumulative_explained_variance_ratio"] = \
        calculate_explained_variance_cached(st.session_state["sorted_eigenvalues"])
    st.session_state["transformed_yield_data"] = \
        transform_data_cached(st.session_state["centered_yield_data"], st.session_state["sorted_eigenvectors"])


page = st.sidebar.selectbox(
    label="Navigation",
    options=[
        "Introduction & Data",
        "PCA Steps 1-5",
        "PCA Interpretation & Transformation",
        "Yield Curve Reconstruction"
    ]
)

if page == "Introduction & Data":
    from application_pages.introduction_and_data import main
    main()
elif page == "PCA Steps 1-5":
    from application_pages.pca_steps_1_to_5 import main
    main()
elif page == "PCA Interpretation & Transformation":
    from application_pages.pca_interpretation_and_transformation import main
    main()
elif page == "Yield Curve Reconstruction":
    from application_pages.yield_curve_reconstruction import main
    main()

# Your code ends here
