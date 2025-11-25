Here's a comprehensive `README.md` file for your Streamlit application lab project, formatted in Markdown.

---

# Yield Curve Decomposition with PCA - Streamlit Lab

![QuLab Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)
*(Note: The logo displayed is from QuantUniversity, as referenced in `app.py`. You may replace this with your project's specific logo.)*

## Project Title and Description

This project, titled "Yield Curve Decomposition with Principal Component Analysis (PCA)", is an interactive Streamlit application designed as a lab environment for **Financial Data Engineers**. It provides a hands-on exploration of how Principal Component Analysis (PCA) can be effectively applied to complex yield curve data.

The application guides users through the entire PCA process, from the generation of synthetic yield curve data and its initial visualization, to the in-depth interpretation of key principal components—namely **Level**, **Slope**, and **Curvature**. A core feature includes an interactive tool for reconstructing yield curves using a varying number of principal components, illustrating their individual and combined impact on the curve's shape.

**Learning Objectives addressed by this lab:**
*   Understand the theoretical foundations and practical applications of PCA in finance.
*   Execute the core PCA steps programmatically (data centering, covariance matrix, eigendecomposition, sorting).
*   Interpret the financial significance of the Level, Slope, and Curvature components.
*   Analyze and visualize the contribution of each principal component to yield curve shapes.
*   Gain insights into PCA's role in risk management, hedging, and economic analysis within financial markets.

## Features

This Streamlit application offers the following key functionalities:

*   **Synthetic Data Generation**: Simulate realistic historical yield curve data for various maturities and time periods.
*   **Configurable Parameters**: Adjust the number of business days and a list of maturities for data simulation directly from the sidebar.
*   **Data Visualization**: Plot the time series of simulated historical yield curves to observe trends and volatilities.
*   **Step-by-Step PCA Implementation**:
    *   **Data Centering**: Perform and display the results of centering the yield data.
    *   **Covariance Matrix Computation**: Calculate and present the covariance matrix of the centered data.
    *   **Eigendecomposition**: Compute and show the eigenvalues and eigenvectors of the covariance matrix.
    *   **Component Sorting**: Sort eigenvalues and eigenvectors by explained variance.
*   **Explained Variance Analysis**: Calculate and visualize the individual and cumulative explained variance ratio of each principal component, including thresholds for 95% and 99% variance capture.
*   **Principal Component Interpretation**: Plot and interpret the shapes of the first three principal components as Level, Slope, and Curvature, explaining their financial significance.
*   **Data Transformation**: Project the centered yield data onto the principal component space to obtain PCA scores.
*   **Yield Curve Reconstruction**:
    *   Demonstrate the reconstruction of a specific day's yield curve using a fixed number of principal components.
    *   **Interactive Reconstruction Tool**: A dynamic slider allows users to select 1, 2, or 3 principal components to reconstruct a chosen day's yield curve, visually demonstrating how each component contributes to approximating the original curve's shape.
*   **Comprehensive Explanations**: Detailed markdown sections guide users through each theoretical and practical aspect of the PCA process and its financial implications.

## Getting Started

Follow these instructions to set up and run the Streamlit application on your local machine.

### Prerequisites

Ensure you have the following installed:
*   **Python 3.8+** (recommended)
*   **pip** (Python package installer)

### Installation

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
    *(If this is a local lab project without a Git repository, simply navigate to the directory containing `app.py` and `application_pages/`.)*

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    The application relies on several Python libraries. Install them using pip:
    ```bash
    pip install streamlit numpy pandas matplotlib seaborn scipy
    ```
    Alternatively, create a `requirements.txt` file with the following content and install:
    ```
    streamlit
    numpy
    pandas
    matplotlib
    seaborn
    scipy
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the Streamlit application:

1.  **Navigate to the project's root directory** in your terminal where `app.py` is located.
2.  **Execute the Streamlit command:**
    ```bash
    streamlit run app.py
    ```
3.  **Open in Browser**: Streamlit will automatically open a new tab in your default web browser (or provide a local URL, typically `http://localhost:8501`).

### Basic Interaction

*   **Sidebar Navigation**: Use the "Navigation" dropdown in the sidebar to select different sections or labs (currently, only "Yield Curve Decomposition with PCA" is available).
*   **Configuration Parameters**: In the sidebar, under "Configuration", you can adjust:
    *   **Number of Business Days**: Changes the length of the simulated historical data.
    *   **Maturities**: Define the specific bond maturities (e.g., 0.25, 1, 5, 10, 30 years) for which yields are simulated.
*   **Interactive Reconstruction**: Scroll down to the "Interactive Yield Curve Reconstruction" section.
    *   Use the **"Select Day for Interactive Reconstruction"** dropdown in the sidebar to pick a specific historical day.
    *   Use the **"Select Number of Principal Components for Reconstruction"** slider within the main content area to dynamically observe the impact of Level, Slope, and Curvature components on the reconstructed yield curve.

## Project Structure

```
.
├── app.py
└── application_pages/
    └── pca_yield_curve_decomposition.py
```

*   `app.py`: The main entry point for the Streamlit application. It sets up the page configuration, displays the main title and description, and handles navigation to different lab pages.
*   `application_pages/`: A directory intended to hold individual Streamlit lab pages.
    *   `pca_yield_curve_decomposition.py`: Contains the core logic for the Yield Curve Decomposition with PCA lab. This file defines all the functions for data simulation, PCA steps, visualization, and the interactive components specific to this analysis.

## Technology Stack

The application is built using the following technologies:

*   **Python**: The primary programming language.
*   **Streamlit**: For creating interactive web applications with pure Python.
*   **NumPy**: Essential for numerical operations, especially array manipulations and linear algebra.
*   **Pandas**: Used for data manipulation and analysis, particularly with DataFrames.
*   **Matplotlib**: For generating static, interactive, and animated visualizations in Python.
*   **Seaborn**: A data visualization library based on Matplotlib, providing a high-level interface for drawing attractive statistical graphics.
*   **SciPy**: Provides more advanced scientific computing tools, though `numpy.linalg.eig` is used directly for eigendecomposition in this specific code.

## Contributing

As this is a lab project, direct contributions via pull requests might not be the primary mechanism. However, if you wish to extend or improve it:

1.  Fork the repository (if hosted on GitHub/GitLab).
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes and ensure they adhere to the existing code style.
4.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
5.  Push to the branch (`git push origin feature/AmazingFeature`).
6.  Open a Pull Request (if applicable).

For bug reports or feature suggestions, please open an issue in the project's issue tracker.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(Note: A `LICENSE` file would typically be present in the root directory of a real project.)*

## Contact

For any questions or feedback regarding this lab project, please contact:

*   **Project Maintainer**: [Your Name/Organization Name]
*   **Email**: [your.email@example.com]
*   **Website/Organization**: [https://www.quantuniversity.com](https://www.quantuniversity.com) (or your relevant project/organization link)

---