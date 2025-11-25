
# Technical Specification for Jupyter Notebook: Yield Curve Decomposer - PCA in Action

## 1. Notebook Overview

**Learning Goals:**
This Jupyter notebook aims to provide Financial Data Engineers with a comprehensive and interactive educational experience on applying Principal Component Analysis (PCA) to decompose yield curves. Upon completing this notebook, users will be able to:

*   Understand the theoretical foundations and practical steps of the PCA algorithm, including data centering, covariance matrix computation, and eigendecomposition.
*   Apply PCA to simulated historical yield curve data to extract key underlying drivers.
*   Visually identify and interpret the financial significance of the first three principal components (level, slope, and curvature) within the context of yield curve movements.
*   Explore the impact of these principal components on yield curve reconstruction and quantify the variance explained by each component.
*   Gain an intuitive understanding of how PCA effectively reduces dimensionality while preserving maximum variance in financial time series data.

**Target Audience:**
This notebook is specifically designed for **Financial Data Engineers**. It assumes a foundational understanding of financial markets, basic statistics, and familiarity with Python programming concepts. The content focuses on enhancing their analytical toolkit for understanding complex financial structures like yield curve dynamics using advanced statistical techniques.

## 2. Code Requirements

**List of Expected Libraries:**

*   `numpy`: For numerical operations, especially array manipulations and linear algebra.
*   `pandas`: For data manipulation and time series handling.
*   `matplotlib.pyplot`: For static plotting and visualization.
*   `seaborn`: For enhanced statistical data visualization.
*   `ipywidgets`: For creating interactive controls.
*   `scipy.linalg`: Specifically for `eig` function for eigendecomposition.

**List of Algorithms or Functions to be Implemented (without their code implementations):**

1.  `simulate_yield_curve_data(num_days: int, maturities: list) -> pandas.DataFrame`:
    *   Generates a synthetic `pandas.DataFrame` of historical yield curve data.
    *   `num_days`: Number of historical data points (e.g., 252 for a year of trading days).
    *   `maturities`: A list of maturity tenors (e.g., `[0.25, 1, 5, 10, 30]` for 3-month, 1-year, 5-year, 10-year, 30-year).
    *   The data generation process should incorporate a stochastic element (e.g., random walk for a base rate) and ensure that longer maturities generally move with shorter ones but allow for varying spreads to simulate realistic yield curve dynamics and implicitly contain level, slope, and curvature movements.

2.  `center_data(data: pandas.DataFrame) -> tuple[pandas.DataFrame, pandas.Series]`:
    *   Centers the input `pandas.DataFrame` by subtracting the mean of each column (maturity).
    *   Returns the centered data as a `pandas.DataFrame` and the original means of each column as a `pandas.Series`.

3.  `compute_covariance_matrix(centered_data: pandas.DataFrame) -> numpy.ndarray`:
    *   Computes the covariance matrix of the centered data.
    *   Returns a `numpy.ndarray` representing the covariance matrix.

4.  `perform_eigendecomposition(covariance_matrix: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]`:
    *   Performs eigendecomposition on the covariance matrix.
    *   Returns a tuple: `(eigenvalues: numpy.ndarray, eigenvectors: numpy.ndarray)`.
    *   Eigenvectors should be column vectors, such that `eigenvectors[:, i]` is the eigenvector corresponding to `eigenvalues[i]`.

5.  `sort_eigen_components(eigenvalues: numpy.ndarray, eigenvectors: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]`:
    *   Sorts eigenvalues in descending order and reorders eigenvectors accordingly.
    *   Returns a tuple: `(sorted_eigenvalues: numpy.ndarray, sorted_eigenvectors: numpy.ndarray)`.

6.  `calculate_explained_variance(sorted_eigenvalues: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]`:
    *   Calculates the explained variance ratio and cumulative explained variance ratio from sorted eigenvalues.
    *   Returns a tuple: `(explained_variance_ratio: numpy.ndarray, cumulative_explained_variance_ratio: numpy.ndarray)`.

7.  `transform_data(centered_data: pandas.DataFrame, principal_components: numpy.ndarray) -> pandas.DataFrame`:
    *   Projects the centered data onto the selected principal components.
    *   `principal_components`: The selected eigenvectors (as a `numpy.ndarray`, where columns are components).
    *   Returns the transformed data (scores) as a `pandas.DataFrame`.

8.  `reconstruct_yield_curve(transformed_data: pandas.DataFrame, principal_components: numpy.ndarray, mean_yields: pandas.Series, num_components: int) -> pandas.DataFrame`:
    *   Reconstructs the yield curve using a specified number of principal components.
    *   `transformed_data`: The data projected onto all principal components.
    *   `principal_components`: All sorted eigenvectors.
    *   `mean_yields`: The original mean yields for each maturity, used to uncenter the data.
    *   `num_components`: The number of top principal components to use for reconstruction.
    *   Returns the reconstructed yield curves as a `pandas.DataFrame`.

**Visualization like charts, tables, plots that should be generated:**

1.  **Original Yield Curve Time Series Plot**:
    *   Type: Line plot.
    *   Content: Time series of yield rates for each maturity.
    *   Features: Legend for maturities, x-axis label 'Date' or 'Time', y-axis label 'Yield (%)'.

2.  **Principal Component Shapes Plot**:
    *   Type: Line plot with multiple subplots or overlaid lines.
    *   Content: The first three principal component vectors plotted against maturities. Each component should be clearly labeled (Level, Slope, Curvature).
    *   Features: x-axis label 'Maturity', y-axis label 'Component Weight'.

3.  **Explained Variance Plot**:
    *   Type: Bar plot for individual explained variance ratio, overlaid with a line plot for cumulative explained variance ratio.
    *   Content: Each principal component's contribution to total variance and the cumulative sum.
    *   Features: x-axis label 'Principal Component Index', y-axis label 'Explained Variance Ratio', clear labeling for individual and cumulative plots.

4.  **Interactive Yield Curve Reconstruction Plot**:
    *   Type: Line plot, dynamically updating based on `ipywidgets` selection.
    *   Content: Compares the original yield curve (or an arbitrarily selected day's curve) with its reconstruction using 1, 2, or 3 principal components.
    *   Features: Slider to select the number of principal components. Plots of the original curve and the reconstructed curve, with clear labels and a title indicating the number of components used.

## 3. Notebook Sections (in detail)

---

### Section 1: Introduction to Yield Curve Decomposition and Principal Component Analysis (PCA)

*   **Markdown Cell:**
    ```markdown
    ### Understanding Yield Curve Decomposition with PCA

    The yield curve is a fundamental indicator of economic health and market expectations, representing the relationship between the yield on bonds of the same credit quality but different maturities. Analyzing its movements is crucial for financial professionals. However, yield curves are high-dimensional objects, making direct analysis challenging.

    **Principal Component Analysis (PCA)** is a powerful dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving the maximum possible variance [7]. In finance, PCA is extensively used to decompose complex structures, such as yield curves, into a few interpretable underlying factors. This notebook will guide Financial Data Engineers through the application of PCA to yield curve data, revealing its dominant drivers: level, slope, and curvature [8].

    The core idea is to identify orthogonal (uncorrelated) components, known as principal components, that capture the most significant variations in the data. For yield curves, these components correspond to economically meaningful movements. We will follow the PCA algorithm steps as outlined in Figure 4 [7].
    ```

---

### Section 2: Generating Synthetic Historical Yield Curve Data

*   **Markdown Cell:**
    ```markdown
    ### Generating Synthetic Yield Curve Data

    To demonstrate PCA on yield curves, we will first generate synthetic historical yield curve data. This simulated dataset will include yields for various maturities, such as 3-month, 1-year, 5-year, 10-year, and 30-year bonds. The generation process aims to mimic realistic yield curve dynamics, ensuring that the underlying factors (level, slope, curvature) can be clearly identified through PCA. We will simulate daily data for a period equivalent to one trading year (252 days).
    ```

*   **Code Cell (Function Implementation):**
    ```python
    # Define function to simulate yield curve data
    def simulate_yield_curve_data(num_days: int, maturities: list) -> pd.DataFrame:
        # Implementation details for generating realistic yield curve data with
        # a base rate, slope, and curvature components that evolve stochastically
        # and are combined to form the final yield curve rates.
        # Ensure yields are non-negative and exhibit some correlation across maturities.
        # Use a random seed for reproducibility.
        pass # Placeholder for function implementation
    ```

*   **Code Cell (Function Execution):**
    ```python
    # Set parameters for simulation
    num_days = 252
    maturities = [0.25, 1, 5, 10, 30] # 3-month, 1-year, 5-year, 10-year, 30-year

    # Generate the simulated yield curve data
    yield_data = simulate_yield_curve_data(num_days, maturities)

    # Display the first few rows of the generated data
    print("Simulated Yield Curve Data (first 5 days):")
    print(yield_data.head())

    # Display descriptive statistics
    print("\nDescriptive Statistics of Simulated Yields:")
    print(yield_data.describe())
    ```

*   **Markdown Cell:**
    ```markdown
    The synthetic yield curve data, displayed above, provides a realistic representation of how yield rates might evolve across different maturities over time. Notice the range and variability in yields, which are crucial for PCA to extract meaningful components. Each column represents a different maturity, and each row corresponds to a specific day.
    ```

---

### Section 3: Visualizing Original Yield Curve Data

*   **Markdown Cell:**
    ```markdown
    ### Visualizing Historical Yield Curve Movements

    Before applying PCA, it's beneficial to visualize the raw historical yield curve data. This allows us to observe the general trends, volatilities, and correlations between different maturities over the simulated period. We expect to see parallel shifts, steepening/flattening, and changes in convexity, which PCA will later decompose into distinct components.
    ```

*   **Code Cell (Function Implementation):**
    ```python
    # Define function to plot yield curve time series
    def plot_yield_curves_timeseries(data: pd.DataFrame, maturities: list):
        plt.figure(figsize=(12, 6))
        for i, maturity in enumerate(maturities):
            plt.plot(data.index, data.iloc[:, i], label=f'{maturity}-year')
        plt.title('Simulated Historical Yield Curves Over Time')
        plt.xlabel('Date Index')
        plt.ylabel('Yield (%)')
        plt.legend(title='Maturity')
        plt.grid(True)
        plt.show()
    ```

*   **Code Cell (Function Execution):**
    ```python
    # Plot the simulated yield curve data
    plot_yield_curves_timeseries(yield_data, maturities)
    ```

*   **Markdown Cell:**
    ```markdown
    The plot above illustrates the time series of yields for each maturity. We can observe the typical behavior of yield curves, including periods where all rates move in the same direction (parallel shifts) and periods where the spreads between short and long-term rates change (steepening or flattening). These visual patterns confirm the presence of systematic movements that PCA aims to capture.
    ```

---

### Section 4: Step 1: Data Centering

*   **Markdown Cell:**
    ```markdown
    ### Step 1: Data Centering

    The first crucial step in PCA is to center the data. This involves subtracting the mean of each feature (in our case, each maturity) from its respective observations. Centering ensures that the first principal component explains the maximum variance, as it will pass through the origin in the transformed feature space. Without centering, the first principal component might simply correspond to the mean of the data rather than the direction of maximum variance.

    Mathematically, if $X$ is our data matrix (where each column is a maturity and each row is a day), and $\text{mean}(X)$ is a row vector of the means of each column, then the centered data $X_{\text{centered}}$ is calculated as:

    $$ X_{\text{centered}} = X - \text{mean}(X) $$
    ```

*   **Code Cell (Function Implementation):**
    ```python
    # Define function to center the data
    def center_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        mean_yields = data.mean()
        centered_data = data - mean_yields
        return centered_data, mean_yields
    ```

*   **Code Cell (Function Execution):**
    ```python
    # Center the yield data
    centered_yield_data, mean_yields = center_data(yield_data)

    # Display the first few rows of the centered data
    print("Centered Yield Curve Data (first 5 days):")
    print(centered_yield_data.head())

    # Verify that the mean of each column is approximately zero
    print("\nMean of Centered Data Columns:")
    print(centered_yield_data.mean())
    ```

*   **Markdown Cell:**
    ```markdown
    The table above shows the first five rows of the centered yield curve data. As expected, the mean of each maturity column in the `centered_yield_data` DataFrame is now very close to zero, confirming that the data has been correctly centered around its origin. This prepares our data for the next step: covariance matrix computation.
    ```

---

### Section 5: Step 2: Computing the Covariance Matrix

*   **Markdown Cell:**
    ```markdown
    ### Step 2: Computing the Covariance Matrix

    After centering the data, the next step is to compute the covariance matrix. The covariance matrix quantifies the degree to which each pair of maturities varies together. A positive covariance indicates that two maturities tend to move in the same direction, while a negative covariance suggests they move in opposite directions. The diagonal elements of the covariance matrix represent the variance of each individual maturity.

    For a centered data matrix $X_{\text{centered}}$ with $n$ observations (days) and $m$ features (maturities), the covariance matrix $C$ is computed as:

    $$ C = \frac{1}{n-1} X_{\text{centered}}^T X_{\text{centered}} $$

    Here, $X_{\text{centered}}^T$ is the transpose of the centered data matrix. This matrix will be square, with dimensions $m \times m$.
    ```

*   **Code Cell (Function Implementation):**
    ```python
    # Define function to compute the covariance matrix
    def compute_covariance_matrix(centered_data: pd.DataFrame) -> np.ndarray:
        covariance_matrix = np.cov(centered_data, rowvar=False) # rowvar=False means columns are variables
        return covariance_matrix
    ```

*   **Code Cell (Function Execution):**
    ```python
    # Compute the covariance matrix
    covariance_matrix = compute_covariance_matrix(centered_yield_data)

    # Display the covariance matrix
    print("Covariance Matrix:")
    print(pd.DataFrame(covariance_matrix, index=maturities, columns=maturities))
    ```

*   **Markdown Cell:**
    ```markdown
    The covariance matrix above shows the relationships between different maturities. The diagonal elements represent the variance of each maturity, indicating its volatility. The off-diagonal elements show the covariance between pairs of maturities. As anticipated, we observe positive covariances, indicating that yields across different maturities generally move in the same direction, reflecting the correlated nature of yield curve movements.
    ```

---

### Section 6: Step 3: Performing Eigendecomposition

*   **Markdown Cell:**
    ```markdown
    ### Step 3: Performing Eigendecomposition

    Eigendecomposition is the core mathematical operation in PCA. It decomposes the covariance matrix into a set of eigenvectors and eigenvalues.
    *   **Eigenvectors** represent the principal components. These are orthogonal (uncorrelated) directions in the original feature space along which the data varies the most. They define the new coordinate system.
    *   **Eigenvalues** quantify the amount of variance explained by each corresponding eigenvector. A larger eigenvalue means its associated eigenvector captures more of the data's variance.

    For a symmetric matrix like our covariance matrix $C$, eigendecomposition finds a matrix of eigenvectors $V$ and a diagonal matrix of eigenvalues $\Lambda$ such that:

    $$ C = V \Lambda V^T $$

    Where $V$ contains the eigenvectors as its columns, and $\Lambda$ is a diagonal matrix where the diagonal entries are the eigenvalues.
    ```

*   **Code Cell (Function Implementation):**
    ```python
    # Define function to perform eigendecomposition
    def perform_eigendecomposition(covariance_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        return eigenvalues, eigenvectors
    ```

*   **Code Cell (Function Execution):**
    ```python
    # Perform eigendecomposition on the covariance matrix
    eigenvalues, eigenvectors = perform_eigendecomposition(covariance_matrix)

    print("Eigenvalues (Unsorted):")
    print(eigenvalues)
    print("\nEigenvectors (Unsorted, columns are eigenvectors):")
    print(eigenvectors)
    ```

*   **Markdown Cell:**
    ```markdown
    The unsorted eigenvalues and eigenvectors obtained from the eigendecomposition are displayed above. The eigenvalues represent the variance explained by each principal component, and the eigenvectors define the directions of these components in the original maturity space. In the next step, we will sort these components by the magnitude of their eigenvalues to identify the most significant principal components.
    ```

---

### Section 7: Step 4 & 5: Sorting and Selecting Principal Components

*   **Markdown Cell:**
    ```markdown
    ### Step 4 & 5: Sorting and Selecting Principal Components

    To identify the most significant principal components, we need to sort the eigenvalues in descending order. The eigenvector corresponding to the largest eigenvalue is the first principal component, capturing the most variance in the data. Subsequent eigenvectors, ordered by their eigenvalues, capture progressively less variance.

    After sorting, we select the top $k$ principal components (eigenvectors) that collectively explain a significant portion of the total variance. For yield curve decomposition, it is well-established that the first three principal components typically explain **95% to 99%** of the total yield curve movements [8].
    ```

*   **Code Cell (Function Implementation):**
    ```python
    # Define function to sort eigenvalues and eigenvectors
    def sort_eigen_components(eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Sort eigenvalues in descending order and get the indices
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        return sorted_eigenvalues, sorted_eigenvectors

    # Define function to calculate explained variance
    def calculate_explained_variance(sorted_eigenvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        total_variance = np.sum(sorted_eigenvalues)
        explained_variance_ratio = sorted_eigenvalues / total_variance
        cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
        return explained_variance_ratio, cumulative_explained_variance_ratio
    ```

*   **Code Cell (Function Execution):**
    ```python
    # Sort the eigen components
    sorted_eigenvalues, sorted_eigenvectors = sort_eigen_components(eigenvalues, eigenvectors)

    # Calculate explained variance ratios
    explained_variance_ratio, cumulative_explained_variance_ratio = \
        calculate_explained_variance(sorted_eigenvalues)

    print("Sorted Eigenvalues:")
    print(sorted_eigenvalues)
    print("\nExplained Variance Ratio:")
    print(explained_variance_ratio)
    print("\nCumulative Explained Variance Ratio:")
    print(cumulative_explained_variance_ratio)
    ```

*   **Markdown Cell:**
    ```markdown
    The sorted eigenvalues and their corresponding explained variance ratios clearly show the dominance of the first few principal components. The cumulative explained variance ratio provides insight into how much of the total data variability is captured by a subset of these components. This confirms our expectation that a small number of components can explain a large proportion of yield curve movements.
    ```

---

### Section 8: Visualizing Explained Variance

*   **Markdown Cell:**
    ```markdown
    ### Visualizing Explained Variance

    Visualizing the explained variance ratio helps us determine the optimal number of principal components to retain. A "scree plot" (bar plot for individual variance and line plot for cumulative variance) is commonly used for this purpose. We expect to see that the first three components explain the vast majority of the variance, aligning with financial literature [8].
    ```

*   **Code Cell (Function Implementation):**
    ```python
    # Define function to plot explained variance
    def plot_explained_variance(explained_variance_ratio: np.ndarray, cumulative_explained_variance_ratio: np.ndarray):
        num_components = len(explained_variance_ratio)
        components_idx = np.arange(1, num_components + 1)

        plt.figure(figsize=(10, 6))
        plt.bar(components_idx, explained_variance_ratio, alpha=0.7, label='Individual Explained Variance')
        plt.plot(components_idx, cumulative_explained_variance_ratio, marker='o', linestyle='--', color='red', label='Cumulative Explained Variance')
        plt.axhline(y=0.95, color='gray', linestyle=':', label='95% Threshold')
        plt.axhline(y=0.99, color='darkgray', linestyle=':', label='99% Threshold')

        plt.title('Explained Variance by Principal Component')
        plt.xlabel('Principal Component Index')
        plt.ylabel('Explained Variance Ratio')
        plt.xticks(components_idx)
        plt.legend()
        plt.grid(True)
        plt.show()
    ```

*   **Code Cell (Function Execution):**
    ```python
    # Plot the explained variance
    plot_explained_variance(explained_variance_ratio, cumulative_explained_variance_ratio)
    ```

*   **Markdown Cell:**
    ```markdown
    The explained variance plot clearly demonstrates that the first three principal components indeed capture a substantial portion of the total variance in the yield curve data, typically between 95% and 99%. This reinforces the idea that yield curve movements can be effectively summarized by a small number of underlying factors, significantly reducing the dimensionality of the problem while retaining most of the important information.
    ```

---

### Section 9: Understanding Principal Component Shapes (Level, Slope, Curvature)

*   **Markdown Cell:**
    ```markdown
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
    ```

*   **Code Cell (Function Implementation):**
    ```python
    # Define function to plot principal component shapes
    def plot_principal_component_shapes(principal_components: np.ndarray, maturities: list, num_to_plot: int = 3):
        plt.figure(figsize=(10, 6))
        component_names = ['Level', 'Slope', 'Curvature']
        for i in range(min(num_to_plot, principal_components.shape[1])):
            plt.plot(maturities, principal_components[:, i], marker='o', label=f'PC {i+1}: {component_names[i] if i < len(component_names) else ""}')
        plt.title('Principal Component Shapes (Eigenvectors)')
        plt.xlabel('Maturity (Years)')
        plt.ylabel('Component Weight')
        plt.xticks(maturities)
        plt.grid(True)
        plt.legend()
        plt.show()
    ```

*   **Code Cell (Function Execution):**
    ```python
    # Plot the first three principal component shapes
    plot_principal_component_shapes(sorted_eigenvectors, maturities, num_to_plot=3)
    ```

*   **Markdown Cell:**
    ```markdown
    The plot clearly visualizes the shapes of the first three principal components.
    *   **PC1 (Level)** shows weights that are all positive and relatively uniform across maturities, signifying a parallel shift in the yield curve.
    *   **PC2 (Slope)** exhibits positive weights for short maturities and negative weights for long maturities, indicating that it captures the steepening or flattening behavior.
    *   **PC3 (Curvature)** displays weights that are positive at short and long ends but negative in the middle, representing changes in the convexity or "bow" of the yield curve.
    These shapes perfectly align with their financial interpretations, providing an intuitive understanding of the fundamental drivers of yield curve movements.
    ```

---

### Section 10: Step 6: Transforming Data to Principal Component Space

*   **Markdown Cell:**
    ```markdown
    ### Step 6: Transforming Data to Principal Component Space

    Once the principal components (eigenvectors) are identified, the original centered data can be projected onto these new orthogonal axes. This transformation creates a new dataset, often called the "scores" or "transformed data," where each column corresponds to a principal component. These scores represent the daily values or "magnitudes" of the level, slope, and curvature movements affecting the yield curve.

    The transformed data $Y$ is obtained by multiplying the centered data $X_{\text{centered}}$ by the matrix of selected principal components $W$:

    $$ Y = X_{\text{centered}} W $$

    Here, $W$ would typically contain the top $k$ eigenvectors as its columns. In our case, we'll transform the data using all components for completeness, but we can later select a subset for reconstruction.
    ```

*   **Code Cell (Function Implementation):**
    ```python
    # Define function to transform data
    def transform_data(centered_data: pd.DataFrame, principal_components: np.ndarray) -> pd.DataFrame:
        transformed_data = centered_data @ principal_components
        return pd.DataFrame(transformed_data, index=centered_data.index,
                            columns=[f'PC {i+1}' for i in range(principal_components.shape[1])])
    ```

*   **Code Cell (Function Execution):**
    ```python
    # Transform the centered data using all sorted principal components
    transformed_yield_data = transform_data(centered_yield_data, sorted_eigenvectors)

    # Display the first few rows of the transformed data
    print("Transformed Yield Data (first 5 days, PCA scores):")
    print(transformed_yield_data.head())
    ```

*   **Markdown Cell:**
    ```markdown
    The table above displays the first few rows of the transformed data. Each column now represents the "score" for a specific principal component on a given day. For example, the 'PC 1' column indicates the daily magnitude of the 'level' movement in the yield curve, 'PC 2' for the 'slope', and so on. These scores are uncorrelated and capture the underlying daily movements in a more compact and interpretable form.
    ```

---

### Section 11: Yield Curve Reconstruction

*   **Markdown Cell:**
    ```markdown
    ### Yield Curve Reconstruction

    One of the powerful aspects of PCA is the ability to reconstruct the original data using a subset of the principal components. This demonstrates how much information about the original yield curve shape is captured by the selected components. By using only the top principal components, we can essentially denoise the data and isolate the most significant movements.

    The reconstruction process is the inverse of the transformation. If $Y_k$ are the scores from the top $k$ principal components, and $W_k$ are the corresponding $k$ eigenvectors, then the reconstructed centered data $\hat{X}_{\text{centered}}$ is:

    $$ \hat{X}_{\text{centered}} = Y_k W_k^T $$

    To get the reconstructed yield curve $\hat{X}$, we add back the original mean yields:

    $$ \hat{X} = \hat{X}_{\text{centered}} + \text{mean}(X) $$
    ```

*   **Code Cell (Function Implementation):**
    ```python
    # Define function to reconstruct yield curve
    def reconstruct_yield_curve(transformed_data: pd.DataFrame, principal_components: np.ndarray,
                                mean_yields: pd.Series, num_components: int) -> pd.DataFrame:
        # Select the top 'num_components' from the transformed data and principal components
        selected_transformed_data = transformed_data.iloc[:, :num_components]
        selected_principal_components = principal_components[:, :num_components]

        # Reconstruct the centered data
        reconstructed_centered_data = selected_transformed_data @ selected_principal_components.T

        # Add back the mean yields to get the full reconstructed curve
        reconstructed_curve = reconstructed_centered_data + mean_yields

        return reconstructed_curve
    ```

*   **Code Cell (Function Execution):**
    ```python
    # Select a specific day (e.g., the last day) to illustrate reconstruction
    day_index = yield_data.index[-1]
    original_curve_day = yield_data.loc[day_index]

    # Reconstruct the yield curve using 3 principal components
    reconstructed_curve_3pc = reconstruct_yield_curve(
        transformed_yield_data, sorted_eigenvectors, mean_yields, num_components=3
    )
    reconstructed_curve_day_3pc = reconstructed_curve_3pc.loc[day_index]

    # Display original and reconstructed curve for a selected day
    print(f"Original Yield Curve for Day {day_index}:")
    print(original_curve_day)
    print(f"\nReconstructed Yield Curve (using 3 PCs) for Day {day_index}:")
    print(reconstructed_curve_day_3pc)

    # Plot for visual comparison
    plt.figure(figsize=(10, 6))
    plt.plot(maturities, original_curve_day, marker='o', label='Original Yield Curve')
    plt.plot(maturities, reconstructed_curve_day_3pc, marker='x', linestyle='--', label='Reconstructed (3 PCs)')
    plt.title(f'Yield Curve Reconstruction for Day {day_index} (Using 3 Principal Components)')
    plt.xlabel('Maturity (Years)')
    plt.ylabel('Yield (%)')
    plt.xticks(maturities)
    plt.grid(True)
    plt.legend()
    plt.show()
    ```

*   **Markdown Cell:**
    ```markdown
    The comparison between the original and reconstructed yield curve for a specific day visually demonstrates the effectiveness of PCA. Using just three principal components, the reconstructed curve closely approximates the original, confirming that these components capture the essential shape and movements of the yield curve. The slight differences highlight the small amount of variance not captured by the top components.
    ```

---

### Section 12: Interactive Yield Curve Reconstruction

*   **Markdown Cell:**
    ```markdown
    ### Interactive Yield Curve Reconstruction

    To further explore the impact of each principal component, we will create an interactive tool that allows you to reconstruct a yield curve using a chosen number of principal components (1, 2, or 3). This interactive feature will dynamically update the reconstructed curve, illustrating how adding each component refines the approximation of the original curve and captures more of its variance. You will observe how the 'level' component forms a basic curve, 'slope' adjusts its steepness, and 'curvature' fine-tunes its convexity.
    ```

*   **Code Cell (Function Implementation):**
    ```python
    # Define an interactive function for reconstruction
    def create_interactive_reconstruction(
        original_data: pd.DataFrame,
        transformed_data: pd.DataFrame,
        principal_components: np.ndarray,
        mean_yields: pd.Series,
        maturities: list,
        day_to_reconstruct_idx: int
    ):
        # Select a specific day's data for the interactive visualization
        original_curve_selected_day = original_data.iloc[day_to_reconstruct_idx]

        def update_reconstruction(num_components_selected):
            reconstructed_curve_full_df = reconstruct_yield_curve(
                transformed_data, principal_components, mean_yields, num_components_selected
            )
            reconstructed_curve_selected_day = reconstructed_curve_full_df.iloc[day_to_reconstruct_idx]

            plt.figure(figsize=(10, 6))
            plt.plot(maturities, original_curve_selected_day, marker='o', label='Original Yield Curve', color='blue')
            plt.plot(maturities, reconstructed_curve_selected_day, marker='x', linestyle='--', color='red',
                     label=f'Reconstructed ({num_components_selected} PCs)')
            plt.title(f'Yield Curve Reconstruction (Using {num_components_selected} Principal Components)')
            plt.xlabel('Maturity (Years)')
            plt.ylabel('Yield (%)')
            plt.xticks(maturities)
            plt.grid(True)
            plt.legend()
            plt.ylim(original_data.min().min() - 0.5, original_data.max().max() + 0.5) # Consistent y-axis
            plt.show()

        # Create an interactive slider widget
        interact(update_reconstruction, num_components_selected=IntSlider(min=1, max=3, step=1, value=3, description='Num PCs:'))
    ```

*   **Code Cell (Function Execution):**
    ```python
    from ipywidgets import interact, IntSlider
    import matplotlib.pyplot as plt

    # Choose a specific day for interactive reconstruction (e.g., the 50th day)
    day_to_reconstruct_index = 49 # Using 0-based index

    # Create the interactive reconstruction widget
    create_interactive_reconstruction(
        yield_data,
        transformed_yield_data,
        sorted_eigenvectors,
        mean_yields,
        maturities,
        day_to_reconstruct_index
    )
    ```

*   **Markdown Cell:**
    ```markdown
    Interact with the slider above to see how adding more principal components (Level, then Slope, then Curvature) progressively improves the reconstruction of the yield curve. Observe how:
    *   With **1 PC (Level)**, you get a basic, parallel shifted curve.
    *   With **2 PCs (Level + Slope)**, the steepness or flatness of the curve is accurately captured.
    *   With **3 PCs (Level + Slope + Curvature)**, the overall shape, including the convexity, is closely matched to the original curve, demonstrating the comprehensive explanatory power of these components.
    This interactive visualization provides a clear, intuitive understanding of how these primary factors collectively determine the shape and movements of yield curves.
    ```

---

### Section 13: Conclusion and Financial Implications

*   **Markdown Cell:**
    ```markdown
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
    ```

---
