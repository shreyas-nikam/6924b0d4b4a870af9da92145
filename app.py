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