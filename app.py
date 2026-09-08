import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Matplotlib font config to avoid font weight warnings
plt.rcParams['font.weight'] = 'normal'

st.set_page_config(page_title="EDA Interface", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { font-size: 1.8rem; font-weight: 600; letter-spacing: -0.025em; }
    h3 { font-size: 1.2rem; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

st.title("Exploratory Data Analysis Interface")

st.sidebar.markdown("### Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filtering & Customization")
    
    enable_filtering = st.sidebar.checkbox("Filter Data by Column Value")
    filtered_df = df.copy()
    
    if enable_filtering:
        filter_col = st.sidebar.selectbox("Column to filter", df.columns)
        unique_vals = df[filter_col].dropna().unique()
        if len(unique_vals) < 50:
            selected_vals = st.sidebar.multiselect(f"Select values for {filter_col}", unique_vals, default=list(unique_vals))
            filtered_df = df[df[filter_col].isin(selected_vals)]
        else:
            st.sidebar.warning("Too many unique values for multiselect filtering.")

    selected_column = st.sidebar.selectbox("Select Attribute for Analysis", filtered_df.columns)

    st.markdown("### Dataset Overview")
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Rows", f"{filtered_df.shape[0]:,}", delta=f"{filtered_df.shape[0] - df.shape[0]}" if len(filtered_df) < len(df) else None)
    col_b.metric("Columns", f"{filtered_df.shape[1]}")
    col_c.metric("Missing Values", f"{filtered_df.isnull().sum().sum():,}")

    st.markdown("#### Preview (First 5 Rows)")
    st.dataframe(filtered_df.head(), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Data Types")
        st.dataframe(pd.DataFrame(filtered_df.dtypes, columns=["Type"]).astype(str), width="stretch", height=200)
    with c2:
        st.markdown("#### Missing Values")
        miss = pd.DataFrame({"Missing": filtered_df.isnull().sum(), "Percent": (filtered_df.isnull().sum() / len(filtered_df)) * 100})
        st.dataframe(miss.round(2), width="stretch", height=200)

    st.markdown("#### Statistical Summary")
    st.dataframe(filtered_df.describe().T[["mean", "50%", "min", "max"]].rename(columns={"50%": "median"}), width="stretch")

    st.markdown("---")
    st.markdown("### Interactive Visualization Canvas")

    if selected_column:
        data = filtered_df[selected_column].dropna()
        
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            color_theme = st.selectbox("Chart Color Theme", ["#2563eb", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"])
        with col_ctrl2:
            fig_height = st.slider("Chart Height", min_value=3.5, max_value=8.0, value=4.5, step=0.5)

        fig, ax = plt.subplots(figsize=(9, fig_height))
        
        if pd.api.types.is_numeric_dtype(data) and data.nunique() > 10:
            show_kde = st.checkbox("Show KDE Density Curve", value=True)
            bins_count = st.slider("Number of Bins", min_value=5, max_value=100, value=30)
            
            sns.histplot(data, kde=show_kde, bins=bins_count, ax=ax, color=color_theme, edgecolor="none", alpha=0.6)
            ax.set_title(f"Distribution of {selected_column}", fontsize=11, fontweight='bold')
            ax.set_xlabel(selected_column, fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
        else:
            top_n = st.slider("Limit Top Categories", min_value=5, max_value=50, value=10)
            counts = data.value_counts().head(top_n)
            
            ax.bar(counts.index.astype(str), counts.values, color=color_theme, width=0.6)
            ax.set_title(f"Frequency of {selected_column} (Top {top_n})", fontsize=11, fontweight='bold')
            ax.set_xlabel(selected_column, fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            plt.xticks(rotation=30, ha='right')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle=":", alpha=0.5)
        plt.tight_layout()
        
        st.pyplot(fig)
else:
    st.info("Upload a CSV dataset via the sidebar to initialize analysis.")