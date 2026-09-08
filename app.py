import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EDA Interface", page_icon="📈", layout="wide")

# Clean, professional styling avoiding generic template fluff
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
    selected_column = st.sidebar.selectbox("Select Attribute", df.columns)

    st.markdown("### Dataset Overview")
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Rows", f"{df.shape[0]:,}")
    col_b.metric("Columns", f"{df.shape[1]}")
    col_c.metric("Missing Values", f"{df.isnull().sum().sum():,}")

    st.markdown("#### Preview (First 5 Rows)")
    st.dataframe(df.head(), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Data Types")
        st.dataframe(pd.DataFrame(df.dtypes, columns=["Type"]).astype(str), width="stretch", height=200)
    with c2:
        st.markdown("#### Missing Values")
        miss = pd.DataFrame({"Missing": df.isnull().sum(), "Percent": (df.isnull().sum() / len(df)) * 100})
        st.dataframe(miss.round(2), width="stretch", height=200)

    st.markdown("#### Statistical Summary")
    st.dataframe(df.describe().T[["mean", "50%", "min", "max"]].rename(columns={"50%": "median"}), width="stretch")

    st.markdown("---")
    st.markdown("### Visualization Canvas")

    if selected_column:
        data = df[selected_column].dropna()
        fig, ax = plt.subplots(figsize=(9, 4.5))
        
        if pd.api.types.is_numeric_dtype(data) and data.nunique() > 10:
            sns.histplot(data, kde=True, ax=ax, color="#2563eb", edgecolor="none", alpha=0.6)
            ax.set_title(f"Distribution of {selected_column}", fontsize=11, fontweight='600')
            ax.set_xlabel(selected_column, fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
        else:
            counts = data.value_counts()
            ax.bar(counts.index.astype(str), counts.values, color="#3b82f6", width=0.6)
            ax.set_title(f"Frequency of {selected_column}", fontsize=11, fontweight='600')
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