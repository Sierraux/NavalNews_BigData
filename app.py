import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Naval News Analytics", page_icon="🚢", layout="wide")

# --- LOAD CSS ---
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        # Load the dataset produced by the cleaned scraping process
        df = pd.read_csv("naval_news_cleaned.csv")
        return df
    except FileNotFoundError:
        st.error("Dataset 'naval_news_cleaned.csv' was not found. Please make sure the file is in the same folder.")
        return pd.DataFrame()

df = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🚢 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🗂️ Dataset", "📈 Visualization", "🤖 Prediction"])

st.sidebar.markdown("---")
st.sidebar.info("**Author:** Daffa Mashabi Akmal Syarif\n\n**Student ID:** 24.61.0282")

# ==========================================
# 1. PAGE: HOME
# ==========================================
if page == "🏠 Home":
    st.title("📊 Naval News Analytics Dashboard")
    st.markdown("""
    Welcome to the **Naval News Analytics Dashboard**! 
    This dashboard analyzes the structural characteristics of naval defense news articles to explore their relationship with article length.
    """)
    st.markdown("---")

    if not df.empty:
        st.subheader("Data Summary")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(label="Total Articles", value=f"{len(df):,}")
        with col2:
            st.metric(label="Total Authors", value=f"{df['author'].nunique():,}")
        with col3:
            st.metric(label="Average Word Count", value=f"{int(df['word_count'].mean())}")
        with col4:
            st.metric(label="Average Images", value=f"{int(df['num_images'].mean())}")
        with col5:
            st.metric(label="Average Paragraphs", value=f"{int(df['num_paragraphs'].mean())}")

# ==========================================
# 2. PAGE: DATASET
# ==========================================
elif page == "🗂️ Dataset":
    st.title("🗂️ Naval News Dataset")
    st.markdown("The table below shows the dataset that has gone through the web scraping and data cleaning process.")

    if not df.empty:
        # Search feature
        search_query = st.text_input("🔍 Search by article title...")
        if search_query:
            filtered_df = df[df['title'].str.contains(search_query, case=False, na=False)]
        else:
            filtered_df = df

        st.dataframe(filtered_df, use_container_width=True, height=400)

        # Download button
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Data as CSV",
            data=csv,
            file_name='naval_news_cleaned_view.csv',
            mime='text/csv',
        )

# ==========================================
# 3. PAGE: VISUALIZATION
# ==========================================
elif page == "📈 Visualization":
    st.title("📈 Data Visualization")

    if not df.empty:
        tab1, tab2, tab3, tab4 = st.tabs(["Word Count Histogram", "Scatter Plot", "Correlation Heatmap", "Top Authors"])

        with tab1:
            st.subheader("Distribution of Word Count")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(df["word_count"], kde=True, bins=30, color="steelblue", ax=ax)
            ax.set_title("Distribution of Article Word Count")
            st.pyplot(fig)

        with tab2:
            st.subheader("Scatter Plot: Images vs Words")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.regplot(data=df, x="num_images", y="word_count", scatter_kws={"alpha": 0.4}, line_kws={"color": "red"}, ax=ax)
            ax.set_title("Number of Images vs Number of Words")
            st.pyplot(fig)

        with tab3:
            st.subheader("Correlation Heatmap")
            feature_cols = ["word_count", "num_images", "num_paragraphs", "num_external_links", "title_word_count", "title_char_count"]
            corr_matrix = df[feature_cols].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True, ax=ax)
            ax.set_title("Correlation Heatmap of Article Features")
            st.pyplot(fig)

        with tab4:
            st.subheader("Top 10 Authors by Article Count")
            author_counts = df['author'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(y=author_counts.index, x=author_counts.values, palette="viridis", ax=ax)
            ax.set_title("Top 10 Authors by Number of Articles")
            st.pyplot(fig)

# ==========================================
# 4. PAGE: PREDICTION
# ==========================================
elif page == "🤖 Prediction":
    st.title("🤖 Word Count Prediction")
    st.markdown("Enter the article characteristics below to predict the article length using a **Multiple Linear Regression** model.")

    st.markdown("---")

    # Input data
    col1, col2 = st.columns(2)
    with col1:
        num_paragraphs = st.number_input("Number of Paragraphs", min_value=1, value=10, step=1)
        num_images = st.number_input("Number of Images", min_value=0, value=2, step=1)
    with col2:
        title_words = st.number_input("Number of Words in Title", min_value=1, value=10, step=1)
        ext_links = st.number_input("Number of External Links", min_value=0, value=1, step=1)

    st.markdown("---")

    if st.button("🚀 Predict Estimated Word Count"):
        # Try to load the model from pickle; if it fails, use the original coefficients from the notebook
        try:
            with open("model_multi.pkl", "rb") as f:
                model_multi = pickle.load(f)
            input_features = np.array([[num_images, num_paragraphs, ext_links, title_words]])
            pred_word_count = model_multi.predict(input_features)[0]
        except Exception as e:
            # Uses the intercept and coefficient values from the report as a fallback
            # Intercept: 47.785, images: 16.531, paragraphs: 40.882, links: 20.450, title: 1.776
            pred_word_count = 47.785 + (16.531 * num_images) + (40.882 * num_paragraphs) + (20.450 * ext_links) + (1.776 * title_words)
            st.warning("⚠️ Model file (model_multi.pkl) was not found. Using manually calculated coefficients from the Google Colab analysis.")

        st.success("Prediction completed successfully!")
        st.metric(label="Estimated Word Count", value=f"{int(pred_word_count)} Words")