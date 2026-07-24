import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Naval News Analytics", page_icon="🚢", layout="wide")

# --- FILE PATHS (v3 dataset & model) ---
DATA_PATH = "naval_news_cleaned_v3.csv"
MODEL_PATH = "model_multi_v3.pkl"

# --- LOAD CSS ---
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# --- LOAD DATA ---
# NOTE: cache key includes the file's last-modified time, so replacing the CSV
# (even with the same filename) automatically invalidates the cache instead of
# silently serving a stale, previously-cached dataframe.
@st.cache_data
def load_data(path: str, _mtime: float):
    df = pd.read_csv(path)
    return df

if os.path.exists(DATA_PATH):
    df = load_data(DATA_PATH, os.path.getmtime(DATA_PATH))
else:
    st.error(
        f"Dataset '{DATA_PATH}' was not found. Make sure it has been pushed to "
        f"the repository and that the filename matches exactly."
    )
    df = pd.DataFrame()

# --- LOAD MODEL ---
@st.cache_resource
def load_model(path: str, _mtime: float):
    with open(path, "rb") as f:
        return pickle.load(f)

model_multi = None
model_error = None
if os.path.exists(MODEL_PATH):
    try:
        model_multi = load_model(MODEL_PATH, os.path.getmtime(MODEL_PATH))
    except Exception as e:
        model_error = str(e)
else:
    model_error = f"Model file '{MODEL_PATH}' was not found in the repository."

# --- Feature order MUST exactly match the training order in the Colab notebook (Section 7) ---
FEATURE_ORDER = [
    "num_images", "num_paragraphs", "num_external_links", "num_internal_links",
    "num_headings", "num_bold", "num_lists", "num_blockquotes", "num_captions",
    "title_word_count", "title_has_number", "is_weekend",
    "num_countries_mentioned", "num_ship_type_mentions", "is_submarine_related",
    "is_surface_combatant_related", "is_uncrewed_related", "num_weapon_system_mentions",
    "has_contract_mention", "has_exercise_mention",
]

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🚢 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🗂️ Dataset", "📈 Visualization", "🤖 Prediction"])

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh cached data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("**Author:** Daffa Mashabi Akmal Syarif\n\n**Student ID:** 24.61.0282")

# ==========================================
# 1. PAGE: HOME
# ==========================================
if page == "🏠 Home":
    st.title("📊 Naval News Analytics Dashboard")
    st.markdown(
        """
        Welcome to the **Naval News Analytics Dashboard**!
        This dashboard analyzes both the **structural** characteristics (images, paragraphs,
        headings, links...) and the **content** characteristics (countries mentioned, ship
        types, weapon systems discussed...) of naval defense news articles, and how they
        relate to article length (`word_count`).
        """
    )
    st.markdown("---")

    if not df.empty:
        st.subheader("Data Summary")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(label="Total Articles", value=f"{len(df):,}")
        with col2:
            st.metric(label="Total Authors", value=f"{df['author'].nunique():,}")
        with col3:
            st.metric(label="Avg. Word Count", value=f"{int(df['word_count'].mean())}")
        with col4:
            st.metric(label="Avg. Paragraphs", value=f"{df['num_paragraphs'].mean():.1f}")
        with col5:
            if "num_countries_mentioned" in df.columns:
                st.metric(label="Avg. Countries Mentioned", value=f"{df['num_countries_mentioned'].mean():.1f}")
            else:
                st.metric(label="Avg. Images", value=f"{df['num_images'].mean():.1f}")

        st.caption(f"Loaded from `{DATA_PATH}` — last modified {pd.Timestamp(os.path.getmtime(DATA_PATH), unit='s')}")
    else:
        st.warning("No data loaded yet — check the error message above.")

# ==========================================
# 2. PAGE: DATASET
# ==========================================
elif page == "🗂️ Dataset":
    st.title("🗂️ Naval News Dataset")
    st.markdown("The table below shows the dataset after web scraping and data cleaning.")

    if not df.empty:
        search_query = st.text_input("🔍 Search by article title...")
        if search_query:
            filtered_df = df[df["title"].str.contains(search_query, case=False, na=False)]
        else:
            filtered_df = df

        st.dataframe(filtered_df, use_container_width=True, height=400)

        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Data as CSV",
            data=csv,
            file_name="naval_news_cleaned_view.csv",
            mime="text/csv",
        )
    else:
        st.warning("No data to display.")

# ==========================================
# 3. PAGE: VISUALIZATION
# ==========================================
elif page == "📈 Visualization":
    st.title("📈 Data Visualization")

    if not df.empty:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Word Count Distribution", "Scatter Plot", "Correlation Heatmap",
             "Top Authors", "Content Insights"]
        )

        with tab1:
            st.subheader("Distribution of Word Count")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(df["word_count"], kde=True, bins=30, color="steelblue", ax=ax)
            ax.set_title("Distribution of Article Word Count")
            ax.set_xlabel("Word Count")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        with tab2:
            st.subheader("Scatter Plot: Paragraphs vs Words")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.regplot(
                data=df, x="num_paragraphs", y="word_count",
                scatter_kws={"alpha": 0.4}, line_kws={"color": "red"}, ax=ax,
            )
            ax.set_title("Number of Paragraphs vs Word Count")
            ax.set_xlabel("Number of Paragraphs")
            ax.set_ylabel("Word Count")
            st.pyplot(fig)

        with tab3:
            st.subheader("Correlation Heatmap")
            candidate_cols = [
                "word_count", "num_images", "num_paragraphs", "num_external_links",
                "num_internal_links", "num_headings", "num_bold", "num_lists",
                "num_blockquotes", "num_captions", "title_word_count",
                "num_countries_mentioned", "num_ship_type_mentions",
                "num_weapon_system_mentions",
            ]
            feature_cols = [c for c in candidate_cols if c in df.columns]
            corr_matrix = df[feature_cols].corr()
            fig, ax = plt.subplots(figsize=(11, 9))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True,
                        annot_kws={"size": 7}, ax=ax)
            ax.set_title("Correlation Heatmap of Article Features")
            st.pyplot(fig)

        with tab4:
            st.subheader("Top 10 Authors by Article Count")
            author_counts = df["author"].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(
                y=author_counts.index, x=author_counts.values,
                hue=author_counts.index, palette="viridis", legend=False, ax=ax,
            )
            ax.set_title("Top 10 Authors by Number of Articles")
            ax.set_xlabel("Number of Articles")
            ax.set_ylabel("Author")
            st.pyplot(fig)

        with tab5:
            if "primary_country" in df.columns:
                st.subheader("Most Frequently Discussed Countries")
                top_countries = df["primary_country"].value_counts().head(10)
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(
                    y=top_countries.index, x=top_countries.values,
                    hue=top_countries.index, palette="mako", legend=False, ax=ax,
                )
                ax.set_title("Top 10 Countries / Navies Mentioned in Articles")
                ax.set_xlabel("Number of Articles")
                ax.set_ylabel("Country")
                st.pyplot(fig)
            else:
                st.info("Content-based columns (e.g. `primary_country`) were not found in this dataset.")

    else:
        st.warning("No data to visualize.")

# ==========================================
# 4. PAGE: PREDICTION
# ==========================================
elif page == "🤖 Prediction":
    st.title("🤖 Word Count Prediction")
    st.markdown(
        "Enter the article's structural and content characteristics below to estimate "
        "its word count using the trained **Multiple Linear Regression** model."
    )

    if model_error:
        st.error(f"⚠️ Prediction model could not be loaded: {model_error}")
        st.stop()

    st.markdown("---")
    st.subheader("Structural features")
    col1, col2, col3 = st.columns(3)
    with col1:
        num_paragraphs = st.number_input("Number of Paragraphs", min_value=1, value=10, step=1)
        num_images = st.number_input("Number of Images", min_value=0, value=2, step=1)
        num_headings = st.number_input("Number of Headings (h2/h3)", min_value=0, value=1, step=1)
    with col2:
        num_bold = st.number_input("Number of Bold Text Elements", min_value=0, value=2, step=1)
        num_lists = st.number_input("Number of Lists (ul/ol)", min_value=0, value=0, step=1)
        num_blockquotes = st.number_input("Number of Blockquotes", min_value=0, value=0, step=1)
    with col3:
        num_captions = st.number_input("Number of Image Captions", min_value=0, value=1, step=1)
        num_external_links = st.number_input("Number of External Links", min_value=0, value=1, step=1)
        num_internal_links = st.number_input("Number of Internal Links", min_value=0, value=1, step=1)

    st.subheader("Title & publication")
    col4, col5, col6 = st.columns(3)
    with col4:
        title_word_count = st.number_input("Number of Words in Title", min_value=1, value=10, step=1)
    with col5:
        title_has_number = st.checkbox("Title contains a number (e.g. a year or model name)")
    with col6:
        is_weekend = st.checkbox("Published on a weekend")

    st.subheader("Content features")
    col7, col8, col9 = st.columns(3)
    with col7:
        num_countries_mentioned = st.number_input("Number of Countries/Navies Mentioned", min_value=0, value=1, step=1)
        num_ship_type_mentions = st.number_input("Ship Type Mentions", min_value=0, value=2, step=1)
    with col8:
        num_weapon_system_mentions = st.number_input("Weapon System Mentions", min_value=0, value=1, step=1)
        is_submarine_related = st.checkbox("Discusses submarines")
    with col9:
        is_surface_combatant_related = st.checkbox("Discusses surface combatants (frigate/destroyer/carrier)")
        is_uncrewed_related = st.checkbox("Discusses uncrewed/autonomous vessels")

    col10, col11 = st.columns(2)
    with col10:
        has_contract_mention = st.checkbox("Mentions a contract / acquisition / procurement")
    with col11:
        has_exercise_mention = st.checkbox("Mentions a military exercise / drill")

    st.markdown("---")

    if st.button("🚀 Predict Estimated Word Count"):
        input_values = {
            "num_images": num_images,
            "num_paragraphs": num_paragraphs,
            "num_external_links": num_external_links,
            "num_internal_links": num_internal_links,
            "num_headings": num_headings,
            "num_bold": num_bold,
            "num_lists": num_lists,
            "num_blockquotes": num_blockquotes,
            "num_captions": num_captions,
            "title_word_count": title_word_count,
            "title_has_number": int(title_has_number),
            "is_weekend": int(is_weekend),
            "num_countries_mentioned": num_countries_mentioned,
            "num_ship_type_mentions": num_ship_type_mentions,
            "is_submarine_related": int(is_submarine_related),
            "is_surface_combatant_related": int(is_surface_combatant_related),
            "is_uncrewed_related": int(is_uncrewed_related),
            "num_weapon_system_mentions": num_weapon_system_mentions,
            "has_contract_mention": int(has_contract_mention),
            "has_exercise_mention": int(has_exercise_mention),
        }

        # Build the input vector in the EXACT same order used during training
        input_features = np.array([[input_values[f] for f in FEATURE_ORDER]])

        try:
            pred_word_count = model_multi.predict(input_features)[0]
            st.success("Prediction completed successfully!")
            st.metric(label="Estimated Word Count", value=f"{int(pred_word_count)} words")
        except Exception as e:
            st.error(
                f"Prediction failed: {e}\n\n"
                "This usually means the loaded model was trained on a different set/order "
                "of features than FEATURE_ORDER above. Make sure model_multi_v3.pkl was "
                "trained with exactly this feature list and order."
            )