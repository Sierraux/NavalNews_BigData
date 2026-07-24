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
    if "published_date" in df.columns:
        df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
        df["publish_month"] = df["published_date"].dt.to_period("M").astype(str)
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

        st.caption(f"Loaded from `{DATA_PATH}`, last modified {pd.Timestamp(os.path.getmtime(DATA_PATH), unit='s')}")
    else:
        st.warning("No data loaded yet. Check the error message above.")

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

    if df.empty:
        st.warning("No data to visualize.")
    else:
        NUMERIC_FEATURES = [
            c for c in [
                "num_images", "num_paragraphs", "num_external_links", "num_internal_links",
                "num_headings", "num_bold", "num_lists", "num_blockquotes", "num_captions",
                "num_tags", "title_word_count", "title_char_count",
                "num_countries_mentioned", "num_ship_type_mentions", "num_weapon_system_mentions",
            ] if c in df.columns
        ]
        BINARY_FLAGS = [
            c for c in [
                "title_has_number", "is_weekend", "is_submarine_related",
                "is_surface_combatant_related", "is_uncrewed_related",
                "has_contract_mention", "has_exercise_mention",
            ] if c in df.columns
        ]

        tabs = st.tabs([
            "1. Word Count Distribution",
            "2. Correlation Heatmap",
            "3. Correlation Ranking",
            "4. Interactive Scatter",
            "5. Monthly Trend",
            "6. Ship Type Analysis",
            "7. Publication Patterns",
            "8. Content Mentions",
            "9. Top Authors",
            "10. Top Countries",
            "11. Feature Distributions",
            "12. Model Fit",
        ])

        # ---- 1. Word Count Distribution ----
        with tabs[0]:
            st.subheader("Distribution of Word Count")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(df["word_count"], kde=True, bins=30, color="steelblue", ax=ax)
            ax.set_title("Distribution of Article Word Count")
            ax.set_xlabel("Word Count")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        # ---- 2. Correlation Heatmap ----
        with tabs[1]:
            st.subheader("Correlation Heatmap")
            heatmap_cols = ["word_count"] + NUMERIC_FEATURES
            corr_matrix = df[heatmap_cols].corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True,
                        annot_kws={"size": 7}, ax=ax)
            ax.set_title("Correlation Heatmap of All Numeric Features")
            st.pyplot(fig)

        # ---- 3. Correlation Ranking ----
        with tabs[2]:
            st.subheader("Which Features Correlate Most with Word Count?")
            corr_with_y = df[["word_count"] + NUMERIC_FEATURES].corr()["word_count"].drop("word_count")
            corr_with_y = corr_with_y.sort_values()
            fig, ax = plt.subplots(figsize=(10, 7))
            colors = ["#d62728" if v < 0 else "#1f77b4" for v in corr_with_y.values]
            ax.barh(corr_with_y.index, corr_with_y.values, color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title("Correlation of Each Feature with Word Count")
            ax.set_xlabel("Correlation Coefficient")
            ax.set_ylabel("Feature")
            st.pyplot(fig)

        # ---- 4. Interactive Scatter ----
        with tabs[3]:
            st.subheader("Explore Any Feature Against Word Count")
            x_feature = st.selectbox("Choose a feature for the X axis", NUMERIC_FEATURES, index=0)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.regplot(
                data=df, x=x_feature, y="word_count",
                scatter_kws={"alpha": 0.4}, line_kws={"color": "red"}, ax=ax,
            )
            ax.set_title(f"{x_feature} vs Word Count")
            ax.set_xlabel(x_feature)
            ax.set_ylabel("Word Count")
            st.pyplot(fig)

        # ---- 5. Monthly Trend ----
        with tabs[4]:
            st.subheader("Average Word Count Over Time")
            if "publish_month" in df.columns:
                monthly_avg = df.groupby("publish_month")["word_count"].mean().sort_index()
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(monthly_avg.index, monthly_avg.values, marker="o", color="teal")
                ax.set_title("Average Word Count per Publication Month")
                ax.set_xlabel("Month")
                ax.set_ylabel("Average Word Count")
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig)
            else:
                st.info("`published_date` column was not found, so the monthly trend cannot be computed.")

        # ---- 6. Ship Type Analysis ----
        with tabs[5]:
            st.subheader("Word Count by Ship Type Discussed")
            ship_cols = {
                "is_submarine_related": "Submarine",
                "is_surface_combatant_related": "Surface Combatant",
                "is_uncrewed_related": "Uncrewed Vessel",
            }
            available_ship_cols = {k: v for k, v in ship_cols.items() if k in df.columns}
            if available_ship_cols:
                fig, axes = plt.subplots(1, len(available_ship_cols), figsize=(5 * len(available_ship_cols), 5), sharey=True)
                if len(available_ship_cols) == 1:
                    axes = [axes]
                for ax, (col, label) in zip(axes, available_ship_cols.items()):
                    plot_df = df[[col, "word_count"]].copy()
                    plot_df[col] = plot_df[col].map({1: "Yes", 0: "No"})
                    sns.boxplot(data=plot_df, x=col, y="word_count", hue=col, palette="Set2", legend=False, ax=ax)
                    ax.set_title(f"Discusses {label}?")
                    ax.set_xlabel("")
                    ax.set_ylabel("Word Count")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Ship type columns were not found in this dataset.")

        # ---- 7. Publication Patterns ----
        with tabs[6]:
            st.subheader("Word Count by Publication Timing")
            if "is_weekend" in df.columns:
                plot_df = df[["is_weekend", "word_count"]].copy()
                plot_df["is_weekend"] = plot_df["is_weekend"].map({1: "Weekend", 0: "Weekday"})
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(data=plot_df, x="is_weekend", y="word_count", hue="is_weekend",
                            palette="Set3", legend=False, ax=ax)
                ax.set_title("Word Count: Weekday vs Weekend Publication")
                ax.set_xlabel("")
                ax.set_ylabel("Word Count")
                st.pyplot(fig)
            else:
                st.info("`is_weekend` column was not found in this dataset.")

        # ---- 8. Content Mentions ----
        with tabs[7]:
            st.subheader("How Often Each Content Topic Appears")
            if BINARY_FLAGS:
                counts = df[BINARY_FLAGS].sum().sort_values()
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(counts.index, counts.values, color="darkorange")
                ax.set_title("Number of Articles Mentioning Each Topic")
                ax.set_xlabel("Number of Articles")
                ax.set_ylabel("Topic Flag")
                st.pyplot(fig)
            else:
                st.info("No binary content flags were found in this dataset.")

        # ---- 9. Top Authors ----
        with tabs[8]:
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

        # ---- 10. Top Countries ----
        with tabs[9]:
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

        # ---- 11. Feature Distributions ----
        with tabs[10]:
            st.subheader("Distribution of Individual Features")
            grid_features = [f for f in [
                "num_paragraphs", "num_images", "num_headings",
                "title_word_count", "num_ship_type_mentions", "num_weapon_system_mentions",
            ] if f in df.columns]
            if grid_features:
                n_cols = 3
                n_rows = int(np.ceil(len(grid_features) / n_cols))
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
                axes = np.array(axes).reshape(-1)
                for ax, feat in zip(axes, grid_features):
                    sns.histplot(df[feat], bins=20, color="slateblue", ax=ax)
                    ax.set_title(feat)
                    ax.set_xlabel("")
                for ax in axes[len(grid_features):]:
                    ax.axis("off")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No features available for distribution plots.")

        # ---- 12. Model Fit ----
        with tabs[11]:
            st.subheader("Actual vs Predicted Word Count")
            if model_multi is None:
                st.info("Load the prediction model first (see the Prediction page) to view model fit charts.")
            elif not all(f in df.columns for f in FEATURE_ORDER):
                missing = [f for f in FEATURE_ORDER if f not in df.columns]
                st.info(f"This dataset is missing columns required by the model: {', '.join(missing)}")
            else:
                X_all = df[FEATURE_ORDER]
                y_all = df["word_count"]
                y_pred_all = model_multi.predict(X_all)
                residuals = y_all - y_pred_all

                col_left, col_right = st.columns(2)
                with col_left:
                    fig, ax = plt.subplots(figsize=(7, 6))
                    ax.scatter(y_all, y_pred_all, alpha=0.4)
                    ax.plot([y_all.min(), y_all.max()], [y_all.min(), y_all.max()], "r--")
                    ax.set_title("Actual vs Predicted Word Count")
                    ax.set_xlabel("Actual Word Count")
                    ax.set_ylabel("Predicted Word Count")
                    st.pyplot(fig)
                with col_right:
                    fig, ax = plt.subplots(figsize=(7, 6))
                    ax.scatter(y_pred_all, residuals, alpha=0.4)
                    ax.axhline(0, color="red", linestyle="--")
                    ax.set_title("Residual Plot")
                    ax.set_xlabel("Predicted Word Count")
                    ax.set_ylabel("Residual")
                    st.pyplot(fig)

                r2 = 1 - (residuals**2).sum() / ((y_all - y_all.mean())**2).sum()
                st.caption(f"R-squared on the full dataset: {r2:.4f} (computed live from the loaded model, not a held-out test score)")

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