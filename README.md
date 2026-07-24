# 🚢 Naval News Analytics Dashboard 📊

**Predicting article length from structure *and* content — a full data analytics workflow from web scraping to deployed prediction.**

An interactive dashboard for exploring and predicting the length of naval defense news articles based on their structural characteristics (images, paragraphs, headings, links...) and their content characteristics (countries mentioned, ship types, weapon systems discussed...). Built as the Final Project for **Big Data & Predictive Analytics**.

🔗 **Live demo:** [navalnewsdashboardgit-drvojqsv6v9vadz3sugc3t.streamlit.app](https://navalnewsdashboardgit-drvojqsv6v9vadz3sugc3t.streamlit.app/)

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Dataset & Methodology](#-dataset--methodology)
- [Model Performance](#-model-performance)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [How to Run Locally](#-how-to-run-locally)
- [Author](#-author)

---

## 🌊 Project Overview

Naval News does not publish public engagement metrics (views, likes) for its articles, so instead of predicting popularity, this project asks a different question:

> **Can the length of a naval defense news article be predicted from how it is written — its structure and its subject matter?**

The dataset was collected independently through web scraping (BeautifulSoup + Requests) from [navalnews.com](https://www.navalnews.com/category/naval-news/), a public news website, followed by a full cleaning and feature-engineering pipeline. The dashboard below is the final, user-facing layer of that pipeline: it lets anyone explore the data, inspect the correlations behind the model, and generate live predictions.

## ✨ Key Features

| Page | What it does |
|---|---|
| 🏠 **Home** | At-a-glance dataset summary — total articles, authors, average word count, and content-diversity metrics. |
| 🗂️ **Dataset** | Full interactive table of the cleaned dataset, with title search and one-click CSV download. |
| 📈 **Visualization** | Word count distribution, scatter plots, a correlation heatmap across all structural & content features, top authors, and top mentioned countries. |
| 🤖 **Prediction** | Estimate an article's word count from 20 structural and content inputs, powered by a trained Multiple Linear Regression model. |

## 🔬 Dataset & Methodology

- **Source:** Web scraping of public article pages on Naval News, respecting robots/etiquette rules (custom User-Agent, request throttling with `time.sleep()`, checkpointed scraping).
- **Sample size:** 1,000+ cleaned articles (outliers removed via the IQR method on continuous variables only).
- **Dependent variable (Y):** `word_count` — the number of words in the article body.
- **Independent variables (X) — 20 total, in two groups:**
  - **Structural features:** `num_images`, `num_paragraphs`, `num_external_links`, `num_internal_links`, `num_headings`, `num_bold`, `num_lists`, `num_blockquotes`, `num_captions`, `title_word_count`, `title_has_number`, `is_weekend`.
  - **Content features** (extracted via keyword matching against the article text): `num_countries_mentioned`, `num_ship_type_mentions`, `is_submarine_related`, `is_surface_combatant_related`, `is_uncrewed_related`, `num_weapon_system_mentions`, `has_contract_mention`, `has_exercise_mention`.
- **Analysis pipeline:** correlation analysis → Simple Linear Regression → Multiple Linear Regression → classical assumption tests (Multicollinearity/VIF, Heteroscedasticity/Breusch–Pagan, Residual Normality/Shapiro–Wilk) → model evaluation (R², RMSE, MAE).

The full scraping and analysis code is available in the accompanying Google Colab notebook (see `Appendices` in the project report).

## 📊 Model Performance

Exact metrics (R², RMSE, MAE for both the Simple and Multiple Linear Regression models) are exported from the analysis notebook to `model_summary_v3.csv` and summarized in the project report, Chapter 4. Refer to that file/report for the latest figures, since they are regenerated whenever the underlying dataset or feature set changes.

## 🛠️ Technologies Used

- **Python 3**
- **Streamlit** — dashboard framework
- **Pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — visualization
- **Scikit-learn** — Multiple Linear Regression model
- **Pickle** — model serialization/loading

## 📁 Project Structure

```
naval-news-dashboard/
├── app.py                       # Main Streamlit application
├── naval_news_cleaned_v3.csv    # Cleaned dataset (structural + content features)
├── model_multi_v3.pkl           # Trained Multiple Linear Regression model
├── style.css                    # Custom dashboard styling
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation (this file)
```

> ⚠️ **Keep filenames in sync:** `app.py` loads the dataset/model by exact filename (`naval_news_cleaned_v3.csv`, `model_multi_v3.pkl`). If you retrain the model or refresh the dataset, either keep these exact filenames or update the `DATA_PATH` / `MODEL_PATH` constants at the top of `app.py` accordingly — and make sure the prediction form's feature list still matches the order the model was trained on.

## 🚀 How to Run Locally

1. Make sure Python 3.9+ is installed on your system.
2. Clone this repository and move into the project folder:
   ```bash
   git clone <your-repo-url>
   cd naval-news-dashboard
   ```
3. (Recommended) create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
4. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the application:
   ```bash
   streamlit run app.py
   ```
6. Open the local URL Streamlit prints in your terminal (usually `http://localhost:8501`).

If you're deploying to **Streamlit Community Cloud** and updating the dataset/model after a previous deployment, use the app's own **"🔄 Refresh cached data"** button in the sidebar, or reboot the app from *Manage app* — Streamlit's cache does not automatically detect that an uploaded file's contents changed if the filename and code stay the same.

## 👤 Author

**Daffa Mashabi Akmal Syarif**
Student ID: `24.61.0282`
Class: 24 BCI 01

---

*This project demonstrates a complete data analytics workflow — from ethical web scraping and feature engineering, through exploratory analysis and regression modeling, to an interactive, deployed prediction dashboard.*