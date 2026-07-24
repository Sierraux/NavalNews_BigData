# 🚢 Naval News Analytics Dashboard

### Predicting article length from structure and content, from web scraping to a deployed prediction app.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An interactive analytics dashboard that explores and predicts the length of naval defense news articles, based on how they are structured (images, paragraphs, headings, links) and what they are about (countries mentioned, ship types, weapon systems discussed). Built as the Final Project for **Big Data & Predictive Analytics**.

🔗 **Live Demo:** [navalnewsdashboardgit-drvojqsv6v9vadz3sugc3t.streamlit.app](https://navalnewsdashboardgit-drvojqsv6v9vadz3sugc3t.streamlit.app/)

---

## 📖 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Dataset and Methodology](#-dataset-and-methodology)
4. [Feature Engineering](#-feature-engineering)
5. [Modeling Approach](#-modeling-approach)
6. [Model Performance](#-model-performance)
7. [Technologies Used](#-technologies-used)
8. [Project Structure](#-project-structure)
9. [How to Run Locally](#-how-to-run-locally)
10. [Deployment Notes](#-deployment-notes)
11. [Known Limitations](#-known-limitations)
12. [Future Improvements](#-future-improvements)
13. [Author](#-author)
14. [Acknowledgments](#-acknowledgments)

---

## 🌊 Project Overview

Most article prediction projects rely on engagement metrics such as views, likes, or shares. Naval News does not publish any of these publicly, so this project asks a different, still valid regression question:

> **Can the length of a naval defense news article be predicted from how it is written, both in terms of its structure and its subject matter?**

To answer this, a dataset of naval defense articles was collected independently through web scraping (using `requests` and `BeautifulSoup`) from [navalnews.com](https://www.navalnews.com/category/naval-news/), a public news website. The raw HTML was parsed into a set of structural and content features, cleaned, analyzed with correlation and regression techniques, and finally packaged into this dashboard so anyone can explore the data and generate live predictions without touching a single line of code.

The full lifecycle of this project follows a standard data analytics pipeline:

```
Web Scraping -> Data Cleaning -> Exploratory Data Analysis -> Correlation Analysis
-> Regression Modeling -> Classical Assumption Testing -> Model Evaluation
-> Interactive Dashboard (this repository)
```

## ✨ Key Features

| Page | Description |
|---|---|
| 🏠 **Home** | An at a glance summary of the dataset: total articles, total unique authors, average word count, average paragraph count, and content diversity metrics such as the average number of countries mentioned per article. |
| 🗂️ **Dataset** | A fully interactive, searchable table of the cleaned dataset. Search articles by title and download the filtered result as a CSV file with a single click. |
| 📈 **Visualization** | Five visualization tabs covering the word count distribution, a scatter plot of paragraphs versus word count, a full correlation heatmap across all structural and content features, the top 10 most active authors, and the top 10 most frequently mentioned countries or navies. |
| 🤖 **Prediction** | A guided form with 20 structural and content inputs (with sensible defaults) that feeds a trained Multiple Linear Regression model and returns an estimated word count instantly. |
| 🔄 **Cache Refresh** | A one click "Refresh cached data" button in the sidebar, useful whenever the underlying dataset or model file is updated after deployment. |

## 🔬 Dataset and Methodology

**Data source:** public article pages under the Naval News category `naval-news`, collected across roughly 140 category pages.

**Scraping ethics and etiquette:**
- A realistic browser `User-Agent` header was used for every request.
- A `time.sleep()` delay was applied between requests to avoid overloading the server.
- Only publicly displayed elements were collected: title, publication date, author, body text, images, links, and tags.
- Progress was checkpointed to Google Drive every 100 articles, so long scraping runs could resume safely if interrupted.

**Data cleaning steps:**
1. Removed rows with missing values in key structural or content columns.
2. Removed duplicate articles based on both URL and title.
3. Removed very short entries (below 50 words), which usually indicate video or photo only posts rather than full articles.
4. Converted the publication date to a proper datetime format.
5. Removed statistical outliers using the IQR method, applied only to continuous variables (word count, paragraph count, image count, link counts) to avoid incorrectly discarding valid rows in sparse, zero heavy features such as blockquote or caption counts.
6. Verified that the final cleaned dataset contains at least 1,000 records, as required by the project guidelines.

**Dependent variable (Y):**
- `word_count`, the number of words in the article body.

## 🧩 Feature Engineering

A total of **20 independent variables** were engineered, split into two categories.

### Structural features (derived from HTML structure)

| Feature | Description |
|---|---|
| `num_images` | Number of images in the article |
| `num_paragraphs` | Number of text paragraphs |
| `num_external_links` | Number of outbound links to other websites |
| `num_internal_links` | Number of links to other Naval News articles |
| `num_headings` | Number of subheadings (h2/h3) |
| `num_bold` | Number of bold or strong text elements |
| `num_lists` | Number of bullet or numbered lists |
| `num_blockquotes` | Number of quoted passages |
| `num_captions` | Number of image captions |
| `title_word_count` | Number of words in the title |
| `title_has_number` | Whether the title contains a digit, such as a year or model number |
| `is_weekend` | Whether the article was published on a weekend |

### Content features (derived from keyword matching on the article text)

| Feature | Description |
|---|---|
| `num_countries_mentioned` | Number of distinct countries or navies mentioned |
| `num_ship_type_mentions` | Total mentions of ship types (frigate, destroyer, submarine, carrier, and so on) |
| `is_submarine_related` | Whether the article discusses submarines |
| `is_surface_combatant_related` | Whether the article discusses frigates, destroyers, corvettes, or carriers |
| `is_uncrewed_related` | Whether the article discusses drones or other uncrewed vessels |
| `num_weapon_system_mentions` | Total mentions of weapon systems (missile, torpedo, radar, sonar, and so on) |
| `has_contract_mention` | Whether the article mentions a contract, acquisition, or procurement |
| `has_exercise_mention` | Whether the article mentions a military exercise or drill |

> This keyword based approach is intentionally lightweight rather than a full NLP or named entity recognition pipeline. That keeps the methodology transparent and easy to explain, at the cost of missing entities phrased in ways not covered by the keyword dictionary. This tradeoff is documented as a known limitation below.

## 📐 Modeling Approach

1. **Correlation analysis** was performed first, to identify which structural and content features are most strongly associated with `word_count`.
2. A **Simple Linear Regression** model was built using the single feature with the highest correlation to word count.
3. A **Multiple Linear Regression** model was built using all 20 structural and content features together.
4. **Classical assumption tests** were applied to the multiple regression model:
   - **Multicollinearity**, checked with Variance Inflation Factor (VIF).
   - **Heteroscedasticity**, checked with the Breusch-Pagan test.
   - **Residual normality**, checked with the Shapiro-Wilk test, supported by a histogram and Q-Q plot.
5. Both models were evaluated on a held out test set using **R squared, RMSE, and MAE**.

## 📊 Model Performance

Exact performance figures (R squared, RMSE, and MAE for both the Simple and Multiple Linear Regression models) are exported automatically from the analysis notebook into `model_summary_v3.csv`, and discussed in detail in Chapter 4 of the accompanying project report. Please refer to that file or the report for the most current figures, since they are regenerated every time the dataset or feature set changes.

| Model | R squared (Test) | RMSE (Test) | MAE (Test) |
|---|---|---|---|
| Simple Linear Regression | see `model_summary_v3.csv` | see `model_summary_v3.csv` | see `model_summary_v3.csv` |
| Multiple Linear Regression | see `model_summary_v3.csv` | see `model_summary_v3.csv` | see `model_summary_v3.csv` |

## 🛠️ Technologies Used

- **Python 3** as the core language.
- **Streamlit** for the interactive dashboard framework.
- **Pandas** and **NumPy** for data manipulation.
- **Matplotlib** and **Seaborn** for visualization.
- **Scikit-learn** for building and evaluating the Multiple Linear Regression model.
- **Statsmodels** and **SciPy** for the classical assumption tests in the analysis notebook.
- **BeautifulSoup** and **Requests** for the original web scraping process.
- **Pickle** for serializing and loading the trained model.

## 📁 Project Structure

```
naval-news-dashboard/
├── app.py                       Main Streamlit application
├── naval_news_cleaned_v3.csv    Cleaned dataset with structural and content features
├── model_multi_v3.pkl           Trained Multiple Linear Regression model
├── style.css                    Custom dashboard styling
├── requirements.txt             Python dependencies
└── README.md                    Project documentation (this file)
```

> ⚠️ **Keep filenames in sync.** `app.py` loads the dataset and model by exact filename (`naval_news_cleaned_v3.csv` and `model_multi_v3.pkl`). If the dataset is refreshed or the model is retrained, either keep these exact filenames, or update the `DATA_PATH` and `MODEL_PATH` constants near the top of `app.py`. Also confirm that the prediction form's feature list still matches, in both name and order, the feature list the model was trained on.

## 🚀 How to Run Locally

1. Make sure Python 3.9 or newer is installed.
2. Clone this repository and move into the project folder:
   ```bash
   git clone <your-repo-url>
   cd naval-news-dashboard
   ```
3. (Recommended) create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
   On Windows, use `venv\Scripts\activate` instead.
4. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the application:
   ```bash
   streamlit run app.py
   ```
6. Open the local URL that Streamlit prints in the terminal, usually `http://localhost:8501`.

## ☁️ Deployment Notes

This dashboard is deployed on **Streamlit Community Cloud**, connected directly to this GitHub repository. A few practical notes for anyone maintaining it:

- Pushing a new commit normally triggers an automatic redeploy, but this can occasionally lag behind. If the live app appears to show outdated data or predictions after an update, use *Manage app* and choose *Reboot*, or click the in app "Refresh cached data" button in the sidebar.
- Streamlit's `@st.cache_data` decorator caches based on the function code, not on the contents of an external file. This app is configured to include each file's last modified timestamp as part of the cache key, so replacing `naval_news_cleaned_v3.csv` or `model_multi_v3.pkl` (even under the same filename) will automatically invalidate the old cache on the next run.
- Make sure `requirements.txt` stays up to date with every library imported in `app.py`, otherwise the Streamlit Cloud build will fail.

## ⚠️ Known Limitations

- Content features are extracted using keyword matching rather than a full NLP pipeline, so entities phrased outside the predefined keyword dictionary will not be detected.
- The multiple regression model may show heteroscedasticity or non normal residuals for some feature combinations. Results of the classical assumption tests are reported transparently in the analysis notebook and project report rather than hidden.
- The dataset reflects a snapshot in time of Naval News articles and will need to be refreshed periodically to stay current.

## 🔮 Future Improvements

- Add proper Named Entity Recognition to replace the current keyword based country and ship type detection.
- Experiment with regularized regression (Ridge or Lasso) or tree based models to compare against the linear baseline.
- Add scheduled re-scraping so the dashboard can refresh its own dataset automatically.
- Add downloadable PDF or image export of the generated charts directly from the dashboard.

## 👤 Author

**Daffa Mashabi Akmal Syarif**
Student ID: `24.61.0282`
Class: 24 BCI 01

## 🙏 Acknowledgments

- [Naval News](https://www.navalnews.com/) for publishing the publicly accessible articles used to build this dataset.
- The Big Data and Predictive Analytics course for the project brief and evaluation framework this dashboard was built to satisfy.

---

*This project demonstrates a complete data analytics workflow: ethical web scraping, feature engineering, exploratory analysis, regression modeling with classical assumption testing, and a deployed, interactive prediction dashboard.*
