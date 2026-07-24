# Naval News Analytics Dashboard 📊🚢

A modern and interactive dashboard for analyzing and predicting article length based on the structural characteristics of Naval News articles. This project was developed as a Final Project for Big Data & Predictive Analytics.

## Project Overview
This dashboard combines data analysis, visualization, and predictive modeling in one simple interface. It helps users explore how features such as the number of images, paragraphs, external links, and title length relate to the total word count of an article.

## Key Features
- **Data Summary:** View essential statistics from the collected article dataset.
- **Interactive Dataset Explorer:** Browse articles, search by title, and download the filtered dataset as CSV.
- **Data Visualization:** Explore insights through histograms, scatter plots, and correlation heatmaps.
- **Prediction Module:** Estimate article word count using a Multiple Linear Regression model.

## Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Pickle for model loading

## How to Run Locally
1. Make sure Python is installed on your system.
2. Open your terminal or command prompt.
3. Install the required libraries:
   `pip install -r requirements.txt`
4. Run the application:
   `streamlit run app.py`

## Project Structure
- `app.py` — Main Streamlit dashboard application
- `naval_news_cleaned.csv` — Cleaned dataset used by the app
- `style.css` — Custom styling for the dashboard
- `requirements.txt` — Python dependencies




## Author
**Daffa Mashabi Akmal Syarif**  
**Student ID:** 24.61.0282

---
This project is designed to showcase a complete data analytics workflow, from data cleaning and exploration to visualization and predictive modeling.