# Restaurant Rating Predictor

This project predicts restaurant ratings based on customer reviews.

## Dataset

The dataset used is the [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) from Kaggle.  

To use this project, download the CSV file from Kaggle and place it in the `data/` folder:

data/amazon_reviews.csv

## Project Structure

Restaurant-rating-predictor/
│
├── data/ # Folder containing the dataset CSV
│ └── amazon_reviews.csv
│
├── models/ # Folder containing trained model and vectorizer
│ ├── model.joblib
│ └── vectorizer.joblib
│
├── venv/ # Python virtual environment (ignored in git)
├── app.py # Streamlit app
├── notebook.py # Jupyter notebook (if used)
├── requirements.txt # Python dependencies
└── README.md # This file

bash
Copy code

## Usage

1. Clone the repository:

git clone https://github.com/ashureddy02/restaurant-rating-predictor.git
cd restaurant-rating-predictor
Install dependencies:


pip install -r requirements.txt
Run the Streamlit app:


streamlit run app.py
Open your browser to see the app running and input restaurant reviews to get predicted ratings.

Notes:
Make sure the CSV dataset (amazon_reviews.csv) is placed inside the data/ folder.

The venv/ folder is ignored in GitHub.
