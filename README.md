🚢 Titanic Survival Prediction – KNN Classifier

This project is part of my 30-Day Machine Learning Algorithms Challenge (Day 2).
The goal is to predict whether a passenger survived the Titanic disaster using the K-Nearest Neighbors (KNN) algorithm.

🔗 Live App:
https://titanicsurvivalpredicationknn-trryik483ev48dwvwzhdr8.streamlit.app/

📌 Project Overview

The Titanic dataset is a binary classification problem where we predict if a passenger:

0 → Did not survive

1 → Survived

We use KNN Classifier along with preprocessing techniques and interactive predictions using Streamlit.

✨ Features

✔️ Upload Titanic dataset (train.csv)

✔️ Automated data preprocessing (missing values, encoding, scaling)

✔️ KNN model training & evaluation

✔️ Metrics: Accuracy, Confusion Matrix, Classification Report

✔️ Interactive passenger survival prediction

✔️ Clear EDA visualizations for better understanding

✔️ Detailed explanation in documentation.txt

📂 Project Structure

titanic_survival_predication-knn

│── data/train.CSV/n
│── app.py                    
│── documentation.txt         
│── README.md                 
│── requirements.txt          

🧠 About KNN Algorithm

KNN (K-Nearest Neighbors) is a supervised ML algorithm that classifies data points based on their k nearest neighbors.

It uses distance metrics (e.g., Euclidean) to find similar data points.

In Titanic’s case, passengers with similar age, fare, class, and family size influence survival predictions.

📌 Why KNN here?
Because Titanic survival depends on groups of similar passengers (families, class, gender), making KNN a great choice.

📊 Exploratory Data Analysis

The app provides:

Survival distribution bar chart

Heatmap of feature correlations

Confusion Matrix Heatmap

Classification report with precision, recall, F1-score

🚀 How to Run Locally

1️⃣ Clone this repository:

git clone https://github.com/babneek/titanic_survival_predication_KNN.git

cd titanic_survival_predication-knn


2️⃣ Create and activate a virtual environment:

python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Mac/Linux


3️⃣ Install dependencies:

pip install -r requirements.txt


4️⃣ Run the Streamlit app:

streamlit run app.py

📊 Results

Accuracy Score displayed after training

Confusion Matrix Heatmap for better understanding

Classification Report Table for survival & non-survival

Interactive predictions

📘 Documentation

For in-depth explanations of:

Dataset preprocessing

KNN theory and working

Evaluation metrics (Accuracy, Precision, Recall, F1-score)

Streamlit implementation

👉 Check documentation.txt in this repository.

📢 Challenge Progress

This project is part of my 30-Day ML Algorithms Challenge:

Day 1: Logistic Regression

Day 2: KNN Classifier ✅

Day 3: Coming soon...

Stay tuned for upcoming algorithms (Decision Trees, Random Forests, SVM, Neural Networks, etc.) 🚀

📬 Contact

👩‍💻 Author: Babneek Kaur
📧 Email: babneeksaini@gmail.com

📱 Phone: +91 8076893417
🌐 GitHub: https://github.com/babneek
