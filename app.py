import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

st.set_page_config(page_title="Titanic Survival Prediction - KNN", layout="wide")

st.title("🚢 Titanic Survival Prediction (K-Nearest Neighbors - KNN)")

# --- Upload Dataset ---
st.header("1. Upload Titanic Dataset")
uploaded_file = st.file_uploader("Upload train.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Dataset")
    st.dataframe(df.head())

    # --- Basic EDA ---
    st.subheader("Exploratory Data Analysis")
    st.write("Survival distribution:")
    st.bar_chart(df['Survived'].value_counts())

    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Blues", ax=ax)
    st.pyplot(fig)

    # --- Preprocessing ---
    st.subheader("Data Preprocessing")

    # Drop irrelevant columns
    df = df.drop(columns=['Name', 'Ticket', 'Cabin'], errors='ignore')

    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Encode categorical
    le = LabelEncoder()
    for col in ['Sex', 'Embarked']:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    # Features & Target
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    
    # Drop PassengerId if exists
    if "PassengerId" in X.columns:
        X = X.drop("PassengerId", axis=1)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Model Training ---
    st.subheader("Model Training & Evaluation")

    k_value = st.slider("Choose value of k (number of neighbors)", 1, 20, 5)

    model = KNeighborsClassifier(n_neighbors=k_value)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.write(f"✅ Model Accuracy with k={k_value}: **{acc:.2f}**")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Died", "Survived"],
                yticklabels=["Died", "Survived"],
                ax=ax)
    st.pyplot(fig)

    # Classification report in nice table
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0))

    # Save model
    joblib.dump((model, scaler, le), "titanic_knn_model.pkl")
    st.success("Model trained & saved as titanic_knn_model.pkl")

    # --- Prediction ---
    st.subheader("🔮 Try Prediction with Custom Input")

    pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 80, 25)
    sibsp = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)
    parch = st.number_input("Parents/Children Aboard", 0, 6, 0)
    fare = st.number_input("Fare", 0.0, 600.0, 32.2)
    embarked = st.selectbox("Embarked", ["C", "Q", "S"])

    if st.button("Predict Survival"):
        # Encode categorical manually
        sex_val = 1 if sex == "male" else 0
        embarked_val = {"C": 0, "Q": 1, "S": 2}[embarked]

        input_data = np.array([[pclass, sex_val, age, sibsp, parch, fare, embarked_val]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        if prediction == 1:
            st.success("🎉 The passenger would have SURVIVED!")
        else:
            st.error("☠️ The passenger would NOT have survived.")
