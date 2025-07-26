import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    f1_score
)
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Page config
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection App")
st.markdown("Upload your transaction data and detect frauds using a Neural Network model.")

# Load model
@st.cache_resource
def load_trained_model():
    return load_model("model/CreditCardFraudDetection.h5")

model = load_trained_model()

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file (same format as training data):", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Data Preview")
    st.write(df.head())

    # Drop Time column if present
    df.drop(['Time'], axis=1, inplace=True, errors='ignore')

    # Store true labels if available
    y_true = df['Class'] if 'Class' in df.columns else None

    # Scale 'Amount' column
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])

    # Drop 'Class' before prediction
    X = df.drop(['Class'], axis=1, errors='ignore')

    # Predict probabilities
    y_probs = model.predict(X).flatten()

    st.subheader("⚙️ Threshold Tuning")
    threshold = st.slider("Set threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    # Predict classes
    y_pred = (y_probs > threshold).astype(int)

    # Show predictions
    df['Fraud Prediction'] = y_pred
    st.subheader("🔍 Prediction Results")
    st.write(df[['Amount', 'Fraud Prediction']].head(10))

    # If ground truth is present
    if y_true is not None:
        st.subheader("📈 Metrics")
        cm = confusion_matrix(y_true, y_pred)
        st.text("Classification Report:")
        st.text(classification_report(y_true, y_pred, digits=4))

        # Confusion Matrix Heatmap
        st.write("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['Legit', 'Fraud'],
                    yticklabels=['Legit', 'Fraud'])
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

        # Precision-Recall Curve
        prec, rec, thresh_pr = precision_recall_curve(y_true, y_probs)
        fig_pr, ax_pr = plt.subplots()
        ax_pr.plot(rec, prec, linewidth=2)
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Precision-Recall Curve")
        ax_pr.grid(True)
        st.write("Precision–Recall Curve")
        st.pyplot(fig_pr)

        # F1 vs Threshold
        f1_scores = [f1_score(y_true, (y_probs > t).astype(int)) for t in thresh_pr[:-1]]
        fig_f1, ax_f1 = plt.subplots()
        ax_f1.plot(thresh_pr[:-1], f1_scores, color='purple')
        ax_f1.set_xlabel("Threshold")
        ax_f1.set_ylabel("F1 Score")
        ax_f1.set_title("F1 Score vs Threshold")
        ax_f1.grid(True)
        st.write("F1 Score vs Threshold")
        st.pyplot(fig_f1)


