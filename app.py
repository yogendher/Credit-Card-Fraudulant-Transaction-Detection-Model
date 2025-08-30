import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import streamlit as st

# =========================
# Load data
# =========================
data = pd.read_csv('creditcard.csv')

# separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# undersample legitimate transactions to balance classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# split features and target
X = data.drop(columns="Class", axis=1)
y = data["Class"]

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=2
)

# train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# evaluate model
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# =========================
# Streamlit app
# =========================
st.title("Credit Card Fraud Detection")
st.write("Enter values for all features to predict if the transaction is legitimate or fraudulent:")

# Create input fields dynamically
input_features = []
for i, col in enumerate(X.columns):
    value = st.number_input(f"{col}", value=0.0, format="%.6f")
    input_features.append(value)

# Submit button
if st.button("Predict"):
    # convert inputs to numpy array
    features = np.array(input_features, dtype=np.float64).reshape(1, -1)
    
    # scale input
    features_scaled = scaler.transform(features)
    
    # prediction
    prediction = model.predict(features_scaled)
    
    if prediction[0] == 0:
        st.success("✅ Legitimate transaction")
    else:
        st.error("⚠️ Fraudulent transaction")

# display model accuracy
st.write(f"Model Training Accuracy: {train_acc*100:.2f}%")
st.write(f"Model Testing Accuracy: {test_acc*100:.2f}%")
