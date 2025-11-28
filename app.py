import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Load model and data
@st.cache_resource
def load_model():
    return joblib.load('models/rf_model.joblib')

@st.cache_data
def load_data():
    X_test = pd.read_csv('models/X_test.csv')
    y_test = pd.read_csv('models/y_test.csv').values.ravel()
    y_pred = np.loadtxt('models/y_pred.csv', delimiter=',', dtype=int)
    y_proba = np.loadtxt('models/y_proba.csv', delimiter=',')
    return X_test, y_test, y_pred, y_proba

model = load_model()
X_test, y_test, y_pred, y_proba = load_data()

# Title
st.title("🎯 Customer Churn Prediction Dashboard")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Model Performance", "Make Predictions"])

# ===== PAGE 1: Model Performance =====
if page == "Model Performance":
    st.header("📊 Model Performance Metrics")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Display metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision", f"{precision:.2%}")
    col3.metric("Recall", f"{recall:.2%}")
    col4.metric("F1 Score", f"{f1:.2%}")
    col5.metric("ROC AUC", f"{roc_auc:.2%}")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: No Churn', 'Predicted: Churn'],
        y=['Actual: No Churn', 'Actual: Churn'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20}
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        st.subheader("Top 10 Feature Importances")
        feature_imp = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        fig = px.bar(feature_imp, x='importance', y='feature', orientation='h')
        st.plotly_chart(fig, use_container_width=True)

# ===== PAGE 2: Make Predictions =====
else:
    st.header("🔮 Make New Predictions")
    
    st.write("Enter customer details to predict churn probability:")
    
    # Create input form
    col1, col2 = st.columns(2)
    
    input_data = {}
    features = X_test.columns.tolist()
    
    # Dynamically create inputs based on your features
    for i, feature in enumerate(features):
        col = col1 if i % 2 == 0 else col2
        
        with col:
            # Get sample values from test set
            sample_val = X_test[feature].iloc[0]
            
            # Check if feature appears to be categorical (few unique values)
            if X_test[feature].nunique() < 10:
                unique_vals = sorted(X_test[feature].unique())
                input_data[feature] = st.selectbox(
                    feature, 
                    options=unique_vals,
                    index=0
                )
            else:
                min_val = float(X_test[feature].min())
                max_val = float(X_test[feature].max())
                mean_val = float(X_test[feature].mean())
                
                input_data[feature] = st.number_input(
                    feature,
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100
                )
    
    # Predict button
    if st.button("Predict Churn", type="primary"):
        # Create dataframe from input
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0, 1]
        
        # Display results
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("⚠️ **HIGH RISK**: Customer likely to churn")
            else:
                st.success("✅ **LOW RISK**: Customer likely to stay")
        
        with col2:
            st.metric("Churn Probability", f"{probability:.1%}")
        
        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Churn Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if probability > 0.5 else "green"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
