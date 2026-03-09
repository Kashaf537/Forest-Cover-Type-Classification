# ForestCoverApp.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import xgboost as xgb
import warnings
import json

warnings.filterwarnings("ignore")

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

st.set_page_config(page_title="Forest Cover Type Classification", layout="wide")
st.title("🌲 Forest Cover Type Classification System")
st.write("""
Predict forest cover type using environmental and cartographic features.
Compare **Random Forest**, **XGBoost**, and **Logistic Regression** models.
""")

# ---- Load Dataset ----
@st.cache_data
def load_data(path='D:/Arch/forest_dataset.csv'):
    # Read the CSV file
    df = pd.read_csv(path)
    
    # The last column (index 54) is the target variable
    # Let's name it 'Cover_Type' for clarity
    df.columns = [f'Feature_{i}' for i in range(df.shape[1]-1)] + ['Cover_Type']
    
    # Separate features and target
    X = df.drop('Cover_Type', axis=1)
    y = df['Cover_Type']
    
    # Map labels 1-7 -> 0-6 (if needed, depends on your data)
    # Check if mapping is needed
    unique_labels = np.sort(y.unique())
    if set(unique_labels) != set(range(len(unique_labels))):
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        y = y.map(label_mapping)
    else:
        label_mapping = {i: i for i in unique_labels}
    
    # Identify continuous columns (first 10 features based on your data)
    continuous_cols = ['Feature_0', 'Feature_1', 'Feature_2', 'Feature_3', 'Feature_4',
                       'Feature_5', 'Feature_6', 'Feature_7', 'Feature_8', 'Feature_9']
    
    # Standardize continuous columns
    scaler = StandardScaler()
    X[continuous_cols] = scaler.fit_transform(X[continuous_cols])
    
    return X, y, label_mapping

# Load the data
try:
    X, y, label_mapping = load_data()
    st.success(f"✅ Data loaded successfully! Shape: {X.shape}")
    
    # Convert to Python native types for display
    label_mapping_display = {int(k): int(v) for k, v in label_mapping.items()}
    class_distribution = {int(k): int(v) for k, v in y.value_counts().to_dict().items()}
    
    st.write("Target classes:", label_mapping_display)
    st.write("Class distribution:", class_distribution)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ---- Split Dataset ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Sidebar: Model Selection & Hyperparameters ----
st.sidebar.header("⚙️ Model Configuration")

# Random Forest
use_rf = st.sidebar.checkbox("Random Forest", True, key="use_rf")
if use_rf:
    st.sidebar.subheader("Random Forest Parameters")
    rf_n_estimators = st.sidebar.slider("RF: n_estimators", 50, 300, 100, step=10, key="rf_n_estimators")
    rf_max_depth = st.sidebar.slider("RF: max_depth", 5, 50, 20, step=1, key="rf_max_depth")
    rf_min_samples_split = st.sidebar.slider("RF: min_samples_split", 2, 10, 2, step=1, key="rf_min_samples_split")

# XGBoost
use_xgb = st.sidebar.checkbox("XGBoost", True, key="use_xgb")
if use_xgb:
    st.sidebar.subheader("XGBoost Parameters")
    xgb_n_estimators = st.sidebar.slider("XGB: n_estimators", 50, 300, 100, step=10, key="xgb_n_estimators")
    xgb_max_depth = st.sidebar.slider("XGB: max_depth", 3, 20, 6, step=1, key="xgb_max_depth")
    xgb_learning_rate = st.sidebar.slider("XGB: learning_rate", 0.01, 0.3, 0.1, step=0.01, key="xgb_learning_rate")
    xgb_subsample = st.sidebar.slider("XGB: subsample", 0.5, 1.0, 1.0, step=0.05, key="xgb_subsample")

# Logistic Regression
use_lr = st.sidebar.checkbox("Logistic Regression", True, key="use_lr")

# ---- Evaluation Function ----
def evaluate_model(y_true, y_pred, name):
    st.subheader(f"📊 {name} Evaluation")
    
    # Convert numpy types to Python native for display
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average='weighted'))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{acc:.4f}")
    with col2:
        st.metric("F1-score", f"{f1:.4f}")
    
    st.text("Classification Report:")
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Format report as dataframe for better display
    report_df = pd.DataFrame(report).T
    report_df = report_df.round(4)
    st.dataframe(report_df)
    
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', ax=ax)
    ax.set_title(f'{name} Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    plt.close()
    
    return acc, f1

# ---- Train & Evaluate ----
results = []

if use_rf:
    with st.spinner("Training Random Forest..."):
        rf = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_split=rf_min_samples_split,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_acc, rf_f1 = evaluate_model(y_test, rf_preds, "Random Forest")
        
        # Feature Importance
        fi_df = pd.DataFrame({
            'Feature': X.columns, 
            'Importance': rf.feature_importances_
        })
        fi_df = fi_df.sort_values(by='Importance', ascending=False)
        
        st.subheader("🔝 Top 20 Feature Importances (RF)")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(x='Importance', y='Feature', data=fi_df.head(20), ax=ax)
        ax.set_title("Random Forest Feature Importance")
        st.pyplot(fig)
        plt.close()
        
        results.append({
            'Model':'Random Forest', 
            'Accuracy': rf_acc,
            'F1-score': rf_f1
        })

if use_xgb:
    with st.spinner("Training XGBoost..."):
        xgb_model = xgb.XGBClassifier(
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            subsample=xgb_subsample,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)
        xgb_acc, xgb_f1 = evaluate_model(y_test, xgb_preds, "XGBoost")
        
        results.append({
            'Model':'XGBoost', 
            'Accuracy': xgb_acc,
            'F1-score': xgb_f1
        })

if use_lr:
    with st.spinner("Training Logistic Regression..."):
        lr = LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42)
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_test)
        lr_acc, lr_f1 = evaluate_model(y_test, lr_preds, "Logistic Regression")
        
        results.append({
            'Model':'Logistic Regression', 
            'Accuracy': lr_acc,
            'F1-score': lr_f1
        })

# ---- Model Comparison ----
if results:
    st.subheader("📈 Model Comparison")
    res_df = pd.DataFrame(results)
    res_df = res_df.round(4)
    
    # Display comparison table with highlighting
    st.dataframe(
        res_df.style.highlight_max(subset=['Accuracy', 'F1-score'], color='lightgreen')
        .highlight_min(subset=['Accuracy', 'F1-score'], color='lightcoral')
    )
    
    # Bar plot comparison
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(res_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, res_df['Accuracy'], width, label='Accuracy', color='skyblue')
    bars2 = ax.bar(x + width/2, res_df['F1-score'], width, label='F1-score', color='lightcoral')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(res_df['Model'])
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    st.pyplot(fig)
    plt.close()

# ---- Additional Dataset Information ----
with st.expander("📋 Dataset Information"):
    st.write("### Feature Statistics")
    st.dataframe(X.describe().round(4))
    
    st.write("### Target Distribution")
    fig, ax = plt.subplots(figsize=(10,6))
    value_counts = y.value_counts().sort_index()
    bars = ax.bar(value_counts.index.astype(str), value_counts.values, color='skyblue')
    ax.set_xlabel('Cover Type')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Forest Cover Types')
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}', ha='center', va='bottom')
    
    st.pyplot(fig)
    plt.close()

# ---- Footer ----
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Scikit-learn, and XGBoost")