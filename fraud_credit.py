import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
 
import warnings
warnings.filterwarnings("ignore")
 
# Load Data
def load_data(filepath='creditcard.csv'):
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Fraudulent transactions: {df[df.Class == 1].shape[0]} / {df.shape[0]}")
    return df
 
# Preprocessing
def preprocess_data(df):
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test
 
# Build Pipeline
def get_pipeline():
    pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ])
    return pipe
 
# Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
    
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
plt.show()
    
    # Optional: Precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(recall, precision, marker='.')
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
plt.show()
 
# Main
def main():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
 
    pipeline = get_pipeline()
pipeline.fit(X_train, y_train)
 
    evaluate_model(pipeline, X_test, y_test)
 
    # Save model
    joblib.dump(pipeline, "fraud_model.pkl")
    print("Model saved as fraud_model.pkl")
 
if __name__ == "__main__":
    main()
