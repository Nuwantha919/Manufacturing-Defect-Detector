# train_logistic_regression.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# ------------------- 1. Load CSV -------------------
csv_path = r"D:\manu project\new\final_data_filewith_ratios.csv"
df = pd.read_csv(csv_path)

# Use r_ratio as feature and defect_status as label
X = df[["r_ratio"]].values
y = df["defect_status"].values

# ------------------- 2. Split dataset -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------- 3. Scale features -------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------- 4. Train Logistic Regression -------------------
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# ------------------- 5. Evaluate -------------------
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ------------------- 6. Save model and scaler -------------------
output_model_path = r"D:\manu project\new\logistic_model.pkl"
with open(output_model_path, "wb") as f:
    pickle.dump({"model": model, "scaler": scaler}, f)

print(f"\n✅ Model saved to {output_model_path}")
