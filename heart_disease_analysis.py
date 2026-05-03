import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ==============================
# 1. LOAD DATA
# ==============================
df = pd.read_csv('heart-disease-dataset.csv')

X = df[['age', 'sex', 'chest_pain_type', 'resting_bp_s', 'cholesterol',
        'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate',
        'exercise_angina', 'oldpeak', 'st_slope']]

y = df['target']

# ==============================
# 2. VISUALIZATION
# ==============================
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

sns.histplot(data=df, x="age", hue="target", bins=30, kde=True)
plt.title("Age Distribution vs Disease")
plt.show()

# ==============================
# 3. TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# 4. SCALING
# ==============================
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# ==============================
# 5. MODEL COMPARISON
# ==============================
print("\nMODEL COMPARISON")
print("="*50)

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_std, y_train)
log_pred = log_model.predict(X_test_std)

print("\nLogistic Regression Accuracy:", accuracy_score(y_test, log_pred))

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_std, y_train)
rf_pred = rf_model.predict(X_test_std)

print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1 Score:", f1_score(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

# ==============================
# 6. FEATURE IMPORTANCE
# ==============================
importance = rf_model.feature_importances_
features = X.columns

imp_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print("\nFEATURE IMPORTANCE:")
print(imp_df)

plt.figure(figsize=(8,5))
sns.barplot(x=imp_df['Importance'], y=imp_df['Feature'])
plt.title("Feature Importance (Random Forest)")
plt.show()

# ==============================
# 7. K-MEANS CLUSTERING
# ==============================
print("\nCLUSTERING ANALYSIS")
print("="*50)

sil_scores = []

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_train_std)
    score = silhouette_score(X_train_std, labels)
    sil_scores.append(score)
    print(f"K={k}, Silhouette Score={score:.4f}")

optimal_k = sil_scores.index(max(sil_scores)) + 2
print(f"\nOptimal K: {optimal_k}")

# Final model
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
train_labels = kmeans_final.fit_predict(X_train_std)

# ==============================
# 8. PCA VISUALIZATION
# ==============================
pca = PCA(n_components=2)
train_2d = pca.fit_transform(X_train_std)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.scatterplot(x=train_2d[:,0], y=train_2d[:,1], hue=train_labels)
plt.title("Clusters")

plt.subplot(1,2,2)
sns.scatterplot(x=train_2d[:,0], y=train_2d[:,1], hue=y_train)
plt.title("Actual Disease")

plt.show()

# ==============================
# 9. CLUSTER ANALYSIS
# ==============================
print("\nCLUSTER ANALYSIS")
print("="*50)

for cluster_id in range(optimal_k):

    cluster_patients = y_train[train_labels == cluster_id]

    total = len(cluster_patients)
    disease_count = (cluster_patients == 1).sum()

    disease_rate = (disease_count / total) * 100

    print(f"\nCluster {cluster_id}:")
    print(f"Total Patients: {total}")
    print(f"Disease Cases: {disease_count}")
    print(f"Disease Rate: {disease_rate:.2f}%")

    if disease_rate > 60:
        print("→ HIGH RISK GROUP")
    elif disease_rate > 40:
        print("→ MODERATE RISK GROUP")
    else:
        print("→ LOW RISK GROUP")

# ==============================
# 10. USER INPUT PREDICTION
# ==============================
print("\n===== PATIENT PREDICTION SYSTEM =====")

age = float(input("Enter Age: "))
sex = int(input("Sex (0=Female, 1=Male): "))
cp = int(input("Chest Pain Type (0-3): "))
bp = float(input("Resting BP: "))
chol = float(input("Cholesterol: "))
fbs = int(input("Fasting Blood Sugar (0/1): "))
ecg = int(input("Resting ECG (0-2): "))
hr = float(input("Max Heart Rate: "))
angina = int(input("Exercise Angina (0/1): "))
oldpeak = float(input("Oldpeak: "))
slope = int(input("ST Slope (0-2): "))

new_data = pd.DataFrame([[
    age, sex, cp, bp, chol, fbs, ecg, hr, angina, oldpeak, slope
]], columns=X.columns)

new_scaled = scaler.transform(new_data)

prediction = rf_model.predict(new_scaled)
probability = rf_model.predict_proba(new_scaled)[0][1]
cluster = kmeans_final.predict(new_scaled)

print("\nPrediction Result:")
if prediction[0] == 1:
    print("High chance of Heart Disease")
else:
    print("Low chance of Heart Disease")

print(f"Probability: {probability:.2f}")
print(f"Assigned Cluster: {cluster[0]}")
