# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
df = pd.read_csv(r"C:\MCA\website_traffic_project_python\website_traffic_project_python\website_traffic_project_data.csv")

# Step 1: Data Cleaning and Preparation
# Handling missing values
df['totals_transactions'].fillna(0, inplace=True)
df['totals_totalTransactionRevenue'].fillna(0, inplace=True)
df['totals_bounces'].fillna(0, inplace=True)
df['totals_pageviews'].fillna(0, inplace=True)

# Convert 'visitStartTime' to readable date format
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

# Feature Engineering: Average time per pageview
df['avg_time_per_pageview'] = df['totals_timeOnSite'] / df['totals_pageviews']

# Step 2: Exploratory Data Analysis (EDA)
sns.set(style="whitegrid")

# 1. Traffic Source Analysis: Count of traffic sources
traffic_source_counts = df['trafficSource_source'].value_counts().head(10)

# Plotting top 10 traffic sources
plt.figure(figsize=(10, 6))
sns.barplot(x=traffic_source_counts.values, y=traffic_source_counts.index, palette="Blues_d")
plt.title('Top 10 Traffic Sources', fontsize=16)
plt.xlabel('Count of Visits')
plt.ylabel('Traffic Source')
plt.show()

# 2. Device Category Analysis
device_category_counts = df['device_deviceCategory'].value_counts()

# Plotting device category distribution
plt.figure(figsize=(8, 5))
device_category_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#99ff99', '#ffcc99'])
plt.title('Device Category Distribution')
plt.ylabel('')
plt.show()

# 3. Engagement Analysis: Distribution of pageviews and time on site
plt.figure(figsize=(12, 6))

# Pageviews
plt.subplot(1, 2, 1)
sns.histplot(df['totals_pageviews'], bins=30, color='blue', kde=True)
plt.title('Distribution of Pageviews', fontsize=14)
plt.xlabel('Pageviews')
plt.ylabel('Frequency')

# Time on site
plt.subplot(1, 2, 2)
sns.histplot(df['totals_timeOnSite'], bins=30, color='green', kde=True)
plt.title('Distribution of Time on Site', fontsize=14)
plt.xlabel('Time on Site (seconds)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Step 3: User Segmentation
df['high_engagement'] = (df['totals_pageviews'] >= df['totals_pageviews'].mean()) & (df['totals_timeOnSite'] >= df['totals_timeOnSite'].mean())

# Step 4: Predictive Modeling
# Target: Whether user completed a transaction (1 if transactions > 0, else 0)
df['completed_transaction'] = df['totals_transactions'].apply(lambda x: 1 if x > 0 else 0)

# Features for prediction
features = ['totals_pageviews', 'totals_timeOnSite', 'totals_bounces', 'device_deviceCategory']
df_model = pd.get_dummies(df[features], drop_first=True)

# Target variable
target = df['completed_transaction']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_model, target, test_size=0.3, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Initialize Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning using Grid Search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(rf_model, param_grid, cv=5)
grid_search.fit(X_resampled, y_resampled)
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Output the results
print(f"Improved Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

# Save the model
joblib.dump(best_model, 'rf_model.pkl')

