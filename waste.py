
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and Preprocess Data
data = pd.read_csv('food_wastage_data.csv')

# Filtering for Urban locations 
data = data[data['Geographical Location'] == 'Urban']

# Derive new target
data['Optimal Quantity'] = data['Quantity of Food'] - data['Wastage Food Amount']

# Features and target
features = ['Type of Food', 'Number of Guests', 'Event Type', 'Storage Conditions', 
            'Purchase History', 'Seasonality', 'Preparation Method', 'Pricing']
target = 'Optimal Quantity'

# Encode categorical variables
le_dict = {}
for col in features:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        le_dict[col] = le

# Splitting features and target
X = data[features]
y = data[target]

scaler = StandardScaler()
X[['Number of Guests']] = scaler.fit_transform(X[['Number of Guests']])

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2 XGBoost Model for Demand Forecasting
# Predicting Optimal Quantity to minimize waste
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, 
                             learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print("XGBoost Performance for Optimal Quantity Prediction:")
print(f"MAE: {mae_xgb:.2f}, RMSE: {rmse_xgb:.2f}, R²: {r2_xgb:.2f}")

# SHAP for Explainability
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
plt.title('SHAP Feature Importance for Food Waste Reduction')
plt.tight_layout()
plt.show()

# Step 3: Behavioral Insights via Clustering (RQ2)
# Cluster events to identify waste patterns 
cluster_features = ['Number of Guests', 'Event Type', 'Preparation Method', 'Wastage Food Amount']
X_cluster = data[cluster_features]
X_cluster[['Number of Guests', 'Wastage Food Amount']] = scaler.fit_transform(X_cluster[['Number of Guests', 'Wastage Food Amount']])

kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_cluster)

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Number of Guests', y='Wastage Food Amount', hue='Cluster', style='Event Type')
plt.title('Behavioral Clusters of Food Waste by Event Type')
plt.show()

# Cluster summary
cluster_summary = data.groupby('Cluster').agg({
    'Wastage Food Amount': 'mean',
    'Number of Guests': 'mean',
    'Event Type': lambda x: le_dict['Event Type'].inverse_transform(x.mode())[0],
    'Preparation Method': lambda x: le_dict['Preparation Method'].inverse_transform(x.mode())[0]
})
print("\nCluster Summary (Behavioral Insights):")
print(cluster_summary)

# Step 4: Real-Time Monitoring Simulation (Research Q3)
# Analyze Storage Conditions' impact on waste
storage_analysis = data.groupby('Storage Conditions').agg({
    'Wastage Food Amount': 'mean',
    'Type of Food': lambda x: le_dict['Type of Food'].inverse_transform(x.mode())[0]
}).reset_index()
storage_analysis['Storage Conditions'] = le_dict['Storage Conditions'].inverse_transform(storage_analysis['Storage Conditions'])
print("\nStorage Conditions Impact on Waste (Monitoring Simulation):")
print(storage_analysis)

# Visualize storage impact
plt.figure(figsize=(8, 5))
sns.barplot(data=storage_analysis, x='Storage Conditions', y='Wastage Food Amount')
plt.title('Average Food Waste by Storage Conditions')
plt.show()

# Step 5: Actionable Insights
# Calculate waste reduction potential
data['Predicted Optimal Quantity'] = xgb_model.predict(data[features])
data['Potential Waste Reduction'] = data['Quantity of Food'] - data['Predicted Optimal Quantity']
mean_reduction = data['Potential Waste Reduction'].mean()

print("\nActionable Insights for Food Waste Reduction in Urban Event Catering:")
print(f"- Predicted optimal quantities could reduce waste by an average of {mean_reduction:.2f} units per event.")
print("- Key drivers (SHAP): Focus on adjusting quantities for high-impact features like Number of Guests and Event Type.")
print(f"- Behavioral clusters: High-waste clusters (e.g., buffets with many guests) suggest portion control interventions.")
print("- Monitoring: Prioritize refrigeration for perishable foods (e.g., Meat, Dairy) to reduce spoilage, as Room Temperature storage increases waste.")

# Optional Visualization: Actual vs Predicted Optimal Quantity
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Optimal Quantity')
plt.ylabel('Predicted Optimal Quantity')
plt.title('XGBoost: Actual vs Predicted Optimal Food Quantity')
plt.show()