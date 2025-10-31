Food Waste Reduction using Machine Learning
Project Overview

This project focuses on predicting optimal food quantities for urban events to minimize food waste.
Using a dataset of catering events, it applies machine learning (XGBoost) and clustering (KMeans) to extract insights for efficient and sustainable food management.

Objectives

Predict optimal food quantity to minimize wastage.

Explain model predictions using SHAP for interpretability.

Identify behavioral clusters contributing to food waste.

Analyze how storage conditions affect food spoilage.

Technologies Used

Programming Language: Python 3.10+

Libraries:

pandas

numpy

scikit-learn

xgboost

shap

matplotlib

seaborn

Project Structure
FoodWasteReduction/
│
├── food_wastage_data.csv      # Dataset
├── waste.py                   # Main Python script
├── requirements.txt           # Dependencies (recommended)
└── README.md                  # Project documentation

Installation and Setup
Step 1: Clone the Repository
git clone https://github.com/Rithin447/FoodWasteReduction.git
cd FoodWasteReduction

Step 2: Create a Virtual Environment

Windows (PowerShell):

python -m venv .venv
.venv\Scripts\Activate.ps1


macOS/Linux:

python3 -m venv .venv
source .venv/bin/activate

Step 3: Install Dependencies
pip install -r requirements.txt


Or manually:

pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn

Running the Project

Ensure food_wastage_data.csv is located in the same directory as waste.py.

Run the script:

python waste.py


The program will:

Train an XGBoost regression model

Output performance metrics (MAE, RMSE, R²)

Generate SHAP plots for feature importance

Perform clustering using KMeans

Display storage condition impact

Print actionable insights for waste reduction

Output

Performance Metrics

XGBoost Performance for Optimal Quantity Prediction:
MAE: 4.35, RMSE: 7.89, R²: 0.83


Generated Visualizations

SHAP summary plot of feature importance

Cluster visualization by event type

Bar chart showing impact of storage conditions

Scatter plot of actual vs. predicted quantities

Insights

Average waste reduction potential

Key drivers influencing waste

Behavioral clusters indicating patterns

Recommendations for improved storage practices

Future Improvements

Implement cross-validation for improved reliability

Replace label encoding with OneHotEncoding for categorical variables

Develop an interactive dashboard for real-time analysis

Extend analysis to non-urban environments

Author

Rithin Srinivas Arakatala
Master’s in Information Systems Technology (Business Intelligence & Analytics)
California State University, San Bernardino
GitHub: Rithin447

License

This project is open-source under the MIT License.
