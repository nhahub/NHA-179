# 🚗 Used Car Price Prediction Project

## 📘 Overview
This project predicts the **price of used cars** based on multiple features such as brand, mileage, age, transmission, engine type, and condition.  
It follows a complete **machine learning pipeline**, including data cleaning, feature engineering, EDA, model training, and saving the best-performing model for deployment.

---

## 🧠 Project Workflow

### **Step 1: Data Understanding**
- Dataset contains **3,933 car listings** with attributes such as:
  - `brand`, `milage`, `fuel_type`, `transmission`, `engine`, `clean_title`, and `price`.
- Target variable: **price** (car selling price).

---

### **Step 2: Data Cleaning**
- Removed symbols and standardized units in `price` and `milage`.
- Unified text formats for `fuel_type`, `transmission`, and `clean_title`.
- Converted columns to correct data types (int, float, or category).
- Filled missing values:
  - `fuel_type` by brand mode → `'OTHER'`
  - `accident` → `'None reported'`
  - `clean_title` → `'Unknown'`
- Removed **67 cars** with model year < 2000.
- Detected and removed **outliers** in `price` and `milage` using the **IQR method** (≈250 rows removed).
- Final dataset shape: **~3630 rows × 18 columns**.

---

### **Step 3: Feature Engineering**
- Created new features to enhance prediction:
  - `Vehicle_Age = 2025 - model_year`
  - `Mileage_per_Year = milage / Vehicle_Age`
- Added binning features:
  - `Age_Mid`, `Age_Old`, `Age_Very Old`
  - `Milage_Medium`, `Milage_High`, `Milage_Very High`
- Added binary features:
  - `Accident_Impact` (1 = accident, 0 = none)
  - `clean_title` (1 = clean, 0 = not clean)
  - `is_v_engine` (1 = V-type, 0 = others)
- Encoded categorical features and dropped redundant columns.

---

### **Step 4: Exploratory Data Analysis (EDA)**
- Visualized feature distributions and relationships with `price`.
- **Correlation Analysis**:
  - `Vehicle_Age` and `milage` → negatively correlated with price.
  - `engine` and `transmission` → strong indicators of price.
- **Categorical Insights**:
  - Gasoline and Automatic cars dominate the dataset.
  - Cars with `clean_title`, `no accidents`, and `V-type engines` tend to be higher priced.
- **Brand Analysis**:
  - Luxury brands (Lexus, BMW, Mercedes) have higher average prices.

---

### **Step 5: Machine Learning Models**
Models used for comparison:
- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **Random Forest Regressor**
- **XGBoost Regressor**

#### **Evaluation Metrics**
- **R² Score**
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**

✅ **Best Model:** XGBoost Regressor  
- Achieved **R² = 0.869**
- Strong prediction performance with close alignment between actual and predicted values.

---

### **Step 6: Model Saving**
The best model was saved as a `.pkl` file for reuse:

```python
import joblib
joblib.dump(best_model, 'car_price_model.pkl')
