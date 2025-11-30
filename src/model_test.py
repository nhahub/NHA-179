import pandas as pd
import pickle
import re
from sklearn.preprocessing import StandardScaler
import joblib

# ============================================================
# 1. FEATURE ENGINEERING FUNCTION
# ============================================================

def apply_feature_engineering(df):
    df_processed = df.copy()
    
    # === SAFE HP EXTRACTION ===
    # Extract HP if present, otherwise use median
    hp_extracted = df_processed['engine'].str.extract(r'(\d+)\s*HP', flags=re.IGNORECASE)[0]
    df_processed['hp'] = hp_extracted.astype(float)
    
    # Calculate median from non-null values only
    hp_median = df_processed['hp'].median() if not df_processed['hp'].isna().all() else 150  # fallback median
    
    df_processed['hp'] = df_processed['hp'].fillna(hp_median)
    df_processed['hp'] = df_processed['hp'].astype(int)

    # === ENGINE DISPLACEMENT ===
    displacement_extracted = df_processed['engine'].str.extract(r'(\d+\.\d+)\s*L')[0]
    df_processed['engine_displacement'] = displacement_extracted.astype(float)
    
    # Calculate median from non-null values only
    disp_median = (df_processed['engine_displacement'].median() 
                   if not df_processed['engine_displacement'].isna().all() 
                   else 2.5)  # fallback median
    
    df_processed['engine_displacement'] = df_processed['engine_displacement'].fillna(disp_median)

    # === V ENGINE FLAG ===
    df_processed['is_v_engine'] = df_processed['engine'].str.contains(r'V\d+', flags=re.IGNORECASE).astype(int)

    # === VEHICLE AGE ===
    df_processed['Vehicle_Age'] = 2025 - df_processed['model_year']

    # === MILEAGE PER YEAR ===
    df_processed['Mileage_per_Year'] = df_processed.apply(
        lambda row: row['milage'] / row['Vehicle_Age']
        if row['Vehicle_Age'] > 0 else row['milage'],
        axis=1
    )

    # === SAFE BINNING ===
    df_processed['Vehicle_Age_Bin'] = pd.cut(df_processed['Vehicle_Age'], bins=4,
                                   labels=['New', 'Mid', 'Old', 'Very Old'])
    df_processed['Mileage_Bin'] = pd.cut(df_processed['milage'], bins=4,
                               labels=['Low', 'Medium', 'High', 'Very High'])

    # === ONE HOT ENCODING FOR BINNED VARIABLES ===
    df_processed = pd.get_dummies(df_processed,
                        columns=['Vehicle_Age_Bin', 'Mileage_Bin'],
                        prefix=['Age', 'Milage'],
                        drop_first=True,
                        dtype=int)

    # === ACCIDENT IMPACT ===
    df_processed['Accident_Impact'] = df_processed['accident'].apply(
        lambda x: 1 if 'accident' in str(x).lower() else 0
    )

    # === CLEAN TITLE ===
    df_processed['clean_title'] = df_processed['clean_title'].apply(
        lambda x: 1 if str(x).strip().lower() == 'yes' else 0
    )

    # === DROP UNUSED COLUMNS ===
    columns_to_drop = ['model', 'model_year', 'engine', 'int_col', 'ext_col', 'accident']
    for col in columns_to_drop:
        if col in df_processed.columns:
            df_processed.drop(columns=[col], inplace=True)

    return df_processed

# ============================================================
# 2. LOAD MODEL, SCALER, AND LABEL ENCODERS
# ============================================================

MODEL_PATH = r"D:\DEPI\Final_project\My_work\xgboost_used_car_price_model.pkl"
SCALER_PATH = r"D:\DEPI\Final_project\My_work\scaler.pkl"
LABEL_ENCODERS_PATH = r"D:\DEPI\Final_project\My_work\label_encoders.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully")
    print(f"📊 Model type: {type(model)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

try:
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler loaded successfully")
except Exception as e:
    print(f"❌ Error loading scaler: {e}")
    exit()

try:
    label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    print("✅ Label encoders loaded successfully")
except Exception as e:
    print(f"❌ Error loading label encoders: {e}")
    exit()

# ============================================================
# 3. CREATE SAMPLE DATA
# ============================================================

raw_sample = pd.DataFrame([{
    "brand": "Lexus",
    "model": "RX 350 RX 350",
    "model_year": 2022,
    "milage": 22372,
    "fuel_type": "Gasoline", 
    "engine": "3.5 Liter DOHC",
    "transmission": "Automatic",
    "ext_col": "Blue",
    "int_col": "Black",
    "accident": "None reported",
    "clean_title": "Yes"
}])


# ============================================================
# 4. APPLY FEATURE ENGINEERING
# ============================================================

processed = apply_feature_engineering(raw_sample)

# ============================================================
# 5. APPLY LABEL ENCODING (SILENT VERSION)
# ============================================================

for col, encoder in label_encoders.items():
    if col in processed.columns:
        # Convert to string
        processed[col] = processed[col].astype(str)
        
        # Handle unseen labels
        unique_values = processed[col].unique()
        for value in unique_values:
            if value not in encoder.classes_:
                # Use the most common class instead of first class
                processed.loc[processed[col] == value, col] = encoder.classes_[0]
        
        # Apply the encoding
        processed[col] = encoder.transform(processed[col])

# ============================================================
# 6. GET EXPECTED COLUMNS
# ============================================================

expected_columns = None

# Try multiple ways to get feature names
try:
    if hasattr(model, 'get_booster'):
        expected_columns = model.get_booster().feature_names
except:
    pass

if expected_columns is None and hasattr(model, 'feature_names_in_'):
    expected_columns = model.feature_names_in_

if expected_columns is None:
    expected_columns = processed.columns.tolist()

# ============================================================
# 7. ALIGN COLUMNS WITH MODEL EXPECTATIONS
# ============================================================

print(f"🔄 Before alignment: {processed.shape}")

# Add missing columns with 0
for col in expected_columns:
    if col not in processed.columns:
        processed[col] = 0

# Remove extra columns that the model doesn't expect
for col in processed.columns:
    if col not in expected_columns:
        processed.drop(columns=[col], inplace=True)

# Reorder columns to match model expectations
processed = processed[expected_columns]

print(f"🔄 After alignment: {processed.shape}")

# ============================================================
# 8. SCALE AND PREDICT
# ============================================================

print("⚡ Making prediction...")
try:
    # Scale the features
    processed_scaled = scaler.transform(processed)
    print("✅ Features scaled successfully")
    
    # Make prediction
    prediction = model.predict(processed_scaled)
    
    print("=" * 60)
    print("🎉 SUCCESS! Prediction Complete")
    print("=" * 60)
    print(f"💰 Predicted Price: ${prediction[0]:,.2f}")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ Error during prediction: {e}")