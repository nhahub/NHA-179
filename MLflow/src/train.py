import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

mlflow.set_tracking_uri("file:./mlruns")

# -------- CONFIG --------
# استخدام اسم الملف الحقيقي الذي أرسلته
DATA_PATH = "data/Second_ride_before_modeling.csv" 
MODEL_NAME = "xgboost_used_car_price_model"

# -------- LOAD DATA --------
df = pd.read_csv(DATA_PATH)

# ==========================================================
# 🥇 الحل النهائي: تطبيق One-Hot Encoding على الأعمدة الفئوية
# ==========================================================
# pd.get_dummies ستقوم بتحديد وتحويل جميع الأعمدة من نوع 'object' إلى أرقام (0s و 1s)
df_processed = pd.get_dummies(df, drop_first=True) 

X = df_processed.drop("price", axis=1)
y = df_processed["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------- TRAIN MODEL --------
model = xgb.XGBRegressor(
n_estimators=300,
learning_rate=0.05,
max_depth=6,
subsample=0.8,
colsample_bytree=0.8,
objective="reg:squarederror"
)

# -------- MLflow START --------
mlflow.set_experiment("UsedCarsPricePrediction")

with mlflow.start_run():
    
    # ==========================================================
    # 🆕 تسجيل مجموعة البيانات (Dataset Logging)
    # ==========================================================
    # تكوين المسار الكامل للملف الأصلي كـ Source Reference
    data_source_path = os.path.join(os.getcwd(), DATA_PATH)

    # إنشاء مرجع البيانات من X_train 
    training_data = mlflow.data.from_pandas(
        X_train, 
        source=data_source_path,
        name="UsedCar_Training_Data_Encoded"
    )
    # تسجيل مجموعة البيانات في MLflow
    mlflow.log_input(training_data, context="training")
    
    # ==========================================================

    model.fit(X_train, y_train) # هنا لن يفشل لأن X_train كلها أرقام

    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # ---- log params / metrics ----
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("max_depth", 6)

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    # ---- log model ----
    mlflow.sklearn.log_model(sk_model=model,artifact_path="model",registered_model_name=MODEL_NAME)

print("Model logged successfully!")