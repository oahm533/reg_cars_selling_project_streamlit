import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import LocalOutlierFactor

# ----- sidebar -------
st.sidebar.image('cars.png')
st.sidebar.title("🧮 إدخال البيانات")

# ----- header -------
st.image('cars2.png')

# قائمة الماركات
brands_list = (
    'Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
    'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
    'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
    'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
    'Ambassador', 'Ashok', 'Isuzu', 'Opel'
)

brand = st.sidebar.selectbox("🏷️ اختر الماركة", brands_list, key="brand")
year = st.sidebar.number_input("📅 سنة الصنع", min_value=1990, max_value=2025, value=2015, key="year")
km_driven = st.sidebar.number_input("🚗 عدد الكيلومترات المقطوعة", min_value=0, value=50000, step=1000, key="km_driven")
fuel = st.sidebar.selectbox("⛽ نوع الوقود", ("Petrol", "Diesel", "CNG", "LPG", "Electric"), key="fuel")
seller_type = st.sidebar.selectbox("🧍 نوع البائع", ("Individual", "Dealer", "Trustmark Dealer"), key="seller_type")
transmission = st.sidebar.selectbox("⚙️ ناقل الحركة", ("Manual", "Automatic"), key="transmission")
owner = st.sidebar.selectbox("👥 عدد المالكين السابقين", ("First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"), key="owner")
mileage = st.sidebar.number_input("📏 المسافة المقطوعة (كم/لتر)", min_value=0.0, value=18.0, key="mileage")
engine = st.sidebar.number_input("🧠 سعة المحرك (cc)", min_value=500, max_value=6000, value=1500, key="engine")
max_power = st.sidebar.number_input("⚡ أقصى قوة (حصان)", min_value=30.0, value=100.0, key="max_power")
seats = st.sidebar.selectbox("💺 عدد المقاعد", (2, 4, 5, 6, 7, 8, 9, 10), key="seats")
torque = st.sidebar.number_input("🌀 عزم الدوران (Nm)", min_value=20.0, max_value=1000.0, value=113.8, key="torque")

st.sidebar.markdown("### تعليمات الإدخال:")
st.sidebar.markdown("""
- اختر الماركة من القائمة المنسدلة.
- أدخل سنة الصنع بين 1990 و 2025.
- أدخل عدد الكيلومترات المقطوعة. تأكد من أن الرقم معقول.
- اختر نوع الوقود من الخيارات المتاحة (البنزين، الديزل، إلخ).
- اختر نوع البائع بين فردي أو تاجر.
- اختر نوع ناقل الحركة (يدوي أو أوتوماتيكي).
- اختر عدد المالكين السابقين.
- أدخل سعة المحرك (بـ "سي سي").
- أدخل أقصى قوة للمحرك (بـ "حصان").
- أدخل عزم الدوران (بـ "نيوتن متر").
""")

if not (year and km_driven and fuel and seller_type and transmission and owner and mileage and engine and max_power and torque and seats and brand):
    st.warning("⚠️ تأكد من إدخال جميع الحقول بشكل صحيح.")

input_data = pd.DataFrame({
    "year": (year,),
    "km_driven": (km_driven,),
    "fuel": (fuel,),
    "seller_type": (seller_type,),
    "transmission": (transmission,),
    "owner": (owner,),
    "mileage": (mileage,),
    "engine": (engine,),
    "max_power": (max_power,),
    "torque": (torque,),
    "seats": (seats,),
    "brand": (brand,)
})

st.write("🔢 المدخلات التي اخترتها:")
st.write(input_data)

# تحميل البيانات الأصلية
df_car = pd.read_csv("reg_cars_selling.csv")

# ------- cleaning -------
df_car.drop_duplicates(inplace=True)
df_car = df_car.dropna()

df_car['max_power'] = df_car['max_power'].astype(str).str.replace(r' bhp$', '', regex=True).astype(float)

df_car['mileage'] = df_car['mileage'].astype(str).str.replace(r' kmpl$', '', regex=True)
df_car['mileage'] = df_car['mileage'].str.replace(r' km/kg$', '', regex=True).astype(float)

df_car['engine'] = df_car['engine'].astype(str).str.replace(r' CC$', '', regex=True)
df_car['engine'] = df_car['engine'].str.replace(r'CC', '', regex=True).astype(float)

# استخراج الماركة وحذف عمود name بما يتوافق مع Pandas 3.0
if 'name' in df_car.columns:
    df_car['brand'] = df_car['name'].apply(lambda x: str(x).split().pop(0))
    df_car.drop(columns='name', inplace=True)

df_car['torque'] = df_car['torque'].astype(str)

def extract_torque(value):
    if pd.isna(value):
        return None
    value = str(value).lower().strip()
    matches = re.findall(r'([\d\.]+)\s*(nm|kgm)?', value, re.IGNORECASE)
    torque_values = list()
    for val_str, unit_str in matches:
        torque_value = float(val_str)
        unit = unit_str.lower() if unit_str else 'nm'
        if unit == "kgm":
            torque_value *= 9.80665
        torque_values.append(torque_value)
    if torque_values:
        return sum(torque_values) / len(torque_values)
    return None

df_car['torque'] = df_car['torque'].apply(extract_torque)
df_car = df_car.dropna()

if 'selling_price' in df_car.columns:
    df_car['selling_price_INR'] = df_car['selling_price']
    df_car.drop(columns='selling_price', inplace=True)

df_car_numeric_analysis = df_car.copy()

# ترميز الأعمدة النصية
categorical_columns = list(('fuel', 'seller_type', 'transmission', 'owner', 'brand'))
df_car[categorical_columns] = df_car[categorical_columns].fillna("Unknown")

input_data_encoded = input_data.copy()
for col in categorical_columns:
    encoder = LabelEncoder()
    df_car[col] = encoder.fit_transform(df_car[col].astype(str))
    input_data_encoded[col] = input_data_encoded[col].fillna("Unknown").astype(str)
    input_data_encoded[col] = input_data_encoded[col].apply(
        lambda x: encoder.transform((x,)).item() if x in encoder.classes_ else 0
    )

# إزالة القيم الشاذة باستخدام Local Outlier Factor
lof = LocalOutlierFactor()
outlier_labels = lof.fit_predict(df_car)
df_car['outlier_labels'] = outlier_labels
df_car = df_car[df_car['outlier_labels'] != -1]
df_car.drop(columns='outlier_labels', inplace=True)

# تقسيم البيانات
X = df_car.drop(columns='selling_price_INR')
y = df_car['selling_price_INR']

# توحيد ترتيب أعمدة المدخلات مع بيانات التدريب
input_data_encoded = input_data_encoded[X.columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# التنبؤ بالسعر بناءً على المدخلات
predicted_price = model.predict(input_data_encoded).item()

# عرض السعر المتوقع ومقاييس الدقة
st.title("💰 السعر المتوقع بناءً على المدخلات:")
st.subheader(f"₹ {predicted_price:,.0f}")

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.write(f"📊 دقة النموذج (R²): {r2:.2f}")
st.write(f"📉 متوسط الخطأ (RMSE): ₹ {rmse:,.0f}")
st.write(f"📁 حجم البيانات النظيفة المستخدمة: {len(df_car):,} سجل")

# ------- الرسوم البيانية -------
st.subheader("📊 مقارنة أسعار السيارات حسب نوع الوقود")

fuel_avg_prices = df_car_numeric_analysis.groupby("fuel")["selling_price_INR"].mean().sort_values(ascending=False)
st.write("### 💡 متوسط الأسعار حسب نوع الوقود:")
st.dataframe(fuel_avg_prices.reset_index().rename(columns={"fuel": "نوع الوقود", "selling_price_INR": "متوسط السعر"}))

fig1, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=fuel_avg_prices.index, y=fuel_avg_prices.values, palette="viridis", ax=ax)
plt.xlabel('نوع الوقود (Fuel)', fontsize=14, weight='bold', color='darkblue')
plt.ylabel('متوسط السعر (INR)', fontsize=12)
plt.xticks(rotation=0, fontsize=12)
st.pyplot(fig1)

st.subheader("📊 توزيع السيارات حسب الماركة")
fig2, ax2 = plt.subplots(figsize=(14, 8))
sns.countplot(
    x='brand',
    data=df_car_numeric_analysis,
    palette='mako',
    edgecolor='black',
    linewidth=1.5,
    ax=ax2
)
plt.title('Distribution of Cars by Brand', fontsize=18, weight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
st.pyplot(fig2)
