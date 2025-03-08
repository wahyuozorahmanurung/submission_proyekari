import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load dataset
file_path = "submission_ari.csv"
df = pd.read_csv(file_path)

# Page Configurations
st.set_page_config(page_title="Air Quality Dashboard", layout="wide", initial_sidebar_state="expanded")

# Sidebar Menu
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1163/1163624.png", width=100)
st.sidebar.title("🌍 Air Quality Dashboard")
st.sidebar.markdown("---")
st.sidebar.header("🔍 Filter Data")
station_choice = st.sidebar.selectbox("Pilih Stasiun", df['station'].unique())
df_filtered = df[df['station'] == station_choice]

# Main Content
st.title("🌍 Air Quality Dashboard")
st.markdown("### Analisis Kualitas Udara berdasarkan Dataset")
st.markdown("---")

st.subheader("📊 Distribusi Polutan")
df_air_quality = df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].dropna()
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df_air_quality, ax=ax)
ax.set_title("Distribusi Polutan")
st.pyplot(fig)

st.markdown("---")
st.subheader("📈 Tren PM2.5 per Bulan")
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df_monthly = df.groupby(['year', 'month'])['PM2.5'].mean().reset_index()
df_monthly['time'] = pd.to_datetime(df_monthly[['year', 'month']].assign(day=1))

fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x=df_monthly['time'], y=df_monthly['PM2.5'], marker='o', linestyle='-', ax=ax)
ax.set_xlabel("Waktu")
ax.set_ylabel("PM2.5")
ax.set_title("Tren Rata-rata PM2.5 per Bulan")
st.pyplot(fig)

st.markdown("---")
st.subheader("🔗 Hubungan antara PM2.5 dan Variabel Cuaca")
selected_features = ['PM2.5', 'TEMP', 'PRES', 'DEWP']
df_pairplot = df[selected_features].dropna()
st.pyplot(sns.pairplot(df_pairplot))

st.markdown("---")
st.subheader("📉 Dekomposisi Time Series PM2.5")
df.set_index('datetime', inplace=True)
df_monthly_pm25 = df['PM2.5'].resample('M').mean().dropna()
decomposition = sm.tsa.seasonal_decompose(df_monthly_pm25, model='additive')

fig, axes = plt.subplots(3, 1, figsize=(12, 8))
decomposition.trend.plot(ax=axes[0], title='Tren', legend=True)
decomposition.seasonal.plot(ax=axes[1], title='Musiman', legend=True, color='green')
decomposition.resid.plot(ax=axes[2], title='Residual', legend=True, color='red')
st.pyplot(fig)

# Set background color
st.markdown(
    """
    <style>
        body {
            background-color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)
