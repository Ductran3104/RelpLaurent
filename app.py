# ==============================================================================
# File: app.py
# Mô tả: Ứng dụng Streamlit dự báo nhu cầu bán hàng cho ABC Manufacturing
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# --- Cấu hình trang ---
st.set_page_config(
    page_title="Dự báo Nhu cầu ABC",
    page_icon="📈",
    layout="wide"
)

# --- Các hàm xử lý dữ liệu và mô hình ---

# Sử dụng cache để không phải tạo lại dữ liệu mỗi lần tương tác
@st.cache_data
def generate_and_process_data():
    """Tạo dữ liệu giả lập và xử lý nó."""
    np.random.seed(42)
    date_range = pd.date_range(start='2022-01-01', end='2025-09-30', freq='D')
    skus = ['SKU-A01', 'SKU-B02', 'SKU-C03']
    data = []
    for date in date_range:
        month_factor = 1.8 if date.month in [10, 11, 12] else 1 + (date.month - 1) / 12 * 0.5
        time_factor = 1 + (date - date_range[0]).days / len(date_range) * 1.5
        for sku in skus:
            if sku == 'SKU-A01': base_quantity, unit_price = np.random.randint(50, 80), 120
            elif sku == 'SKU-B02': base_quantity, unit_price = np.random.randint(20, 40), 250
            else: base_quantity, unit_price = np.random.randint(5, 15), 500
            quantity = int(base_quantity * month_factor * time_factor * (1 + np.random.rand() * 0.2))
            data.append([date, sku, quantity, unit_price])
    
    df = pd.DataFrame(data, columns=['Date', 'SKU', 'Quantity', 'UnitPrice'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_sales = df.groupby(['Month', 'SKU']).agg(TotalQuantity=('Quantity', 'sum')).reset_index()
    monthly_sales['Month'] = monthly_sales['Month'].dt.to_timestamp()
    return monthly_sales

def train_and_forecast(data, sku_to_forecast):
    """Huấn luyện mô hình và dự báo cho một SKU cụ thể."""
    df_model = data[data['SKU'] == sku_to_forecast].copy()
    df_model.set_index('Month', inplace=True)
    
    # Feature Engineering
    df_model['time_index'] = np.arange(len(df_model))
    df_model['month_of_year'] = df_model.index.month
    df_model['lag_1'] = df_model['TotalQuantity'].shift(1)
    df_model['rolling_mean_3'] = df_model['TotalQuantity'].rolling(window=3).mean().shift(1)
    df_model.dropna(inplace=True)
    
    # Huấn luyện mô hình
    features = ['time_index', 'month_of_year', 'lag_1', 'rolling_mean_3']
    target = 'TotalQuantity'
    X, y = df_model[features], df_model[target]
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Dự báo cho tương lai
    all_known_data = df_model.copy()
    future_dates = pd.date_range(start='2025-10-01', periods=3, freq='M')
    future_predictions_list = []
    
    for date in future_dates:
        time_index = all_known_data['time_index'].iloc[-1] + 1
        month_of_year = date.month
        lag_1 = all_known_data['TotalQuantity'].iloc[-1]
        rolling_mean_3 = all_known_data['TotalQuantity'].tail(3).mean()
        future_X = pd.DataFrame([[time_index, month_of_year, lag_1, rolling_mean_3]], columns=features)
        prediction = model.predict(future_X)[0]
        future_predictions_list.append(int(prediction))
        new_row = pd.DataFrame({'TotalQuantity': [prediction], 'time_index': [time_index]}, index=[date])
        all_known_data = pd.concat([all_known_data, new_row])
        
    forecast_df = pd.DataFrame({
        'Tháng Dự Báo': future_dates.strftime('%Y-%m'),
        'Số Lượng Dự Báo': future_predictions_list
    })
    
    return forecast_df

# --- Giao diện người dùng của ứng dụng Streamlit ---

st.title("📈 Ứng dụng Dự báo Nhu cầu Bán hàng")
st.write("**Doanh nghiệp:** ABC Manufacturing")
st.write("---")

# Tải và xử lý dữ liệu
monthly_data = generate_and_process_data()

# Hiển thị các phân tích tổng quan
st.header("1. Phân tích Tổng quan Dữ liệu Lịch sử")

col1, col2 = st.columns(2)

with col1:
    st.subheader("So sánh Hiệu suất bán hàng theo SKU")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.lineplot(ax=ax1, data=monthly_data, x='Month', y='TotalQuantity', hue='SKU', marker='o', ci=None)
    st.pyplot(fig1)

with col2:
    st.subheader("Phân tích Mùa vụ theo Tháng")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    monthly_data['MonthOfYear'] = monthly_data['Month'].dt.month
    sns.boxplot(ax=ax2, data=monthly_data, x='MonthOfYear', y='TotalQuantity')
    st.pyplot(fig2)

# Phần dự báo
st.write("---")
st.header("2. Dự báo Nhu cầu cho Quý Tới")
st.write("Chọn một sản phẩm và nhấn nút để xem dự báo nhu cầu cho 3 tháng tới.")

# Lựa chọn sản phẩm
sku_options = monthly_data['SKU'].unique()
selected_sku = st.selectbox("Chọn một SKU để dự báo:", sku_options)

# Nút bấm để chạy dự báo
if st.button(f"🚀 Chạy dự báo cho {selected_sku}"):
    with st.spinner(f"Đang huấn luyện mô hình và dự báo cho {selected_sku}... Vui lòng chờ trong giây lát."):
        # Chạy mô hình và lấy kết quả
        forecast_result = train_and_forecast(monthly_data, selected_sku)
        
        st.success("Dự báo thành công!")
        
        st.subheader(f"Bảng Kết quả Dự báo cho {selected_sku}")
        st.table(forecast_result.style.format({'Số Lượng Dự Báo': '{:,}'}))
        
        st.info(
            "**Đề xuất:** Dựa vào bảng trên, bộ phận kế hoạch có thể đưa ra quyết định về sản lượng sản xuất, "
            "tồn kho và kế hoạch mua nguyên vật liệu cho quý tới."
        )

st.sidebar.header("Thông tin dự án")
st.sidebar.info(
    "Đây là ứng dụng web demo, sử dụng Streamlit để triển khai một mô hình "
    "khoa học dữ liệu dự báo nhu cầu sản phẩm. Ứng dụng này giúp chuyển đổi "
    "các phân tích phức tạp thành một công cụ hỗ trợ ra quyết định trực quan và dễ sử dụng."
)
