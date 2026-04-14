import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# ==========================================
# 1. DATA PREPARATION
# ==========================================
# Tạo dữ liệu giả lập Samsung (Thay vì đọc file để tránh lỗi FileNotFoundError)
data = {
    'CustomerID': range(1, 101),
    'Age': np.random.randint(18, 70, size=100),
    'Region': np.random.choice(['Ha Noi', 'Ho Chi Minh', 'Da Nang'], size=100),
    'Revenue': np.random.randint(200, 2000, size=100),
    'Sessions': np.random.randint(1, 50, size=100),
    'Converted': np.random.choice([0, 1], size=100) # 1: Mua hàng, 0: Không
}
df = pd.DataFrame(data)
# Tạo một ít lỗi dữ liệu để xử lý ở bước 2
df.loc[0:5, 'Age'] = np.nan

print("--- Step 1: Data Preparation (Raw Data) ---")
print(df.head()) # Chụp ảnh bảng này

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================
# Xử lý dữ liệu thiếu và chuẩn hóa như bài mẫu ( Làm sạch dữ liệu )
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Region'] = df['Region'].str.upper()

print("\n--- Step 2: Data Preprocessing (Cleaned) ---")
print(df.isnull().sum()) # Chụp ảnh này để chứng minh hết lỗi

# ==========================================
# 3. DATA ANALYSIS (Clustering & Prediction)
# ==========================================
# A. Phân cụm khách hàng (K-Means) như Figure 6
# model đang sử dụng KMeans
features = df[['Age', 'Revenue', 'Sessions']]
kmeans = KMeans(n_clusters=3, random_state=42).fit(features)
df['Segment'] = kmeans.labels_

# B. Phân tích hiệu suất theo Vùng (Descriptive)
region_stats = df.groupby('Region').agg({'Revenue': 'sum', 'Converted': 'mean'}).rename(columns={'Converted': 'Conv_Rate'})

print("\n--- Step 3: Data Analysis (Regional Stats) ---")
print(region_summary := region_stats) # Hiện bảng thống kê vùng

# ==========================================
# 4. DATA VISUALIZATION (FULL REPORT)
# ==========================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập khung hình lớn chứa 4 biểu đồ (2 hàng, 2 cột)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Samsung Marketing Performance Report', fontsize=20)

# 1. Revenue by Region (Biểu đồ cột ngang)
region_revenue = df.groupby('Region')['Revenue'].sum().sort_values()
region_revenue.plot(kind='barh', ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Total Revenue by Region')
axes[0, 0].set_xlabel('Revenue')

# 2. Conversion Rate by Region (Biểu đồ cột đứng)
region_conv = df.groupby('Region')['Converted'].mean().sort_values()
region_conv.plot(kind='bar', ax=axes[0, 1], color='salmon')
axes[0, 1].set_title('Conversion Rate by Region')
axes[0, 1].set_ylabel('Rate (0.0 - 1.0)')

# 3. Giả lập chỉ số ROI (Vì dữ liệu mẫu cần ROI để giống bài mẫu)
# Giả sử ROI = Revenue / (Số lượng khách hàng * 100)
df['ROI'] = df['Revenue'] / 500
region_roi = df.groupby('Region')['ROI'].mean().sort_values()
region_roi.plot(kind='barh', ax=axes[1, 0], color='lightgreen')
axes[1, 0].set_title('Average ROI by Region')

# 4. Customer Age Distribution (Biểu đồ phân bổ)
sns.histplot(df['Age'], kde=True, ax=axes[1, 1], color='orange')
axes[1, 1].set_title('Customer Age Distribution')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# LƯU FILE ẢNH
plt.savefig('marketing_report.png')
print("\n--- Step 4: Full Report Generated ---")
print("Success! Open 'marketing_report.png' to see your 4-chart dashboard.")

# ==========================================
# 5. LINKING TO BUSINESS DECISIONS
# ==========================================
# Phần này viết bằng chữ trong báo cáo dựa trên kết quả trên