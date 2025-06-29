import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import warnings
import joblib 

from matplotlib.legend import Legend
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from scipy.stats import mstats

df = pd.read_csv('./content/train.csv')
selected_columns = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', '2ndFlrSF', 'BsmtFinSF1', '1stFlrSF', 'GarageCars' , 'GarageArea', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'Neighborhood', 'TotRmsAbvGrd', 'SalePrice']
df = df[selected_columns]
print(f"Kích thước dữ liệu: {df.shape}")
print(df.info())
print("\n📈 Thống kê mô tả cho biến số:")
print(df.describe().round(2))
print(df.head())

print("🔍 PHÂN TÍCH GIÁ TRỊ THIẾU")
missing = df.isnull().sum()
missing_percent = ((missing / len(df)) * 100).round(2)

missing_df = pd.DataFrame({
            'Cột': missing.index,
            'Số lượng thiếu': missing.values,
            'Tỷ lệ %': missing_percent.values})
missing_df = missing_df[missing_df['Số lượng thiếu'] > 0].sort_values('Tỷ lệ %', ascending=False)
print(missing_df)

print("🎯 PHÂN TÍCH OUTLIERS")

numerical_cols = df.select_dtypes(include=[np.number]).columns
outlier_info = {}

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_count = outliers.shape[0]
    outlier_percent = outlier_count / df.shape[0] * 100

    outlier_info[col] = {
        'outlier_count': outlier_count,
        'outlier_percent': round(outlier_percent, 2)
    }

# Hiển thị kết quả
outlier_df = pd.DataFrame(outlier_info).T
# print(outlier_df)
# Tạo scatter plots cho biến số
target_col = 'SalePrice'  # Thay đổi tên này nếu cần

# Tách biến số (loại bỏ cột target)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numerical_cols:
    numerical_cols.remove(target_col)

# Tạo scatter plots cho biến số
if len(numerical_cols) > 0:
    n_cols = min(3, len(numerical_cols))
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    fig.suptitle(f' Scatter Plot: Biến số vs {target_col}', fontsize=16, y=0.99)

    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        if i < len(axes):
            # Scatter plot
            axes[i].scatter(df[col], df[target_col], alpha=0.5, s=20)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel(target_col)
            axes[i].set_title(f'{col} vs {target_col}\nCorr: {df[col].corr(df[target_col]):.3f}')
            axes[i].grid(True, alpha=0.3)

            # Thêm trend line
            try:
                z = np.polyfit(df[col].dropna(), df[target_col][df[col].dropna().index], 1)
                p = np.poly1d(z)
                axes[i].plot(df[col], p(df[col]), "r--", alpha=0.8, linewidth=2)
            except:
                pass

    # Ẩn các subplot thừa
    for i in range(len(numerical_cols), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    # plt.show()

# Heatmap
plt.figure(figsize=(12, 10))
corr_matrix = df.corr(numeric_only=True)  # Tính ma trận tương quan
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
            center=0, fmt='.2f', linewidths=0.5)
plt.title('Ma trận tương quan giữa các biến số')
plt.tight_layout()
# plt.show()

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Histogram
df['SalePrice'].hist(bins=50, ax=axes[0,0], alpha=0.7, edgecolor='black')
axes[0,0].set_title('Phân phối SalePrice')
axes[0,0].set_xlabel('SalePrice')
axes[0,0].set_ylabel('Frequency')

# Box plot
df['SalePrice'].plot(kind='box', ax=axes[0,1])
axes[0,1].set_title('Box Plot SalePrice')

# Q-Q plot
from scipy import stats
stats.probplot(df['SalePrice'], dist="norm", plot=axes[1,0])
axes[1,0].set_title('Q-Q Plot (Normal Distribution)')

# Log transformation
log_price = np.log1p(df['SalePrice'])
log_price.hist(bins=50, ax=axes[1,1], alpha=0.7, edgecolor='black')
axes[1,1].set_title('Log-transformed SalePrice')

plt.tight_layout()
# plt.show()

def validate_house_data(df):
    """
    Kiểm định dữ liệu đơn giản cho dataset House Prices
    """
    print("=" * 60)
    print("           KIỂM ĐỊNH DỮ LIỆU HOUSE PRICES")
    print("=" * 60)

    # 1. THÔNG TIN TỔNG QUAN
    print("\n1. THÔNG TIN TỔNG QUAN:")
    print(f"   - Số dòng: {df.shape[0]:,}")
    print(f"   - Số cột: {df.shape[1]:,}")
    print(f"   - Kích thước bộ nhớ: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # 2. KIỂM TRA MISSING VALUES
    print("\n2. MISSING VALUES:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Cột': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing %': missing_percent.values
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)

    if len(missing_df) > 0:
        print(f"   - Có {len(missing_df)} cột có missing values:")
        for _, row in missing_df.iterrows():
            print(f"     • {row['Cột']}: {row['Missing Count']} ({row['Missing %']:.1f}%)")
    else:
        print("   ✅ Không có missing values!")

    # 3. KIỂM TRA DUPLICATE
    print(f"\n3. DUPLICATE RECORDS:")
    duplicates = df.duplicated().sum()
    print(f"   - Số dòng trùng lặp: {duplicates}")
    if duplicates > 0:
        print("   ⚠️  Cần xử lý duplicate!")
    else:
        print("   ✅ Không có duplicate!")

    # 4. KIỂM TRA CÁC CỘT NUMERIC
    print(f"\n4. CÁC CỘT NUMERIC ({df.select_dtypes(include=[np.number]).shape[1]} cột):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col == 'SalePrice':
            continue

        # Kiểm tra giá trị âm
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            print(f"   ⚠️  {col}: có {negative_count} giá trị âm")

        # Kiểm tra outliers (IQR method)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

        if outliers > len(df) * 0.05:  # Nếu outliers > 5%
            print(f"   ⚠️  {col}: có {outliers} outliers ({outliers/len(df)*100:.1f}%)")

    # 5. KIỂM TRA TARGET VARIABLE (SalePrice)
    if 'SalePrice' in df.columns:
        print(f"\n5. TARGET VARIABLE (SalePrice):")
        price = df['SalePrice']
        print(f"   - Min: ${price.min():,.0f}")
        print(f"   - Max: ${price.max():,.0f}")
        print(f"   - Mean: ${price.mean():,.0f}")
        print(f"   - Median: ${price.median():,.0f}")
        print(f"   - Std: ${price.std():,.0f}")

        # Kiểm tra phân phối
        skewness = stats.skew(price)
        print(f"   - Skewness: {skewness:.3f}", end="")
        if abs(skewness) > 1:
            print(" (⚠️ Highly skewed)")
        elif abs(skewness) > 0.5:
            print(" (⚠️ Moderately skewed)")
        else:
            print(" (✅ Normal)")

        # Kiểm tra outliers cho SalePrice
        Q1 = price.quantile(0.25)
        Q3 = price.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        price_outliers = ((price < lower_bound) | (price > upper_bound)).sum()
        print(f"   - Outliers: {price_outliers} ({price_outliers/len(df)*100:.1f}%)")

    # 6. KIỂM TRA CÁC CỘT CATEGORICAL
    print(f"\n6. CÁC CỘT CATEGORICAL ({df.select_dtypes(include=['object']).shape[1]} cột):")
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        unique_count = df[col].nunique()
        total_count = len(df[col].dropna())

        # Cảnh báo nếu có quá nhiều categories
        if unique_count > total_count * 0.5:
            print(f"   ⚠️  {col}: {unique_count} unique values ({unique_count/total_count*100:.1f}% của dữ liệu)")
        elif unique_count > 20:
            print(f"   ⚠️  {col}: {unique_count} categories (có thể cần group)")

    # 7. KIỂM TRA LOGIC BUSINESS
    print(f"\n7. KIỂM TRA LOGIC BUSINESS:")

    # YearBuilt vs YearRemodAdd
    if 'YearBuilt' in df.columns and 'YearRemodAdd' in df.columns:
        invalid_years = (df['YearRemodAdd'] < df['YearBuilt']).sum()
        if invalid_years > 0:
            print(f"   ⚠️  {invalid_years} nhà có năm remodel trước năm xây dựng")

    # GarageArea vs GarageCars
    if 'GarageArea' in df.columns and 'GarageCars' in df.columns:
        # Nhà có garage area > 0 nhưng GarageCars = 0
        weird_garage = ((df['GarageArea'] > 0) & (df['GarageCars'] == 0)).sum()
        if weird_garage > 0:
            print(f"   ⚠️  {weird_garage} nhà có diện tích garage > 0 nhưng không có xe")

    # Bedroom logic
    if 'BedroomAbvGr' in df.columns:
        zero_bedroom = (df['BedroomAbvGr'] == 0).sum()
        if zero_bedroom > 0:
            print(f"   ⚠️  {zero_bedroom} nhà có 0 phòng ngủ")

    # Total area logic
    if all(col in df.columns for col in ['1stFlrSF', '2ndFlrSF', 'GrLivArea']):
        calculated_area = df['1stFlrSF'] + df['2ndFlrSF']
        area_mismatch = (abs(calculated_area - df['GrLivArea']) > 1).sum()
        if area_mismatch > 0:
            print(f"   ⚠️  {area_mismatch} nhà có diện tích không khớp logic")

    print(f"\n8. TÓM TẮT:")
    issues = []
    if len(missing_df) > 0:
        issues.append(f"{len(missing_df)} cột có missing values")
    if duplicates > 0:
        issues.append(f"{duplicates} dòng duplicate")

    if len(issues) == 0:
        print("   ✅ Dữ liệu tương đối sạch!")
    else:
        print("   ⚠️  Các vấn đề cần xử lý:")
        for issue in issues:
            print(f"      - {issue}")

    return missing_df

# Hàm tạo biểu đồ phân phối cho các cột quan trọng
def plot_distributions(df, key_columns=None):
    """
    Vẽ biểu đồ phân phối cho các cột quan trọng
    """
    if key_columns is None:
        # Các cột quan trọng thường dùng để dự đoán giá nhà
        key_columns = ['SalePrice', 'GrLivArea', 'OverallQual', 'YearBuilt',
                      'TotalBsmtSF', 'GarageArea', 'LotArea']

    # Chỉ lấy các cột có trong dataset
    key_columns = [col for col in key_columns if col in df.columns]

    if len(key_columns) == 0:
        print("Không có cột nào để vẽ biểu đồ!")
        return

    n_cols = min(3, len(key_columns))
    n_rows = (len(key_columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    for i, col in enumerate(key_columns):
        if df[col].dtype in ['int64', 'float64']:
            axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean: {df[col].mean():.0f}')
            axes[i].axvline(df[col].median(), color='orange', linestyle='--', label=f'Median: {df[col].median():.0f}')
        axes[i].set_title(f'Phân phối của {col}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    # Ẩn các subplot thừa
    for i in range(len(key_columns), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    # plt.show()

# CÁCH SỬ DỤNG:
# ===============
# 1. Load dữ liệu
# df = pd.read_csv('your_dataset.csv')

# 2. Chạy validation
# missing_report = validate_house_data(df)

# 3. Vẽ biểu đồ (tùy chọn)
# plot_distributions(df)

# 4. Xem chi tiết missing values
# print(missing_report)

print("✅ Code kiểm định dữ liệu đã sẵn sàng!")
print("Sử dụng: missing_report = validate_house_data(df)")
print("Vẽ biểu đồ: plot_distributions(df)")

missing_report = validate_house_data(df)
plot_distributions(df)
print(missing_report)

# Tách features và target
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

print(f"   ├── Shape cuối: X={X.shape}, y={y.shape}")
print("✅ Preprocessing hoàn thành!")

# 9. TRAIN-VAL-TEST SPLIT
print("\n🔄 Chia dữ liệu train/val/test...")

# Chia train (60%) và temp (40%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=pd.qcut(y, q=5, duplicates='drop')
)

# Chia temp thành val (20%) và test (20%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=pd.qcut(y_temp, q=5, duplicates='drop')
)
print(f"   ├── Train: {X_train.shape} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   ├── Val: {X_val.shape} ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"   ├── Test: {X_test.shape} ({X_test.shape[0]/len(X)*100:.1f}%)")

y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_test)
y_test_log = np.log1p(y_test)

train_data = X_train.copy()
train_data['SalePrice'] = y_train

train_data = X_train.copy()
train_data['SalePrice'] = y_train

neigh_price = train_data.groupby('Neighborhood')['SalePrice'].mean()
# Map vào train
X_train['Neighborhood_Price'] = X_train['Neighborhood'].map(neigh_price)

# Map vào val/test (nếu Neighborhood nào chưa thấy, thì điền NaN hoặc giá trị trung bình)
X_val['Neighborhood_Price'] = X_val['Neighborhood'].map(neigh_price)
X_val['Neighborhood_Price'].fillna(neigh_price.mean(), inplace=True)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import mstats

# =====================================================================
# 🔄 PREPROCESSING PIPELINE - THỨ TỰ ĐÚNG
# =====================================================================

print("🏠 BẮT ĐẦU PREPROCESSING HOUSING DATA...")
print("=" * 60)

# 1. LOAD DATA (giả sử df đã được load)
print("\n1️⃣ Load và kiểm tra dữ liệu ban đầu...")
print(f"   ├── Shape gốc: {df.shape}")
print(f"   ├── Missing values: {df.isnull().sum().sum()}")

# 2. XỬ LÝ CƠ BẢN (chỉ những thứ cần thiết cho split)
print("\n2️⃣ Xử lý cơ bản trước khi split...")

# Chỉ drop những columns hoàn toàn không cần thiết
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)
    print("   ├── Dropped Id column")

# Backup original data
df_original = df.copy()
print("   ├── Backup data gốc")

# 3. TÁCH FEATURES VÀ TARGET (RAW DATA)
print("\n3️⃣ Tách features và target...")
X_raw = df.drop('SalePrice', axis=1)
y_raw = df['SalePrice']

print(f"   ├── X_raw shape: {X_raw.shape}")
print(f"   ├── y_raw shape: {y_raw.shape}")
print(f"   ├── y_raw range: ${y_raw.min():,.0f} to ${y_raw.max():,.0f}")

# 4. TRAIN-VAL-TEST SPLIT (QUAN TRỌNG: SPLIT RAW DATA TRƯỚC!)
print("\n4️⃣ Chia dữ liệu train/val/test...")

# Chia train (60%) và temp (40%)
X_train_raw, X_temp_raw, y_train_raw, y_temp_raw = train_test_split(
    X_raw, y_raw, 
    test_size=0.4, 
    random_state=42, 
    stratify=pd.qcut(y_raw, q=5, duplicates='drop')
)

# Chia temp thành val (20%) và test (20%)
X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
    X_temp_raw, y_temp_raw, 
    test_size=0.5, 
    random_state=42, 
    stratify=pd.qcut(y_temp_raw, q=5, duplicates='drop')
)

print(f"   ├── Train: {X_train_raw.shape} ({X_train_raw.shape[0]/len(X_raw)*100:.1f}%)")
print(f"   ├── Val: {X_val_raw.shape} ({X_val_raw.shape[0]/len(X_raw)*100:.1f}%)")
print(f"   ├── Test: {X_test_raw.shape} ({X_test_raw.shape[0]/len(X_raw)*100:.1f}%)")

# 5. MISSING VALUE IMPUTATION (CHỈ HỌC TỪ TRAIN)
print("\n5️⃣ Xử lý missing values...")

# Copy để xử lý
X_train = X_train_raw.copy()
X_val = X_val_raw.copy()
X_test = X_test_raw.copy()

# 5.1 Garage columns - học từ train
print("   ├── Xử lý Garage columns...")
garage_cols = ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_cols:
    if col in X_train.columns:
        X_train[col] = X_train[col].fillna('No_Garage')
        X_val[col] = X_val[col].fillna('No_Garage')
        X_test[col] = X_test[col].fillna('No_Garage')

# 5.2 Basement columns - học từ train
print("   ├── Xử lý Basement columns...")
basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for col in basement_cols:
    if col in X_train.columns:
        X_train[col] = X_train[col].fillna('No_Basement')
        X_val[col] = X_val[col].fillna('No_Basement')
        X_test[col] = X_test[col].fillna('No_Basement')

# 5.3 LotFrontage - impute theo median của Neighborhood (CHỈ HỌC TỪ TRAIN)
print("   ├── Xử lý LotFrontage...")
if 'LotFrontage' in X_train.columns and 'Neighborhood' in X_train.columns:
    # Tính median từ train data thôi
    neighborhood_medians = X_train.groupby('Neighborhood')['LotFrontage'].median()
    
    # Apply cho tất cả sets
    X_train['LotFrontage'] = X_train['LotFrontage'].fillna(
        X_train['Neighborhood'].map(neighborhood_medians)
    )
    X_val['LotFrontage'] = X_val['LotFrontage'].fillna(
        X_val['Neighborhood'].map(neighborhood_medians)
    )
    X_test['LotFrontage'] = X_test['LotFrontage'].fillna(
        X_test['Neighborhood'].map(neighborhood_medians)
    )

# 5.4 MasVnrArea - impute bằng 0
print("   ├── Xử lý MasVnrArea...")
if 'MasVnrArea' in X_train.columns:
    X_train['MasVnrArea'] = X_train['MasVnrArea'].fillna(0)
    X_val['MasVnrArea'] = X_val['MasVnrArea'].fillna(0)
    X_test['MasVnrArea'] = X_test['MasVnrArea'].fillna(0)

# 5.5 Electrical - impute bằng mode (CHỈ HỌC TỪ TRAIN)
print("   ├── Xử lý Electrical...")
if 'Electrical' in X_train.columns:
    train_mode = X_train['Electrical'].mode()[0]
    X_train['Electrical'] = X_train['Electrical'].fillna(train_mode)
    X_val['Electrical'] = X_val['Electrical'].fillna(train_mode)
    X_test['Electrical'] = X_test['Electrical'].fillna(train_mode)

# 6. OUTLIER TREATMENT (CHỈ HỌC TỪ TRAIN)
print("\n6️⃣ Xử lý outliers...")
outlier_cols = ['BsmtFinSF2', 'EnclosedPorch', 'ScreenPorch', 'OpenPorchSF']
for col in outlier_cols:
    if col in X_train.columns:
        # Tính percentiles từ train data
        p1 = np.percentile(X_train[col], 1)
        p99 = np.percentile(X_train[col], 99)
        
        # Apply winsorization cho tất cả sets với cùng thresholds
        X_train[col] = np.clip(X_train[col], p1, p99)
        X_val[col] = np.clip(X_val[col], p1, p99)
        X_test[col] = np.clip(X_test[col], p1, p99)

print(f"   ├── Winsorized {len(outlier_cols)} columns")

# 7. FEATURE ENGINEERING (CHỈ HỌC TỪ TRAIN)
print("\n7️⃣ Feature Engineering...")

# 7.1 Neighborhood grouping (CHỈ HỌC TỪ TRAIN)
if 'Neighborhood' in X_train.columns:
    # Tính median SalePrice theo Neighborhood từ train data
    neighborhood_price_medians = pd.Series(y_train_raw, index=X_train.index).groupby(X_train['Neighborhood']).median()
    
    # Tạo bins dựa trên train data
    neighborhood_bins = pd.qcut(neighborhood_price_medians, q=5, labels=['Low', 'Low_Mid', 'Mid', 'Mid_High', 'High'])
    neighborhood_mapping = neighborhood_bins.to_dict()
    
    # Apply cho tất cả sets
    X_train['Neighborhood_Group'] = X_train['Neighborhood'].map(neighborhood_mapping)
    X_val['Neighborhood_Group'] = X_val['Neighborhood'].map(neighborhood_mapping)
    X_test['Neighborhood_Group'] = X_test['Neighborhood'].map(neighborhood_mapping)
    
    # Fill missing với 'Mid' cho những neighborhood không có trong train
    X_val['Neighborhood_Group'] = X_val['Neighborhood_Group'].fillna('Mid')
    X_test['Neighborhood_Group'] = X_test['Neighborhood_Group'].fillna('Mid')

# 7.2 Tạo features mới
print("   ├── Tạo features mới...")
for X_set in [X_train, X_val, X_test]:
    if 'YearBuilt' in X_set.columns:
        X_set['House_Age'] = 2025 - X_set['YearBuilt']
    if 'YearRemodAdd' in X_set.columns:
        X_set['Remod_Age'] = 2025 - X_set['YearRemodAdd']
    
    # Total bathrooms
    bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    if all(col in X_set.columns for col in bath_cols):
        X_set['Total_Bathrooms'] = (X_set['FullBath'] + 0.5 * X_set['HalfBath'] + 
                                   X_set['BsmtFullBath'] + 0.5 * X_set['BsmtHalfBath'])
    
    # Total porch
    porch_cols = ['OpenPorchSF', 'EnclosedPorch', 'ScreenPorch']
    if all(col in X_set.columns for col in porch_cols):
        X_set['Total_Porch_SF'] = X_set[porch_cols].sum(axis=1)
    
    # Binary features
    if 'GarageArea' in X_set.columns:
        X_set['HasGarage'] = (X_set['GarageArea'] > 0).astype(int)
    if 'TotalBsmtSF' in X_set.columns:
        X_set['HasBasement'] = (X_set['TotalBsmtSF'] > 0).astype(int)

# 8. ENCODING (CHỈ HỌC TỪ TRAIN)
print("\n8️⃣ Encoding categorical variables...")

# Tìm object columns
object_cols = X_train.select_dtypes(include=['object']).columns.tolist()
print(f"   ├── Object columns: {len(object_cols)}")

# Label encoding cho từng column
label_encoders = {}
for col in object_cols:
    le = LabelEncoder()
    
    # Fit chỉ trên train data
    le.fit(X_train[col].astype(str))
    
    # Transform tất cả sets
    X_train[col + '_encoded'] = le.transform(X_train[col].astype(str))
    
    # Xử lý unseen categories cho val/test
    def safe_transform(series, encoder):
        result = []
        for val in series.astype(str):
            if val in encoder.classes_:
                result.append(encoder.transform([val])[0])
            else:
                # Assign to most frequent class hoặc -1
                result.append(0)  # hoặc encoder.transform([encoder.classes_[0]])[0]
        return np.array(result)
    
    X_val[col + '_encoded'] = safe_transform(X_val[col], le)
    X_test[col + '_encoded'] = safe_transform(X_test[col], le)
    
    label_encoders[col] = le

# Drop original object columns
X_train = X_train.drop(columns=object_cols)
X_val = X_val.drop(columns=object_cols)
X_test = X_test.drop(columns=object_cols)

# One-hot encode Neighborhood_Group nếu có
if 'Neighborhood_Group' in X_train.columns:
    # Get all possible categories từ train
    train_categories = X_train['Neighborhood_Group'].unique()
    
    for X_set in [X_train, X_val, X_test]:
        X_set_dummies = pd.get_dummies(X_set['Neighborhood_Group'], prefix='NeighGroup')
        
        # Ensure all sets have same columns
        for cat in train_categories:
            col_name = f'NeighGroup_{cat}'
            if col_name not in X_set_dummies.columns:
                X_set_dummies[col_name] = 0
        
        # Drop first category and original column
        X_set_dummies = X_set_dummies.drop(columns=[f'NeighGroup_{train_categories[0]}'])
        X_set = X_set.drop(columns=['Neighborhood_Group'])
        X_set = pd.concat([X_set, X_set_dummies], axis=1)
        
        # Update the sets
        if X_set is X_train:
            X_train = X_set
        elif X_set is X_val:
            X_val = X_set
        else:
            X_test = X_set

print(f"   ├── Final feature count: {X_train.shape[1]}")

# 9. TARGET TRANSFORMATION (RIÊNG BIỆT CHO TỪNG SET)
print("\n9️⃣ Target transformation...")

y_train_log = np.log1p(y_train_raw)
y_val_log = np.log1p(y_val_raw)
y_test_log = np.log1p(y_test_raw)

print(f"   ├── Train target log: {y_train_log.min():.3f} to {y_train_log.max():.3f}")
print(f"   ├── Val target log: {y_val_log.min():.3f} to {y_val_log.max():.3f}")
print(f"   ├── Test target log: {y_test_log.min():.3f} to {y_test_log.max():.3f}")

# 10. FEATURE SCALING (CHỈ HỌC TỪ TRAIN)
print("\n🔟 Feature Scaling...")

scaler = StandardScaler()

# Fit chỉ trên train
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"   ├── Scaler fitted trên {X_train.shape[0]} train samples")
print(f"   ├── Train scaled shape: {X_train_scaled.shape}")
print(f"   ├── Val scaled shape: {X_val_scaled.shape}")
print(f"   ├── Test scaled shape: {X_test_scaled.shape}")

# 11. FINAL CHECK
print("\n✅ PREPROCESSING HOÀN THÀNH - KIỂM TRA CUỐI:")
print("=" * 50)
print(f"📊 Data shapes:")
print(f"   ├── X_train_scaled: {X_train_scaled.shape}")
print(f"   ├── X_val_scaled: {X_val_scaled.shape}")
print(f"   ├── X_test_scaled: {X_test_scaled.shape}")
print(f"   ├── y_train_log: {y_train_log.shape}")
print(f"   ├── y_val_log: {y_val_log.shape}")
print(f"   ├── y_test_log: {y_test_log.shape}")

print(f"\n📈 Target distributions (original scale):")
print(f"   ├── Train: ${y_train_raw.mean():,.0f} ± ${y_train_raw.std():,.0f}")
print(f"   ├── Val: ${y_val_raw.mean():,.0f} ± ${y_val_raw.std():,.0f}")
print(f"   ├── Test: ${y_test_raw.mean():,.0f} ± ${y_test_raw.std():,.0f}")

print(f"\n🔍 Scaled features check:")
print(f"   ├── Train mean: {X_train_scaled.mean():.6f}, std: {X_train_scaled.std():.6f}")
print(f"   ├── Val mean: {X_val_scaled.mean():.6f}, std: {X_val_scaled.std():.6f}")
print(f"   ├── Test mean: {X_test_scaled.mean():.6f}, std: {X_test_scaled.std():.6f}")

print("\n🎯 SẴN SÀNG TRAINING MODEL!")
print("✅ Không có data leakage")
print("✅ Tất cả transformations đều học từ train data")
print("✅ Thứ tự preprocessing chính xác")

# =====================================================================
# 📝 CÁCH SỬ DỤNG KẾT QUẢ
# =====================================================================
"""
Bây giờ bạn có thể train model:

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train_log)

# Predictions
y_train_pred_log = model.predict(X_train_scaled)
y_val_pred_log = model.predict(X_val_scaled)

# Convert back to original scale
y_train_pred = np.expm1(y_train_pred_log)
y_val_pred = np.expm1(y_val_pred_log)

# Evaluate
train_rmse = np.sqrt(mean_squared_error(y_train_raw, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val_raw, y_val_pred))

print(f"Train RMSE: ${train_rmse:,.0f}")
print(f"Val RMSE: ${val_rmse:,.0f}")
"""

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train_log)

# Predictions on all sets
y_train_pred_log = model.predict(X_train_scaled)
y_val_pred_log = model.predict(X_val_scaled)
y_test_pred_log = model.predict(X_test_scaled)  # Thêm prediction cho test set

# Convert back to original scale
y_train_pred = np.expm1(y_train_pred_log)
y_val_pred = np.expm1(y_val_pred_log)
y_test_pred = np.expm1(y_test_pred_log)  # Convert test predictions

# Evaluate with RMSE
train_rmse = np.sqrt(mean_squared_error(y_train_raw, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val_raw, y_val_pred))
test_rmse = np.sqrt(mean_squared_error(y_test_raw, y_test_pred))  # Test RMSE

# Evaluate with R² score
train_r2 = r2_score(y_train_raw, y_train_pred)
val_r2 = r2_score(y_val_raw, y_val_pred)
test_r2 = r2_score(y_test_raw, y_test_pred)  # Test R²

# Print results
print("=== Model Performance ===")
print(f"Train RMSE: ${train_rmse:,.0f}")
print(f"Val RMSE:   ${val_rmse:,.0f}")
print(f"Test RMSE:  ${test_rmse:,.0f}")
print()
print(f"Train R²:   {train_r2:.4f}")
print(f"Val R²:     {val_r2:.4f}")
print(f"Test R²:    {test_r2:.4f}")

# Check for overfitting
print("\n=== Overfitting Check ===")
if abs(train_rmse - val_rmse) / train_rmse > 0.1:
    print("⚠️  Potential overfitting detected (RMSE difference > 10%)")
else:
    print("✅ No significant overfitting (RMSE)")

if abs(train_r2 - val_r2) > 0.05:
    print("⚠️  Potential overfitting detected (R² difference > 0.05)")
else:
    print("✅ No significant overfitting (R²)")

# Final model assessment
print(f"\n=== Final Assessment ===")
print(f"Final Test Performance: RMSE = ${test_rmse:,.0f}, R² = {test_r2:.4f}")
if test_r2 > 0.8:
    print("🎯 Excellent model performance!")
elif test_r2 > 0.6:
    print("👍 Good model performance")
elif test_r2 > 0.4:
    print("📊 Moderate model performance")
else:
    print("📉 Model needs improvement")

joblib.dump(model, 'house_price_model.pkl')
print("✅ Model saved to 'house_price_model.pkl'")