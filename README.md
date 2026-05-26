# Time Series Sales Forecasting with XGBoost

Dự báo bán hàng theo chuỗi thời gian sử dụng XGBoost với kỹ thuật feature engineering tiên tiến.

## 📋 Mục Đích

Dự báo doanh số bán hàng cho nhiều cửa hàng bằng cách:
- Tạo các feature dựa trên thời gian và lịch sử
- Sử dụng mô hình XGBoost với tuning hyperparameter
- Đánh giá hiệu năng chi tiết theo từng cửa hàng
- Trực quan hóa dự báo

## 📁 Cấu Trúc Dự Án

```
time_series/
├── main.py                    # Script chính
├── data_loader.py             # Load & chuẩn bị dữ liệu
├── feature_engineering.py     # Tạo features
├── preprocessing.py           # Split train/validation/test
├── model_training.py          # Training model
├── evaluation.py              # Tính metrics
├── visualization.py           # Vẽ biểu đồ
├── data/
│   └── train.csv             # Dữ liệu đầu vào
└── README.md
```

## 🚀 Cách Sử Dụng

### 1. Chuẩn Bị Dữ Liệu
- Đặt file `train.csv` trong thư mục `./data/`
- File phải có cột: `date`, `store`, `sales`

### 2. Cài Đặt Dependencies
```bash
pip install pandas numpy xgboost scikit-learn matplotlib tabulate
```

### 3. Chạy Script
```bash
python main.py
```
### link kaggle:
https://www.kaggle.com/code/haquangdo/time-series/edit/run/321739777

## 📦 Mô Tả Các Module

### `main.py`
Script chính điều phối toàn bộ quy trình từ load data đến trực quan hóa kết quả.

### `data_loader.py`
- `load_and_prepare_data()`: Load CSV và tổng hợp doanh số theo store/ngày

### `feature_engineering.py`
- `create_features()`: Tạo các feature:
  - **Time features**: day_of_week, month, day_of_year, is_weekend
  - **Lag features**: sales_lag_1, sales_lag_7, sales_lag_365
  - **Rolling mean**: 7, 14, 30 ngày
  - **Cyclical feature**: rolling mean theo day of week (4 ngày)
  
- `get_feature_list()`: Trả về danh sách features

### `preprocessing.py`
- `split_data()`: Tách train/validation/test:
  - Train: < 2017-01-01
  - Validation: 2016-10-01 -> 2016-12-31
  - Test: >= 2017-01-01
  
- `prepare_xy()`: Chuẩn bị X, y với log transformation

### `model_training.py`
- `train_model()`: Huấn luyện XGBoost với GridSearch:
  - Hyperparameters: max_depth, learning_rate, subsample, colsample_bytree
  - Cross-validation: TimeSeriesSplit (5 folds)
  - Scoring: MAE

### `evaluation.py`
- `calculate_metrics()`: Tính MAE, MSE, RMSE, MAPE, R²
- `evaluate_overall()`: Metrics tổng thể
- `evaluate_by_store()`: Metrics từng cửa hàng

### `visualization.py`
- `plot_feature_importance()`: Biểu đồ tầm quan trọng features
- `plot_forecast()`: Vẽ 10 cửa hàng so sánh actual vs predicted

## 📊 Kết Quả Đầu Ra

Script sẽ in ra:
1. **Overall metrics**: MAE, MSE, RMSE, MAPE, R² cho train/validation/test
2. **Store-wise metrics**: Metrics chi tiết cho từng cửa hàng
3. **Feature importance**: Xếp hạng tầm quan trọng features
4. **Visualizations**: 
   - Biểu đồ feature importance
   - Subplots 5x2 so sánh dự báo vs thực tế cho 10 cửa hàng

## 🔧 Tuning & Tùy Chỉnh

### Thay đổi hyperparameters
Chỉnh sửa trong `model_training.py`:
```python
param_grid = {
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.3],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9]
}
```

### Thay đổi ngày split
Chỉnh sửa trong `preprocessing.py`:
```python
train_data = df[df["date"] < "2017-01-01"].copy()
valid_final = train_data[train_data["date"] >= "2016-10-01"].copy()
```

## 💡 Features Được Sử Dụng

| Feature | Loại | Mô Tả |
|---------|------|-------|
| store | Categorical | ID cửa hàng |
| day_of_week | Time | Thứ trong tuần (0-6) |
| month | Time | Tháng |
| day_of_year | Time | Ngày trong năm |
| is_weekend | Time | Là weekend (0/1) |
| sales_lag_1 | Lag | Doanh số hôm trước |
| sales_lag_7 | Lag | Doanh số 7 ngày trước |
| sales_lag_365 | Lag | Doanh số 1 năm trước |
| rolling_mean_7 | Rolling | Trung bình 7 ngày |
| rolling_mean_14 | Rolling | Trung bình 14 ngày |
| rolling_mean_30 | Rolling | Trung bình 30 ngày |
| rolling_mean_day_of_week_4 | Cyclical | Trung bình theo thứ (4 tuần) |

## 📈 Metrics

- **MAE** (Mean Absolute Error): Sai số tuyệt đối trung bình
- **MSE** (Mean Squared Error): Sai số bình phương trung bình
- **RMSE** (Root Mean Squared Error): Căn bậc hai MSE
- **MAPE** (Mean Absolute Percentage Error): Sai số phần trăm tuyệt đối
- **R²** (R-squared): Hệ số xác định

## 🎯 Tips Sử Dụng

1. Đảm bảo dữ liệu có đủ lịch sử (ít nhất 1 năm) để lag features hoạt động tốt
2. Nếu có GPU, set `device="cuda"` trong XGBoost để tăng tốc
3. Thay đổi `n_jobs` tùy theo số core CPU có sẵn
4. Validation set nên có kích thước đủ lớn (3-6 tháng dữ liệu)

## 📝 Yêu Cầu

- Python 3.7+
- pandas >= 1.0
- numpy >= 1.18
- xgboost >= 1.0
- scikit-learn >= 0.24
- matplotlib >= 3.0
- tabulate >= 0.8