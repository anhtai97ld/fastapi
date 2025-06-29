# House Price Prediction FastAPI

## Yêu cầu hệ thống

- Python 3.8+
- pip

## Cài đặt

1. **Clone repository**  
   ```sh
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Cài đặt các thư viện cần thiết**  
   ```sh
   pip install -r requirements.txt
   ```

3. **Kiểm tra/cập nhật dữ liệu**  
   - Đảm bảo file `content/train.csv` đã có dữ liệu.

4. **Huấn luyện lại mô hình (nếu muốn)**
   ```sh
   python app/train.py
   ```
   - Mô hình sẽ được lưu thành file `house_price_model.pkl`.

5. **Chạy API server**
   ```sh
   uvicorn app.main:app --reload
   ```

6. **Dự đoán giá nhà**
   - Gửi file CSV chứa dữ liệu nhà lên endpoint `/predict` bằng Postman hoặc curl:
   ```sh
   curl -X POST "http://127.0.0.1:8000/predict" -F "file=@your_input.csv"
   ```

## Cấu trúc thư mục

- `app/`: Chứa mã nguồn FastAPI và các script xử lý
- `content/`: Chứa dữ liệu huấn luyện
- `house_price_model.pkl`: File mô hình đã huấn luyện
- `requirements.txt`: Danh sách thư viện Python

## Ghi chú

- Nếu chỉ muốn chạy thử API, bạn có thể dùng sẵn file mô hình `house_price_model.pkl` mà không cần train lại.
- Đảm bảo tên cột trong file CSV đầu vào giống với dữ liệu huấn