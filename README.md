# ICF Detector

ICF Detector là dự án Học máy nhằm phát hiện các kênh YouTube **AI Slop** – những kênh sử dụng nội dung tự động, clickbait, hoặc kém chất lượng – dựa trên metadata và đặc trưng văn bản.

## Mục tiêu

Xây dựng pipeline hoàn chỉnh từ thu thập dữ liệu, trích xuất đặc trưng, huấn luyện mô hình, đến dự đoán. Hiện tại mô hình Random Forest đạt **F1 ≈ 0.89** trên tập test.

## Dữ liệu

- **File đặc trưng chính**: `features_final.csv` (chia sẻ qua Google Drive, luôn cập nhật phiên bản mới nhất).  
- **Số lượng**: 835 kênh (359 AI Slop, 476 Genuine).  
- **13 đặc trưng tĩnh** cho mỗi kênh, thuộc 4 nhóm:
  - *Chuỗi thời gian & vận tốc đăng bài*: `time_interval_std`(Độ lệch chuẩn của khoảng thời gian tải video), `upload_burst_ratio` (Tỷ lệ đăng bài dồn dập), `video_upload_frequency` (Tần suất tải video), `view_per_video`  (Lượt xem trên mỗi video)
  - *Dấu vết văn bản AI*: `dash_density` (Mật độ dấu gạch ngang), `title_length_std` (Độ lệch chuẩn của độ dài tiêu đề), `capitalization_ratio` (Tỷ lệ viết hoa), `opening_repeat_ratio` (Tỷ lệ lặp từ mở đầu), `temporal_clickbait_ratio` (Tỷ lệ giật gân theo thời gian)
  - *Độ đa dạng & tương đồng nội dung*: `type_token_ratio` (Tỷ lệ đa dạng từ vựng), `avg_title_similarity` (Độ tương đồng tiêu đề trung bình)
  - *Chỉ số tương tác*: `sub_to_view_ratio` (Tỷ lệ đăng ký trên lượt xem), `subscriber_velocity` (Tốc độ tăng trưởng người đăng ký)
- **Nhãn**: `label` (0 = Genuine, 1 = AI Slop).  
- **Ngôn ngữ**: Cả tiếng Anh và tiếng Việt, đã được tokenizer riêng (Underthesea cho tiếng Việt) xử lý trong pipeline.

🔗 **Link dữ liệu (Google Drive – tự động cập nhật)**:  
`https://drive.google.com/uc?id=<YOUR_FILE_ID>` (ID được cung cấp riêng) 

## Cấu trúc repository

```text
ICF_Detector/
├── main.py # Entry point (nếu có)
├── requirements.txt
├── src/
│ ├── config.py # Cấu hình trung tâm (API key, paths, tham số)
│ ├── crawl.py # Thu thập dữ liệu kênh YouTube
│ ├── crawl_expand.py # Thu thập bổ sung cho kênh mới
│ ├── crawl_auto_expand.py # Thu thập tự động kênh mới (dựa trên pipeline)
│ ├── features.py # Trích xuất đặc trưng
│ ├── train.py # Huấn luyện Random Forest (pipeline an toàn)
│ ├── predict.py # Dự đoán nhanh một kênh bất kỳ
│ ├── expand.py # Tự động mở rộng dataset dựa trên mô hình
│ ├── compare_models.py # So sánh nhiều mô hình (baseline)
│ ├── anomaly.py # Tính anomaly score bằng Isolation Forest
│ └── utils.py # Các hàm tiện ích
├── data/
│ ├── raw/ # File seed (URL kênh)
│ ├── collected/ # Kênh đã crawl (CSV)
│ └── processed/ # Đặc trưng, intervals, so sánh mô hình
├── models/ # Mô hình đã lưu (random_forest.pkl)
└── notebooks/ # Notebook EDA và trực quan hóa
```

## Bắt đầu nhanh

1. **Clone repo**:
   ```bash
   git clone https://github.com/tinthu02/ICF_Detector.git
   cd ICF_Detector
   ```

2. **Tạo virtual environment & cài đặt**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Chuẩn bị dữ liệu**:
   * Tải file `features_final.csv` từ link Drive trên.
   * Đặt file vừa tải vào thư mục `data/processed/`.

4. **Huấn luyện mô hình**:
   ```bash
   python src/train.py
   ```

5. **Dự đoán kênh mới**:
   ```bash
   python src/predict.py "https://www.youtube.com/@ExampleChannel"
   ```
