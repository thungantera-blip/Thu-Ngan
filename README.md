# Ứng Dụng Tối Ưu Hóa Danh Mục Đầu Tư (LSTM-GRU Hybrid Model)

Ứng dụng web được phát triển bằng Streamlit giúp tối ưu hóa tỷ trọng danh mục đầu tư các cổ phiếu Việt Nam (dữ liệu được lấy trực tiếp qua thư viện `vnstock`). Ứng dụng sử dụng mô hình học sâu lai (Hybrid LSTM-GRU) với hàm mất mát Sharpe tùy chỉnh để trực tiếp tìm ra tỷ trọng cổ phiếu tối ưu nhằm đạt được hệ số Sharpe cao nhất.

---

## Các Tính Năng Chính
- **Lấy dữ liệu tự động**: Tải dữ liệu lịch sử giá cổ phiếu theo ngành (Thép, Ngân hàng, Bất động sản, Bán lẻ, Công nghệ...) từ nguồn dữ liệu KBS thông qua thư viện `vnstock`.
- **Lọc mã cổ phiếu tối ưu**: Tính toán hệ số Sharpe lịch sử của toàn bộ các mã trong ngành và lựa chọn Top cổ phiếu tốt nhất để làm sạch dữ liệu và huấn luyện.
- **Mô hình học sâu LSTM-GRU**: 
  - Huấn luyện trực tiếp trên chuỗi đặc trưng (Returns, Volatility, MA Ratios, RSI).
  - Tối ưu hóa hàm lỗi Sharpe Loss (kèm entropy để đa dạng hóa danh mục) để xuất ra phân bổ tỷ trọng tối ưu.
- **Đánh giá & So sánh Backtest**: So sánh hiệu quả thực tế (Lợi nhuận, độ lệch chuẩn, Sharpe) trên tập kiểm thử (Test Set) giữa 3 chiến lược:
  1. Phân bổ Động LSTM-GRU (Dynamic Portfolio)
  2. Phân bổ Đều (Equal Weight Portfolio)
  3. Phân bổ 80-20 dựa trên thứ hạng Sharpe.
- **Giao diện Trực quan & Premium**: Thiết kế giao diện hiện đại với các tab điều hướng, biểu đồ Plotly tương tác cao, và các thẻ chỉ số KPI sinh động.

---

## Hướng Dẫn Chạy Ứng Dụng Dưới Local

### 1. Chuẩn bị Môi trường
Ứng dụng yêu cầu Python phiên bản từ **3.9 đến 3.11** (khuyên dùng Python 3.10 hoặc 3.11).

Cài đặt các thư viện cần thiết thông qua file `requirements.txt`:
```bash
pip install -r requirements.txt
```

*Lưu ý:* Để quá trình cài đặt diễn ra nhanh và tránh lỗi bộ nhớ, chúng tôi sử dụng gói `tensorflow-cpu` (không cần hỗ trợ GPU).

### 2. Khởi chạy Ứng dụng
Chạy lệnh Streamlit để khởi động ứng dụng:
```bash
streamlit run app.py
```
Sau khi chạy thành công, trình duyệt sẽ tự động mở trang dashboard tại địa chỉ mặc định `http://localhost:8501`.

---

## Hướng Dẫn Deploy Lên Streamlit Cloud

### Bước 1: Đưa Code Lên GitHub
1. Tạo một repository mới trên GitHub (ví dụ: `vietnam-portfolio-optimizer`).
2. Khởi tạo Git trong thư mục chứa code và đẩy code lên GitHub:
```bash
git init
git add .
git commit -m "Initial commit - Portfolio Optimizer App"
git branch -M main
git remote add origin https://github.com/USERNAME/vietnam-portfolio-optimizer.git
git push -u origin main
```

### Bước 2: Deploy Trên Streamlit Share (Streamlit Cloud)
1. Truy cập trang web [Streamlit Community Cloud](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấn vào nút **New App**.
3. Chọn repository của bạn (`vietnam-portfolio-optimizer`), chọn branch (`main`), và chỉ định file chạy chính là `app.py`.
4. Nhấn vào **Deploy!**. Streamlit Cloud sẽ tự động đọc file `requirements.txt`, cài đặt các thư viện (bao gồm tensorflow-cpu, vnstock, plotly, scikit-learn), và khởi chạy ứng dụng của bạn chỉ trong vài phút.

---

## Cấu Trúc Thư Mục Dự Án
```
├── app.py                # File ứng dụng Streamlit chính
├── industry_tickers.py   # Danh sách mã cổ phiếu phân nhóm theo ngành Việt Nam
├── requirements.txt      # Định nghĩa các thư viện Python cần cài đặt
└── README.md             # Tài liệu hướng dẫn sử dụng và triển khai này
```

## Các Tùy Chỉnh Cấu Hình Trên Ứng Dụng
- **Chọn Ngành**: Chọn một hoặc nhiều ngành để tạo rổ cổ phiếu (Ví dụ: Thép, Ngân hàng, Bất động sản...).
- **Chọn Mã Cụ Thể**: Mặc định sẽ tải toàn bộ mã thuộc ngành đã chọn. Bạn có thể chọn chế độ "Tùy chỉnh" để chọn thủ công các mã mong muốn (giúp tăng tốc tải dữ liệu).
- **Tham số Huấn luyện**: Tùy chỉnh số lượng Epochs (khuyên dùng 10-20 để kiểm tra nhanh dưới dạng web, 100 để có kết quả tối ưu nhất), Batch size, Kích thước cửa sổ trượt (Window size), Tỷ suất phi rủi ro năm (Risk-free rate).
