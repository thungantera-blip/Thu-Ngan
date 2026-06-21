import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime
import time
import os
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Portfolio Optimization using LSTM-GRU",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom css for premium aesthetics
st.markdown("""
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, .main-title {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #636EFA, #EF553B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #888888;
        margin-bottom: 2rem;
    }
    
    /* KPI Card styling */
    .kpi-card {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #334155;
        text-align: center;
        margin-bottom: 15px;
    }
    
    .kpi-title {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-bottom: 8px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    
    .kpi-strategy {
        font-size: 0.75rem;
        color: #38BDF8;
        margin-top: 5px;
    }
    
    /* Styled tab headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Import INDUSTRY_TICKERS
try:
    from data.industry_tickers import INDUSTRY_TICKERS
except ModuleNotFoundError:
    try:
        from industry_tickers import INDUSTRY_TICKERS
    except ModuleNotFoundError:
        st.error("Không tìm thấy file `industry_tickers.py`. Vui lòng đảm bảo file này có mặt trong thư mục dự án.")
        st.stop()

# Set random seeds helper
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Streamlit Keras callback to show training progress
class StreamlitTrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, epochs, progress_bar, status_text):
        super().__init__()
        self.epochs = epochs
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        self.losses.append(loss)
        self.val_losses.append(val_loss)
        percent = (epoch + 1) / self.epochs
        self.progress_bar.progress(percent)
        self.status_text.text(f"Đang huấn luyện: Epoch {epoch+1}/{self.epochs} - Loss: {loss:.4f} - Val Loss: {val_loss:.4f}")

# Caching downloader using vnstock
@st.cache_data(show_spinner=False)
def load_all_stocks_data(tickers, start_date, end_date):
    from vnstock import Vnstock
    all_dfs = []
    failed_symbols = []
    
    # We download data with progress reporting inside the app, so we just download here
    # To prevent rate limit, we will let app.py handle printing and progress bars, but this function is cached.
    # Note: st.progress cannot be called from cached function, so we do simple loop and returns list of dataframes
    for ticker in tickers:
        success = False
        retry = 0
        while not success and retry < 3:
            try:
                stock = Vnstock().stock(symbol=ticker, source="KBS")
                df = stock.quote.history(start=start_date, end=end_date, interval="1D")
                if df is not None and not df.empty:
                    df = df.copy()
                    df["ticker"] = ticker
                    if "time" not in df.columns:
                        if "date" in df.columns:
                            df["time"] = df["date"]
                        elif "datetime" in df.columns:
                            df["time"] = df["datetime"]
                    
                    keep_cols = [c for c in ["time", "open", "high", "low", "close", "volume", "ticker"] if c in df.columns]
                    df = df[keep_cols]
                    all_dfs.append(df)
                    success = True
                else:
                    failed_symbols.append(ticker)
                    success = True # Stop retrying
            except Exception as e:
                retry += 1
                time.sleep(2)
        if not success:
            failed_symbols.append(ticker)
            
    return all_dfs, failed_symbols

# ----------------- SIDEBAR CONFIGURATIONS -----------------
st.sidebar.markdown("## ⚙️ Cấu Hình Mô Hình & Dữ Liệu")

# Sector Selectbox
all_sectors = list(INDUSTRY_TICKERS.keys())
selected_sector = st.sidebar.selectbox("Chọn ngành phân tích", all_sectors, index=all_sectors.index("Thép") if "Thép" in all_sectors else 0)

# Stock pool customization
tickers_in_sector = INDUSTRY_TICKERS[selected_sector]
st.sidebar.markdown(f"Ngành **{selected_sector}** có {len(tickers_in_sector)} mã.")

stock_select_mode = st.sidebar.radio("Chế độ chọn mã", ["Tất cả các mã trong ngành", "Tự chọn danh sách mã"], index=0)

if stock_select_mode == "Tự chọn danh sách mã":
    selected_tickers = st.sidebar.multiselect("Chọn mã trong ngành", tickers_in_sector, default=tickers_in_sector[:min(10, len(tickers_in_sector))])
else:
    selected_tickers = tickers_in_sector

if not selected_tickers:
    st.sidebar.error("Vui lòng chọn ít nhất 1 mã cổ phiếu!")
    st.stop()

# Dates configurations
st.sidebar.markdown("### 📅 Thời gian phân tích")
train_start = st.sidebar.date_input("Bắt đầu huấn luyện", datetime.date(2015, 1, 1))
train_end = st.sidebar.date_input("Kết thúc huấn luyện", datetime.date(2024, 12, 31))
test_start = st.sidebar.date_input("Bắt đầu kiểm thử (Backtest)", datetime.date(2025, 1, 1))
test_end = st.sidebar.date_input("Kết thúc kiểm thử (Backtest)", datetime.date(2025, 12, 31))

# Hyperparameters
st.sidebar.markdown("### 🧠 Siêu tham số Mô hình")
rf_annual = st.sidebar.slider("Tỷ suất phi rủi ro năm (Risk-free)", 0.0, 0.10, 0.045, step=0.005)
trading_days = st.sidebar.number_input("Số ngày giao dịch/năm", value=252)
epochs = st.sidebar.slider("Số lượng Epochs huấn luyện", 5, 150, 20, step=5, help="Nên dùng 20-30 epochs khi chạy trên web để huấn luyện nhanh hơn. Chạy 100 epochs để tối ưu tối đa.")
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
window_size = st.sidebar.slider("Độ dài chuỗi quan sát (Window size)", 10, 60, 30)
horizon = st.sidebar.slider("Thời gian tối ưu phía trước (Horizon)", 1, 10, 5)
lambda_entropy = st.sidebar.slider("Hệ số Entropy (Đa dạng hóa danh mục)", 0.0, 0.1, 0.01, step=0.005)
lstm_units = st.sidebar.slider("LSTM Units", 16, 128, 96, step=16)
gru_units = st.sidebar.slider("GRU Units", 16, 128, 48, step=16)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.0001, 0.0005, 0.001, 0.005], index=1)

# Main Page Title
st.markdown('<div class="main-title">Tối Ưu Hóa Danh Mục Đầu Tư Bằng LSTM-GRU</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ứng dụng học máy sâu lai để phân bổ tỷ trọng tối ưu cho thị trường chứng khoán Việt Nam</div>', unsafe_allow_html=True)

# ----------------- MAIN APP TABS -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 1. Tổng quan & Tải Dữ liệu", 
    "🎯 2. Tính Sharpe & Chọn mã", 
    "🧠 3. Huấn luyện & Tối ưu hóa", 
    "📈 4. Backtest & Đánh giá"
])

# Global state dictionary
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'top_symbols' not in st.session_state:
    st.session_state.top_symbols = []
if 'returns_df' not in st.session_state:
    st.session_state.returns_df = None
if 'pivot_df' not in st.session_state:
    st.session_state.pivot_df = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'best_weights' not in st.session_state:
    st.session_state.best_weights = None
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None

# Tab 1: Overview & Data Selection
with tab1:
    st.markdown("""
    ### Phương pháp luận
    Mô hình học sâu lai **LSTM-GRU** tận dụng ưu thế của cả hai cấu trúc:
    - **LSTM (Long Short-Term Memory)**: Lưu giữ tốt thông tin chuỗi thời gian dài hạn và các biến động phức tạp của thị trường.
    - **GRU (Gated Recurrent Unit)**: Có kiến trúc gọn nhẹ hơn, huấn luyện nhanh hơn và bắt tốt các xu thế ngắn hạn.
    
    Thay vì dự báo giá cổ phiếu rồi tính toán danh mục qua lý thuyết Markowitz truyền thống (thường bị sai lệch do nhiễu dự báo), mô hình này sử dụng **Sharpe Loss** làm hàm tối ưu trực tiếp. Đầu ra của mạng nơ-ron đi qua lớp kích hoạt *Softmax* để trực tiếp cho ra tỷ trọng phân bổ của các cổ phiếu sao cho hệ số Sharpe trên tập huấn luyện là cao nhất.
    """)
    
    st.markdown("---")
    st.subheader(f"Tải dữ liệu ngành: {selected_sector}")
    
    # Download Button
    if st.button("Tải dữ liệu từ KBS API", key="btn_download"):
        with st.spinner("Đang kết nối API và tải dữ liệu lịch sử các mã cổ phiếu..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_dfs = []
            failed_symbols = []
            
            total_tickers = len(selected_tickers)
            for i, ticker in enumerate(selected_tickers):
                status_text.text(f"Đang tải ({i+1}/{total_tickers}): {ticker}")
                
                # Fetch data via vnstock
                ticker_dfs, ticker_failed = load_all_stocks_data([ticker], str(train_start), str(test_end))
                all_dfs.extend(ticker_dfs)
                failed_symbols.extend(ticker_failed)
                
                # Update progress
                progress_bar.progress((i + 1) / total_tickers)
                time.sleep(1.0) # Safe sleep interval to prevent KBS rate limit blocking
                
            progress_bar.empty()
            status_text.empty()
            
            if all_dfs:
                raw_data = pd.concat(all_dfs, ignore_index=True)
                st.session_state.raw_data = raw_data
                st.session_state.failed_symbols = failed_symbols
                st.session_state.data_loaded = True
                st.success(f"Tải dữ liệu thành công! Shape: {raw_data.shape}. Số mã lỗi/không có dữ liệu: {len(failed_symbols)}")
                if failed_symbols:
                    st.warning(f"Mã lỗi: {failed_symbols}")
            else:
                st.error("Không tải được dữ liệu cho bất kỳ mã nào. Vui lòng kiểm tra kết nối mạng hoặc thử lại sau.")
                
    if st.session_state.data_loaded:
        raw_data = st.session_state.raw_data
        
        # Create close price pivot table
        pivot_df = raw_data.pivot_table(
            index="time",
            columns="ticker",
            values="close",
            aggfunc="last"
        ).sort_index()
        pivot_df.index = pd.to_datetime(pivot_df.index)
        st.session_state.pivot_df = pivot_df
        
        # Display data sample
        st.markdown("#### Bảng giá đóng cửa (Close Prices - Mẫu 5 dòng đầu)")
        st.dataframe(pivot_df.head())
        
        # Plotly chart of Close Prices
        st.markdown("#### Biểu đồ diễn biến giá đóng cửa")
        fig_price = go.Figure()
        # Limit to first 10 columns to keep chart readable
        display_cols = pivot_df.columns[:10]
        for col in display_cols:
            fig_price.add_trace(go.Scatter(x=pivot_df.index, y=pivot_df[col], name=col, mode='lines'))
        
        fig_price.update_layout(
            template="plotly_dark",
            xaxis_title="Thời gian",
            yaxis_title="Giá đóng cửa (VNĐ)",
            legend_title="Mã CK",
            margin=dict(l=20, r=20, t=30, b=20),
            height=500
        )
        st.plotly_chart(fig_price, use_container_width=True)
        if len(pivot_df.columns) > 10:
            st.info(f"Lưu ý: Biểu đồ trên chỉ hiển thị 10 mã đầu tiên trong số {len(pivot_df.columns)} mã để đảm bảo tính dễ nhìn.")

# Tab 2: Sharpe Calculation & Stock Selection
with tab2:
    if not st.session_state.data_loaded:
        st.warning("Vui lòng tải dữ liệu ở Tab 1 trước!")
    else:
        st.subheader("Phân tích hệ số Sharpe & Chọn lọc cổ phiếu ưu tú")
        pivot_df = st.session_state.pivot_df
        
        # Preprocessing: Fill prices first
        price_filled = pivot_df.ffill().bfill()
        
        # Calculate daily returns
        returns_df_all = price_filled.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")
        
        # Calculate annualized Sharpe ratio
        rf_daily = rf_annual / trading_days
        mean_ret = returns_df_all.mean()
        std_ret = returns_df_all.std().replace(0, np.nan)
        
        sharpe_ratio = ((mean_ret - rf_daily) / std_ret).dropna()
        sharpe_annual = sharpe_ratio * np.sqrt(trading_days)
        sharpe_annual_sorted = sharpe_annual.sort_values(ascending=False)
        
        # Display Sharpe ratios
        st.markdown("#### Hệ số Sharpe năm của các cổ phiếu trong ngành (Sắp xếp giảm dần)")
        
        fig_sharpe = px.bar(
            x=sharpe_annual_sorted.index,
            y=sharpe_annual_sorted.values,
            labels={'x': 'Mã Cổ Phiếu', 'y': 'Hệ số Sharpe Năm'},
            title="Xếp hạng Sharpe các cổ phiếu",
            template="plotly_dark"
        )
        fig_sharpe.update_traces(marker_color='#636EFA')
        st.plotly_chart(fig_sharpe, use_container_width=True)
        
        # Selecting top stocks
        max_stocks = min(10, len(sharpe_annual_sorted))
        top_n_input = st.number_input("Số lượng cổ phiếu ưu tú chọn huấn luyện (Top N)", min_value=2, max_value=len(sharpe_annual_sorted), value=max_stocks)
        
        top_symbols = sharpe_annual_sorted.head(top_n_input).index.tolist()
        st.session_state.top_symbols = top_symbols
        st.session_state.returns_df = returns_df_all
        
        st.success(f"Đã chọn Top {top_n_input} cổ phiếu tốt nhất: **{top_symbols}**")
        
        # Display sample features for top stocks
        st.markdown("#### Đặc trưng kỹ thuật bổ sung (Lợi nhuận, Độ biến động, RSI, MA Ratios...)")
        st.write("Các đặc trưng này sẽ được trích xuất trực tiếp từ chuỗi giá đóng cửa để đưa vào huấn luyện mô hình LSTM-GRU.")

# Tab 3: Model Training & Optimization
with tab3:
    if not st.session_state.data_loaded or not st.session_state.top_symbols:
        st.warning("Vui lòng tải dữ liệu và chọn cổ phiếu ở Tab 1 & Tab 2 trước!")
    else:
        st.subheader("Huấn luyện mô hình Deep Learning & Tối ưu tỷ trọng")
        
        top_symbols = st.session_state.top_symbols
        pivot_df = st.session_state.pivot_df
        
        # Filter top symbols
        price_top = pivot_df[top_symbols].copy()
        
        # Split Train/Test based on dates
        train_prices = price_top.loc[str(train_start):str(train_end)].copy()
        test_prices = price_top.loc[str(test_start):str(test_end)].copy()
        
        if train_prices.empty or test_prices.empty:
            st.error("Thời gian chia tập Train/Test không hợp lệ hoặc dữ liệu trống. Vui lòng kiểm tra lại cấu hình ngày ở sidebar.")
            st.stop()
            
        train_prices = train_prices.sort_index().ffill().bfill()
        test_prices = test_prices.sort_index().ffill().bfill()
        
        # Returns
        train_returns = train_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")
        test_returns = test_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")
        
        # Helper compute RSI
        def compute_rsi(price_df, period=14):
            delta = price_df.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(period, min_periods=period).mean()
            avg_loss = loss.rolling(period, min_periods=period).mean()
            rs = avg_gain / (avg_loss + 1e-9)
            rsi = 100 - (100 / (1 + rs))
            return rsi
            
        # Build features
        def build_features(price_df, return_df):
            common_idx = price_df.index.intersection(return_df.index)
            price_df = price_df.loc[common_idx].copy()
            return_df = return_df.loc[common_idx].copy()
            
            price_df = price_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
            return_df = return_df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            feat_list = []
            
            # Ret 1
            ret_1 = return_df.copy()
            ret_1.columns = [f"{c}_ret1" for c in ret_1.columns]
            feat_list.append(ret_1)
            
            # Ret 5
            ret_5 = price_df.pct_change(5)
            ret_5.columns = [f"{c}_ret5" for c in ret_5.columns]
            feat_list.append(ret_5)
            
            # Ret 10
            ret_10 = price_df.pct_change(10)
            ret_10.columns = [f"{c}_ret10" for c in ret_10.columns]
            feat_list.append(ret_10)
            
            # MA 5 ratio
            ma5_ratio = price_df / (price_df.rolling(5, min_periods=5).mean() + 1e-9) - 1
            ma5_ratio.columns = [f"{c}_ma5_ratio" for c in ma5_ratio.columns]
            feat_list.append(ma5_ratio)
            
            # MA 10 ratio
            ma10_ratio = price_df / (price_df.rolling(10, min_periods=10).mean() + 1e-9) - 1
            ma10_ratio.columns = [f"{c}_ma10_ratio" for c in ma10_ratio.columns]
            feat_list.append(ma10_ratio)
            
            # Vol 5
            vol5 = return_df.rolling(5, min_periods=5).std()
            vol5.columns = [f"{c}_vol5" for c in vol5.columns]
            feat_list.append(vol5)
            
            # Vol 10
            vol10 = return_df.rolling(10, min_periods=10).std()
            vol10.columns = [f"{c}_vol10" for c in vol10.columns]
            feat_list.append(vol10)
            
            # Mom 5
            mom5 = price_df.pct_change(5)
            mom5.columns = [f"{c}_mom5" for c in mom5.columns]
            feat_list.append(mom5)
            
            # RSI 14
            rsi14 = compute_rsi(price_df, period=14) / 100.0
            rsi14.columns = [f"{c}_rsi14" for c in rsi14.columns]
            feat_list.append(rsi14)
            
            features = pd.concat(feat_list, axis=1).replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
            return features
            
        train_features = build_features(train_prices, train_returns)
        test_features = build_features(test_prices, test_returns)
        
        # Scaling
        scaler = StandardScaler()
        train_features_scaled = pd.DataFrame(scaler.fit_transform(train_features), index=train_features.index, columns=train_features.columns)
        test_features_scaled = pd.DataFrame(scaler.transform(test_features), index=test_features.index, columns=test_features.columns)
        
        train_target_returns = train_returns.loc[train_features_scaled.index].copy()
        test_target_returns = test_returns.loc[test_features_scaled.index].copy()
        
        # Create sequences and targets
        def create_sequences_and_targets(features_df, target_returns_df, w_size, hor=5):
            X, y, dates = [], [], []
            feat_values = features_df.values.astype(np.float32)
            target_values = target_returns_df.values.astype(np.float32)
            idx = features_df.index
            
            for i in range(len(features_df) - w_size - hor + 1):
                X.append(feat_values[i:i + w_size])
                y.append(target_values[i + w_size:i + w_size + hor].mean(axis=0))
                dates.append(idx[i + w_size + hor - 1])
                
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), pd.Index(dates)
            
        X_train, y_train_target, train_seq_dates = create_sequences_and_targets(train_features_scaled, train_target_returns, window_size, horizon)
        X_test, y_test_target, test_seq_dates = create_sequences_and_targets(test_features_scaled, test_target_returns, window_size, horizon)
        
        st.write(f"Kích thước tập Train (Sequences): {X_train.shape}")
        st.write(f"Kích thước tập Test (Sequences) : {X_test.shape}")
        
        # Trigger Training
        train_btn = st.button("🚀 Bắt đầu huấn luyện mô hình")
        
        if train_btn:
            set_seed(42) # Set seed for reproducibility
            
            # Local Sharpe loss incorporating current user parameters
            def sharpe_loss(y_true, y_pred):
                portfolio_returns = K.sum(y_true * y_pred, axis=1)
                portfolio_returns = portfolio_returns - (rf_annual / trading_days)
                mean_returns = K.mean(portfolio_returns)
                std_returns = K.std(portfolio_returns)
                sharpe = mean_returns / (std_returns + 1e-9)
                entropy = -K.sum(y_pred * K.log(y_pred + 1e-9), axis=1)
                entropy = K.mean(entropy)
                return -sharpe - lambda_entropy * entropy
                
            # Model architecture builder
            model = Sequential([
                Input(shape=(X_train.shape[1], X_train.shape[2])),
                LSTM(lstm_units, return_sequences=True, activation="tanh", recurrent_activation="sigmoid"),
                Dropout(0.2),
                GRU(gru_units, return_sequences=False, activation="tanh", recurrent_activation="sigmoid"),
                Dropout(0.2),
                Dense(64, activation="relu"),
                Dropout(0.1),
                Dense(len(top_symbols), activation="softmax")
            ])
            
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=sharpe_loss)
            
            # Progress bar for training
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            st_callback = StreamlitTrainingCallback(epochs, progress_bar, status_text)
            early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)
            
            # Run fit
            history = model.fit(
                X_train, y_train_target,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=False,
                verbose=0,
                validation_split=0.2,
                callbacks=[st_callback, early_stop, reduce_lr]
            )
            
            progress_bar.empty()
            status_text.success("Huấn luyện thành công!")
            
            # Save results to session state
            st.session_state.model = model
            st.session_state.history = history.history
            st.session_state.model_trained = True
            
            # Make predictions on test set
            pred_weights_test = model.predict(X_test, verbose=0)
            st.session_state.pred_weights_test = pred_weights_test
            st.session_state.test_seq_dates = test_seq_dates
            st.session_state.y_test_target = y_test_target
            st.session_state.test_returns = test_returns
            st.session_state.train_returns = train_returns
            
        if st.session_state.model_trained:
            history_data = st.session_state.history
            pred_weights_test = st.session_state.pred_weights_test
            test_seq_dates = st.session_state.test_seq_dates
            
            # Plot loss curves
            st.markdown("#### Biểu đồ mất mát (Loss Curve)")
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=history_data["loss"], name="Train Loss", mode="lines"))
            fig_loss.add_trace(go.Scatter(y=history_data["val_loss"], name="Val Loss", mode="lines"))
            fig_loss.update_layout(
                template="plotly_dark",
                xaxis_title="Epoch",
                yaxis_title="Loss Value (Negative Sharpe)",
                margin=dict(l=20, r=20, t=30, b=20),
                height=400
            )
            st.plotly_chart(fig_loss, use_container_width=True)
            
            # Calculate Average Weights
            avg_weights = pred_weights_test.mean(axis=0)
            weights_df = pd.DataFrame({
                "Asset": top_symbols,
                "Weight": avg_weights
            }).sort_values("Weight", ascending=False).reset_index(drop=True)
            
            st.session_state.best_weights = weights_df
            
            # Display weights as a Treemap
            st.markdown("#### Phân bổ tỷ trọng tối ưu trung bình (Optimal Allocation Treemap)")
            fig_tree = px.treemap(
                weights_df, 
                path=["Asset"], 
                values="Weight", 
                title="Tỷ trọng phân bổ các mã",
                template="plotly_dark"
            )
            st.plotly_chart(fig_tree, use_container_width=True)
            
            # Display weights table
            st.markdown("#### Bảng tỷ trọng chi tiết")
            st.dataframe(weights_df.style.format({"Weight": "{:.2%}"}))

# Tab 4: Performance Evaluation & Backtest
with tab4:
    if not st.session_state.model_trained:
        st.warning("Vui lòng huấn luyện mô hình ở Tab 3 trước!")
    else:
        st.subheader("Đánh giá hiệu quả chiến lược & Backtest trên tập kiểm thử (2025)")
        
        # Retrieve stored assets
        pred_weights_test = st.session_state.pred_weights_test
        test_seq_dates = st.session_state.test_seq_dates
        y_test_target = st.session_state.y_test_target
        test_returns = st.session_state.test_returns
        train_returns = st.session_state.train_returns
        top_symbols = st.session_state.top_symbols
        best_weights = st.session_state.best_weights
        
        # Baseline 1: Equal weight
        Allo_1 = pd.DataFrame({
            "Asset": top_symbols,
            "Weight": [1.0 / len(top_symbols)] * len(top_symbols)
        })
        
        # Baseline 2: 80-20 weight allocation
        def build_allocation_80_20(train_returns):
            rf_daily = rf_annual / trading_days
            mean_ret = train_returns.mean()
            std_ret = train_returns.std().replace(0, np.nan)
            sharpe_train = ((mean_ret - rf_daily) / std_ret).dropna().sort_values(ascending=False)
            
            ranked = sharpe_train.reset_index()
            ranked.columns = ["Asset", "Score"]
            
            n_assets = len(ranked)
            top_count = max(1, int(np.ceil(0.2 * n_assets)))
            bottom_count = n_assets - top_count
            
            top_weights = [0.8 / top_count] * top_count
            bottom_weights = [0.2 / bottom_count] * bottom_count if bottom_count > 0 else []
            
            ranked["Weight"] = top_weights + bottom_weights
            return ranked[["Asset", "Weight"]]
            
        Allo_2 = build_allocation_80_20(train_returns)
        
        # 1. Performance characterization functions
        def port_char(weights_df, returns_df, annualize=True, freq=trading_days):
            er = returns_df.mean().reset_index()
            er.columns = ["Asset", "Er"]
            weights_merged = pd.merge(weights_df, er, on="Asset", how="left").fillna(0.0)
            portfolio_er_daily = np.dot(weights_merged["Weight"], weights_merged["Er"])
            
            cov_matrix = returns_df.cov()
            asset_order = weights_merged["Asset"].tolist()
            cov_matrix = cov_matrix.loc[asset_order, asset_order]
            w = weights_merged["Weight"].values
            portfolio_std_daily = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
            
            if annualize:
                portfolio_er = portfolio_er_daily * freq
                portfolio_std_dev = portfolio_std_daily * np.sqrt(freq)
            else:
                portfolio_er = portfolio_er_daily
                portfolio_std_dev = portfolio_std_daily
            return portfolio_er, portfolio_std_dev
            
        def port_char_from_series(portfolio_return_series, annualize=True, freq=trading_days):
            portfolio_return_series = pd.Series(portfolio_return_series).dropna()
            er_daily = portfolio_return_series.mean()
            std_daily = portfolio_return_series.std()
            if annualize:
                er = er_daily * freq
                std = std_daily * np.sqrt(freq)
            else:
                er = er_daily
                std = std_daily
            return er, std
            
        def sharpe_port(weights_df, returns_df, rf=rf_annual, freq=trading_days):
            portfolio_er, portfolio_std_dev = port_char(weights_df, returns_df, annualize=True, freq=freq)
            return (portfolio_er - rf) / (portfolio_std_dev + 1e-12)
            
        def sharpe_from_series(portfolio_return_series, rf=rf_annual, freq=trading_days):
            portfolio_er, portfolio_std_dev = port_char_from_series(portfolio_return_series, annualize=True, freq=freq)
            return (portfolio_er - rf) / (portfolio_std_dev + 1e-12)
            
        # 2. Compute dynamic LSTM-GRU portfolio return
        weights_test_df = pd.DataFrame(pred_weights_test, index=test_seq_dates, columns=top_symbols)
        y_test_df = pd.DataFrame(y_test_target, index=test_seq_dates, columns=top_symbols)
        
        portfolio_returns_lstm_dynamic = (weights_test_df * y_test_df).sum(axis=1)
        
        # Compute returns series for baselines
        portfolio_returns_eq = test_returns[Allo_1['Asset'].tolist()].dot(Allo_1['Weight'].values).loc[test_seq_dates]
        portfolio_returns_80_20 = test_returns[Allo_2['Asset'].tolist()].dot(Allo_2['Weight'].values).loc[test_seq_dates]
        
        # Characteristics
        Er_lstm, std_lstm = port_char_from_series(portfolio_returns_lstm_dynamic, annualize=True)
        Er_1, std_1 = port_char(Allo_1, test_returns, annualize=True)
        Er_2, std_2 = port_char(Allo_2, test_returns, annualize=True)
        
        sharpe_lstm = sharpe_from_series(portfolio_returns_lstm_dynamic)
        sharpe_1 = sharpe_port(Allo_1, test_returns)
        sharpe_2 = sharpe_port(Allo_2, test_returns)
        
        # Build comparison table
        comparison_table = pd.DataFrame({
            "Chiến lược đầu tư": ["LSTM-GRU (Dynamic)", "Phân bổ đều", "Phân bổ 80-20"],
            "Lợi nhuận trung bình": [Er_lstm, Er_1, Er_2],
            "Độ lệch chuẩn": [std_lstm, std_1, std_2],
            "Hệ số Sharpe": [sharpe_lstm, sharpe_1, sharpe_2]
        })
        
        # Format comparison table for display
        comparison_table_display = comparison_table.copy()
        comparison_table_display["Lợi nhuận trung bình"] = comparison_table_display["Lợi nhuận trung bình"] * 100
        comparison_table_display["Độ lệch chuẩn"] = comparison_table_display["Độ lệch chuẩn"] * 100
        
        # Display KPI cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Hệ số Sharpe Cao Nhất</div>
                <div class="kpi-value">{sharpe_lstm:.2f}</div>
                <div class="kpi-strategy">Chiến lược LSTM-GRU</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Lợi nhuận kỳ vọng</div>
                <div class="kpi-value">{Er_lstm*100:.2f}%</div>
                <div class="kpi-strategy">Chiến lược LSTM-GRU</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Độ lệch chuẩn (Biến động)</div>
                <div class="kpi-value">{std_lstm*100:.2f}%</div>
                <div class="kpi-strategy">Chiến lược LSTM-GRU</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Display performance table
        st.markdown("#### Bảng so sánh các chỉ số hiệu quả")
        
        st.dataframe(
            comparison_table_display.style.format({
                "Lợi nhuận trung bình": "{:.2f}%",
                "Độ lệch chuẩn": "{:.2f}%",
                "Hệ số Sharpe": "{:.2f}"
            })
        )
        
        # Cumulative returns comparison chart
        st.markdown("#### Biểu đồ so sánh lợi nhuận lũy kế (Cumulative Returns Backtest)")
        
        cum_lstm = (1 + portfolio_returns_lstm_dynamic).cumprod() - 1
        cum_eq = (1 + portfolio_returns_eq).cumprod() - 1
        cum_80_20 = (1 + portfolio_returns_80_20).cumprod() - 1
        
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=test_seq_dates, y=cum_lstm * 100, name="LSTM-GRU (Dynamic)", mode="lines", line=dict(color='#636EFA', width=3)))
        fig_cum.add_trace(go.Scatter(x=test_seq_dates, y=cum_eq * 100, name="Phân bổ đều (Equal Weight)", mode="lines", line=dict(color='#00CC96', width=2)))
        fig_cum.add_trace(go.Scatter(x=test_seq_dates, y=cum_80_20 * 100, name="Phân bổ 80-20", mode="lines", line=dict(color='#EF553B', width=2)))
        
        fig_cum.update_layout(
            template="plotly_dark",
            xaxis_title="Thời gian",
            yaxis_title="Lợi nhuận tích lũy (%)",
            margin=dict(l=20, r=20, t=30, b=20),
            height=500
        )
        st.plotly_chart(fig_cum, use_container_width=True)
        
        # Dual-Axis performance metrics chart
        st.markdown("#### Biểu đồ gộp so sánh các Metric (Lợi nhuận vs Rủi ro vs Sharpe)")
        
        # We can construct a beautiful bar chart of return and std on left axis, line for sharpe on right axis
        fig_metrics = go.Figure()
        
        fig_metrics.add_trace(go.Bar(
            name="Lợi nhuận (%)",
            x=comparison_table_display["Chiến lược đầu tư"],
            y=comparison_table_display["Lợi nhuận trung bình"],
            yaxis="y",
            marker_color="#636EFA",
            text=comparison_table_display["Lợi nhuận trung bình"].round(2).astype(str) + "%",
            textposition='auto'
        ))
        
        fig_metrics.add_trace(go.Bar(
            name="Độ lệch chuẩn (%)",
            x=comparison_table_display["Chiến lược đầu tư"],
            y=comparison_table_display["Độ lệch chuẩn"],
            yaxis="y",
            marker_color="#00CC96",
            text=comparison_table_display["Độ lệch chuẩn"].round(2).astype(str) + "%",
            textposition='auto'
        ))
        
        fig_metrics.add_trace(go.Scatter(
            name="Hệ số Sharpe",
            x=comparison_table_display["Chiến lược đầu tư"],
            y=comparison_table["Hệ số Sharpe"],
            yaxis="y2",
            mode="lines+markers+text",
            line=dict(color="red", width=3),
            text=comparison_table["Hệ số Sharpe"].round(2).astype(str),
            textposition="top center"
        ))
        
        fig_metrics.update_layout(
            template="plotly_dark",
            yaxis=dict(title="Giá trị (%)"),
            yaxis2=dict(title="Hệ số Sharpe", overlaying="y", side="right"),
            barmode='group',
            margin=dict(l=20, r=20, t=30, b=20),
            height=450,
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
