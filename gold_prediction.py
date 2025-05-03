import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Gold Price Forecasting",
    page_icon="💰",
    layout="wide"
)

# Fungsi untuk styling
def local_css():
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
            margin-bottom: 10px;
        }
        .main {
            padding: 2rem;
        }
        h1 {
            color: #FFD700;
        }
        </style>
    """, unsafe_allow_html=True)

# Konfigurasi cache untuk model dan scaler
@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_scaler():
    try:
        with open('scaler.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.warning("Membuat scaler baru...")
        scaler = MinMaxScaler(feature_range=(0, 1))
        if os.path.exists('gld_price_data.csv'):
            df = pd.read_csv('gld_price_data.csv')
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                data = df[numeric_columns[0]].values.reshape(-1, 1)
                scaler.fit(data)
        else:
            scaler.fit([[0], [1]])
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        return scaler

@st.cache_data
def preprocess_data(data, _scaler, sequence_length=60):
    """
    Preprocess data dengan menggunakan scaler yang diberikan
    Args:
        data: Data yang akan di-preprocess
        _scaler: Scaler yang digunakan
        sequence_length: Panjang sequence untuk input model
    """
    try:
        if len(data) < sequence_length:
            st.error(f"❌ Data terlalu sedikit. Minimal {sequence_length} data point diperlukan.")
            return np.array([])
            
        scaled_data = _scaler.transform(data)
        X = []
        
        # Hanya ambil sequence terakhir untuk prediksi
        last_sequence = scaled_data[-sequence_length:].reshape(sequence_length, 1)
        return last_sequence
        
    except Exception as e:
        st.error(f"❌ Error dalam preprocessing data: {str(e)}")
        return np.array([])

@st.cache_data
def prepare_prediction_data(df, value_column, date_column=None):
    """
    Menyiapkan data untuk prediksi dengan caching
    """
    try:
        # Konversi tanggal jika ada
        if date_column:
            df[date_column] = pd.to_datetime(df[date_column])
            
        # Ambil hanya data numerik dari kolom nilai
        data = df[value_column].values.reshape(-1, 1)
        
        if date_column:
            dates = df[date_column].values
        else:
            dates = range(len(data))
            
        return data, dates
    except Exception as e:
        st.error(f"❌ Error dalam persiapan data: {str(e)}")
        return None, None

def predict_future(model, data, _scaler, days_to_predict=30):
    if model is None:
        st.error("Model tidak dapat dimuat")
        return [], []
    
    try:
        # Pastikan data memiliki dimensi yang benar (60 timesteps)
        if data.shape != (60, 1):
            st.error(f"❌ Format data tidak sesuai. Dibutuhkan (60, 1) tetapi mendapat {data.shape}")
            return [], []
        
        predictions = []
        current_sequence = data.reshape(1, 60, 1)  # Reshape ke format (batch_size, timesteps, features)
        current_date = datetime.now()
        future_dates = []
        
        # Dapatkan nilai terakhir yang diketahui
        last_known_value = float(_scaler.inverse_transform(data[-1].reshape(1, -1))[-1][0])
        
        for i in range(days_to_predict):
            try:
                # Set input tensor
                input_details = model.get_input_details()
                model.set_tensor(input_details[0]['index'], current_sequence.astype(np.float32))
                
                # Jalankan inferensi
                model.invoke()
                
                # Dapatkan output
                output_details = model.get_output_details()
                prediction = model.get_tensor(output_details[0]['index'])
                prediction_value = float(_scaler.inverse_transform(prediction)[0][0])
                
                # Tambahkan hasil prediksi
                predictions.append(prediction_value)
                future_dates.append(current_date + timedelta(days=i+1))
                
                # Update sequence untuk prediksi berikutnya
                new_value = _scaler.transform([[prediction_value]])[0]
                current_sequence = np.append(current_sequence[:, 1:, :], 
                                          new_value.reshape(1, 1, 1), 
                                          axis=1)
                
            except Exception as e:
                st.error(f"❌ Error pada prediksi hari ke-{i+1}: {str(e)}")
                if len(predictions) > 0:
                    return predictions, future_dates[:len(predictions)]
                return [], []
        
        return predictions, future_dates
        
    except Exception as e:
        st.error(f"❌ Error dalam prediksi: {str(e)}")
        return [], []

@st.cache_data
def calculate_yearly_predictions(_daily_predictions, _start_value):
    """
    Menghitung prediksi tahunan berdasarkan tren harian dengan pembatasan pertumbuhan
    """
    try:
        # Menggunakan nilai terakhir prediksi harian sebagai dasar
        current_value = _daily_predictions[-1]  # Menggunakan nilai prediksi terakhir
        
        yearly_predictions = []
        current_year = datetime.now().year
        years = []
        
        # Hitung rata-rata perubahan harian
        daily_changes = [(_daily_predictions[i] - _daily_predictions[i-1])/_daily_predictions[i-1] 
                        for i in range(1, len(_daily_predictions))]
        avg_daily_change = np.mean(daily_changes)
        
        # Batasi perubahan harian rata-rata ke maksimum 0.5%
        avg_daily_change = np.clip(avg_daily_change, -0.005, 0.005)
        
        # Konversi ke perubahan tahunan (252 hari trading)
        # Menggunakan compound growth yang lebih moderat
        yearly_growth_rate = (1 + avg_daily_change) ** 252 - 1
        
        # Batasi pertumbuhan tahunan maksimum ke 20%
        yearly_growth_rate = np.clip(yearly_growth_rate, -0.20, 0.20)
        
        # Prediksi 5 tahun ke depan dengan pertumbuhan yang lebih realistis
        for i in range(5):
            yearly_predictions.append(float(current_value))
            years.append(current_year + i + 1)
            
            # Terapkan pertumbuhan dengan faktor penurunan untuk tahun-tahun berikutnya
            growth_factor = 1.0 / (i + 1)  # Faktor penurunan berdasarkan tahun
            adjusted_growth = yearly_growth_rate * growth_factor
            current_value = current_value * (1 + adjusted_growth)
        
        return yearly_predictions, years
    except Exception as e:
        st.error(f"❌ Error dalam perhitungan tahunan: {str(e)}")
        return [float(_start_value)] * 5, list(range(current_year + 1, current_year + 6))

def format_currency(x):
    """Format nilai ke dalam format currency USD"""
    return f'${x:,.2f}'

@st.cache_data
def create_daily_plot(_df_index, _data, _predictions, _future_dates, date_column, value_column):
    """
    Membuat plot prediksi harian dengan ukuran yang lebih kecil
    """
    try:
        # Buat figure dengan ukuran yang lebih kecil
        fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
        ax.set_facecolor('#f8f9fa')
        
        # Plot hanya data prediksi
        ax.plot(_future_dates, _predictions, 
               label='Prediksi', 
               color='#E74C3C', 
               linewidth=2,
               marker='s',
               markersize=4,
               markerfacecolor='white',
               markeredgecolor='#E74C3C',
               markeredgewidth=1)
        
        # Tambahkan area fill di bawah garis prediksi
        ax.fill_between(_future_dates, _predictions, 
                       alpha=0.1, 
                       color='#E74C3C')
        
        # Format x-axis
        plt.gcf().autofmt_xdate()
        
        # Styling plot yang lebih jelas
        ax.grid(True, linestyle='--', alpha=0.8, color='#cccccc')
        
        # Atur warna dan ukuran font untuk judul dan label
        ax.set_title('Prediksi Harga Emas 30 Hari Kedepan', 
                    pad=10, 
                    fontsize=10, 
                    fontweight='bold',
                    color='#2C3E50')
        ax.set_xlabel('Tanggal', 
                     labelpad=5, 
                     fontsize=8, 
                     fontweight='bold',
                     color='#2C3E50')
        ax.set_ylabel('Harga (USD)', 
                     labelpad=5, 
                     fontsize=8, 
                     fontweight='bold',
                     color='#2C3E50')
        
        # Legend dengan ukuran lebih kecil
        legend = ax.legend(loc='upper left', 
                         fontsize=8, 
                         bbox_to_anchor=(0.02, 0.98),
                         facecolor='white',
                         edgecolor='none')
        
        # Format y-axis values dengan ukuran lebih kecil
        y_min, y_max = min(_predictions), max(_predictions)
        margin = (y_max - y_min) * 0.05
        ax.set_ylim(y_min - margin, y_max + margin)
        
        # Format y-axis ticks dengan ukuran lebih kecil
        yticks = ax.get_yticks()
        ax.set_yticklabels(['${:,.0f}'.format(x) for x in yticks], 
                          fontsize=7,
                          color='#2C3E50')
        ax.tick_params(axis='both', 
                      which='major', 
                      labelsize=7, 
                      colors='#2C3E50')
        
        # Tambahkan label nilai dengan format yang lebih kecil
        if len(_predictions) > 0:
            # Label untuk nilai awal prediksi
            plt.annotate(f'${_predictions[0]:,.2f}', 
                        xy=(_future_dates[0], _predictions[0]),
                        xytext=(5, 5), 
                        textcoords='offset points',
                        fontsize=7,
                        color='#2C3E50',
                        bbox=dict(facecolor='white', 
                                edgecolor='#E74C3C',
                                alpha=0.9,
                                boxstyle='round,pad=0.3'))
            
            # Label untuk nilai akhir prediksi
            plt.annotate(f'${_predictions[-1]:,.2f}',
                        xy=(_future_dates[-1], _predictions[-1]),
                        xytext=(5, -5), 
                        textcoords='offset points',
                        fontsize=7,
                        color='#2C3E50',
                        bbox=dict(facecolor='white', 
                                edgecolor='#E74C3C',
                                alpha=0.9,
                                boxstyle='round,pad=0.3'))
            
            # Tambahkan label persentase perubahan
            pct_change = ((_predictions[-1] - _predictions[0]) / _predictions[0]) * 100
            color = '#27AE60' if pct_change >= 0 else '#E74C3C'
            plt.annotate(f'Perubahan: {pct_change:+.1f}%',
                        xy=(0.98, 0.02),
                        xycoords='axes fraction',
                        fontsize=8,
                        color=color,
                        bbox=dict(facecolor='white', 
                                edgecolor=color,
                                alpha=0.9,
                                boxstyle='round,pad=0.3'),
                        ha='right',
                        va='bottom')
        
        # Atur margin plot
        plt.tight_layout()
        
        # Tambahkan background grid yang lebih halus
        ax.grid(True, which='minor', linestyle=':', alpha=0.4, color='#dddddd')
        ax.minorticks_on()
        
        # Atur spines (border plot)
        for spine in ax.spines.values():
            spine.set_color('#cccccc')
            spine.set_linewidth(0.5)
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Error dalam pembuatan plot harian: {str(e)}")
        return None

@st.cache_data
def create_yearly_plot(_yearly_predictions, value_column):
    """
    Membuat plot prediksi tahunan dengan ukuran yang lebih kecil
    """
    try:
        # Buat figure dengan ukuran yang lebih kecil
        fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
        ax.set_facecolor('#f8f9fa')
        
        # Siapkan data
        current_year = datetime.now().year
        years = [current_year + i for i in range(1, 6)]
        x_positions = range(len(years))
        
        # Plot batang untuk prediksi tahunan
        bars = ax.bar(x_positions, _yearly_predictions,
                     color='#3498db',
                     alpha=0.7,
                     width=0.5)
        
        # Tambahkan garis trend
        ax.plot(x_positions, _yearly_predictions,
                color='#e74c3c',
                linewidth=2,
                marker='o',
                markersize=5,
                markerfacecolor='white',
                markeredgecolor='#e74c3c',
                markeredgewidth=1,
                zorder=5)
        
        # Styling yang lebih jelas
        ax.grid(True, linestyle='--', alpha=0.3, color='#cccccc', zorder=0)
        
        # Atur warna dan ukuran font untuk judul dan label
        ax.set_title('Proyeksi Harga Emas 5 Tahun Kedepan', 
                    pad=10, 
                    fontsize=10, 
                    fontweight='bold',
                    color='#2C3E50')
        ax.set_xlabel('Tahun', 
                     labelpad=5, 
                     fontsize=8, 
                     fontweight='bold',
                     color='#2C3E50')
        ax.set_ylabel('Harga (USD)', 
                     labelpad=5, 
                     fontsize=8, 
                     fontweight='bold',
                     color='#2C3E50')
        
        # Format axis
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(year) for year in years], 
                          fontsize=7,
                          color='#2C3E50')
        
        # Format y-axis dengan style currency
        yticks = ax.get_yticks()
        ax.set_yticklabels(['${:,.0f}'.format(x) for x in yticks], 
                          fontsize=7,
                          color='#2C3E50')
        
        # Tambahkan label nilai dan persentase perubahan
        prev_value = _yearly_predictions[0]
        for i, value in enumerate(_yearly_predictions):
            # Label nilai di atas bar
            ax.text(i, value, f'${value:,.0f}',
                   ha='center',
                   va='bottom',
                   fontsize=7,
                   color='#2C3E50',
                   bbox=dict(facecolor='white',
                           edgecolor='#3498db',
                           alpha=0.9,
                           boxstyle='round,pad=0.3'))
            
            # Label persentase perubahan
            if i > 0:
                pct_change = ((value - prev_value) / prev_value) * 100
                color = '#27AE60' if pct_change >= 0 else '#E74C3C'
                y_pos = min(value, prev_value)
                
                # Tambahkan panah dan label persentase
                ax.annotate(f'{pct_change:+.1f}%',
                          xy=(i-0.5, y_pos),
                          xytext=(0, -15),
                          textcoords='offset points',
                          ha='center',
                          va='top',
                          fontsize=7,
                          color=color,
                          bbox=dict(facecolor='white',
                                  edgecolor=color,
                                  alpha=0.9,
                                  boxstyle='round,pad=0.3'),
                          arrowprops=dict(arrowstyle='->',
                                        color=color,
                                        alpha=0.6))
            prev_value = value
        
        # Tambahkan total perubahan
        total_change = ((_yearly_predictions[-1] - _yearly_predictions[0]) / _yearly_predictions[0]) * 100
        color = '#27AE60' if total_change >= 0 else '#E74C3C'
        ax.text(0.98, 0.02,
                f'Total: {total_change:+.1f}%',
                transform=ax.transAxes,
                ha='right',
                va='bottom',
                fontsize=8,
                bbox=dict(facecolor='white',
                        edgecolor=color,
                        alpha=0.9,
                        boxstyle='round,pad=0.3'),
                color=color)
        
        # Atur spines (border plot)
        for spine in ax.spines.values():
            spine.set_color('#cccccc')
            spine.set_linewidth(0.5)
        
        # Atur margin plot
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Error dalam pembuatan plot tahunan: {str(e)}")
        return None

@st.cache_data
def validate_data(df, date_column, value_column):
    """
    Memvalidasi data yang diupload
    Args:
        df: DataFrame yang akan divalidasi
        date_column: Nama kolom tanggal
        value_column: Nama kolom nilai
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Validasi jumlah data
        if len(df) < 60:
            return False, "Data minimal 60 baris untuk melakukan prediksi"
            
        # Validasi kolom tanggal
        try:
            df[date_column] = pd.to_datetime(df[date_column])
        except Exception as e:
            return False, f"Kolom {date_column} harus berformat tanggal yang valid (contoh: YYYY-MM-DD, MM/DD/YYYY)"
            
        # Validasi kolom nilai
        try:
            df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
            if df[value_column].isnull().any():
                return False, f"Kolom {value_column} mengandung nilai yang tidak valid"
        except Exception as e:
            return False, f"Kolom {value_column} harus berisi nilai numerik"
            
        # Validasi nilai negatif
        if (df[value_column] < 0).any():
            return False, f"Kolom {value_column} tidak boleh mengandung nilai negatif"
            
        # Validasi nilai yang hilang
        if df[value_column].isnull().any() or df[date_column].isnull().any():
            return False, "Data tidak boleh mengandung nilai yang hilang (NA/null)"
            
        # Validasi urutan tanggal
        if not df[date_column].is_monotonic_increasing:
            return False, "Data harus diurutkan berdasarkan tanggal secara ascending"
            
        return True, ""
        
    except Exception as e:
        return False, f"Error validasi data: {str(e)}"

@st.cache_data
def format_prediction_results(predictions, future_dates, yearly_predictions, years):
    """
    Memformat hasil prediksi untuk ditampilkan
    """
    try:
        # Format prediksi harian
        df_daily = pd.DataFrame({
            'Tanggal': [d.strftime('%Y-%m-%d') for d in future_dates],
            'Prediksi (USD)': predictions,
            'Perubahan (USD)': [0] + [predictions[i] - predictions[i-1] for i in range(1, len(predictions))],
            'Perubahan (%)': [0] + [(predictions[i] - predictions[i-1])/predictions[i-1]*100 for i in range(1, len(predictions))]
        })
        
        # Format prediksi tahunan
        df_yearly = pd.DataFrame({
            'Tahun': years,
            'Prediksi (USD)': yearly_predictions,
            'Perubahan (USD)': [0] + [yearly_predictions[i] - yearly_predictions[i-1] for i in range(1, len(yearly_predictions))],
            'Perubahan (%)': ['Base'] + [f"{((yearly_predictions[i] - yearly_predictions[i-1])/yearly_predictions[i-1]*100):.2f}%" 
                                        for i in range(1, len(yearly_predictions))]
        })
        
        return df_daily, df_yearly
        
    except Exception as e:
        st.error(f"❌ Error dalam format hasil prediksi: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def display_prediction_metrics(predictions, yearly_predictions):
    """
    Menampilkan metrik-metrik prediksi
    """
    try:
        # Metrik harian
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Harga Awal", format_currency(predictions[0]))
        with col2:
            st.metric("Harga Akhir", format_currency(predictions[-1]))
        with col3:
            change = predictions[-1] - predictions[0]
            st.metric("Perubahan Total", format_currency(change))
        with col4:
            pct_change = (change / predictions[0]) * 100
            st.metric("Perubahan %", f"{pct_change:.2f}%")
            
        # Metrik tahunan
        st.markdown("### 📊 Ringkasan Prediksi 5 Tahun")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tahun Pertama", format_currency(yearly_predictions[0]))
        with col2:
            st.metric("Tahun Ketiga", format_currency(yearly_predictions[2]))
        with col3:
            st.metric("Tahun Kelima", format_currency(yearly_predictions[4]))
            
    except Exception as e:
        st.error(f"❌ Error dalam menampilkan metrik: {str(e)}")

def home_page():
    # Tambahkan custom CSS untuk judul
    st.markdown("""
        <style>
        .welcome-title {
            background: linear-gradient(120deg, #FFD700, #FFA500);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 1.6em;
            font-weight: bold;
            text-align: center;
            padding: 10px;
            margin-bottom: 15px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }
        .model-info {
            background: linear-gradient(135deg, #FFF8E7, #FFF4D4);
            padding: 8px;
            border-radius: 6px;
            border-left: 3px solid #FFD700;
            margin: 10px 0;
        }
        .model-info h4 {
            color: #B8860B;
            margin: 0 0 5px 0;
            font-size: 0.95em;
            font-weight: bold;
        }
        .model-info p {
            color: #333;
            margin: 0;
            line-height: 1.3;
            font-size: 0.85em;
        }
        .feature-title {
            color: #FFB700;
            font-size: 1em;
            font-weight: bold;
            margin-top: 12px;
            display: inline-block;
        }
        .feature-list {
            color: #333;
            font-size: 0.85em;
            margin-left: 12px;
            line-height: 1.3;
        }
        .feature-list ul {
            margin: 5px 0;
            padding-left: 15px;
        }
        .feature-list li {
            margin: 3px 0;
        }
        .author-info {
            background-color: #FFF8E7;
            padding: 10px;
            border-radius: 6px;
            border-left: 3px solid #FFD700;
            margin-top: 12px;
        }
        .author-info h3 {
            color: #B8860B;
            font-size: 0.95em;
            margin: 0 0 5px 0;
        }
        .author-info ul {
            margin: 0;
            padding-left: 0;
        }
        .author-info li {
            font-size: 0.85em;
            margin: 2px 0;
        }
        h3 {
            font-size: 1em;
            margin: 10px 0 5px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Gunakan kelas CSS kustom untuk judul
    st.markdown('<h1 class="welcome-title">📈 Selamat Datang di Aplikasi Gold Price Forecasting</h1>', unsafe_allow_html=True)
    
    try:
        # Menggunakan use_container_width alih-alih use_column_width
        st.image("gold.jpg", caption="Prediksi Harga Emas", use_container_width=True)
    except Exception as e:
        st.warning("Gambar gold.jpg tidak ditemukan. Pastikan file gambar berada di direktori yang sama dengan script.")
    
    st.markdown("""
    ### <span class="feature-title">🎯 Tentang Aplikasi</span>
    <div class="feature-list">
    Aplikasi ini dikembangkan untuk memprediksi harga emas menggunakan model machine learning berbasis Deep Learning LSTM-RNN (Long Short-Term Memory - Recurrent Neural Network).
    </div>

    <div class="model-info">
    <h4>🤖 Tentang Model LSTM-RNN:</h4>
    <p>
    Model LSTM-RNN adalah arsitektur deep learning yang sangat cocok untuk prediksi data time series seperti harga emas. LSTM memiliki kemampuan untuk mengingat pola jangka panjang, sementara RNN membantu memahami konteks temporal data, memungkinkan prediksi yang lebih akurat berdasarkan tren historis.
    </p>
    </div>
    
    ### <span class="feature-title">✨ Fitur Utama:</span>
    <div class="feature-list">
    <ul>
    <li><b>Prediksi Harga:</b>
       <ul>
       <li>Prediksi harga emas harian (30 hari ke depan)</li>
       <li>Prediksi harga emas tahunan (5 tahun ke depan)</li>
       </ul>
    </li>
    <li><b>Visualisasi Data:</b>
       <ul>
       <li>Grafik historis harga emas</li>
       <li>Analisis tren dan pola</li>
       </ul>
    </li>
    <li><b>Upload Data:</b>
       <ul>
       <li>Dukungan untuk analisis data kustom</li>
       <li>Format CSV yang fleksibel</li>
       </ul>
    </li>
    </ul>
    </div>
    
    ### <span class="feature-title">📝 Cara Penggunaan:</span>
    <div class="feature-list">
    <ul>
    <li>Gunakan menu navigasi di sidebar untuk berpindah antar halaman</li>
    <li>Pilih "PREDIKSI" untuk melihat prediksi harga emas</li>
    <li>Pilih "VISUALISASI MODEL" untuk melihat analisis data</li>
    </ul>
    </div>
    
    <div class="author-info">
    <h3>👨‍💻 Dibuat Oleh:</h3>
    <ul style="list-style-type: none;">
        <li><b>Nama:</b> Irfan Zulkarnaen</li>
        <li><b>NIM:</b> 231352002</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def visualization_page():
    st.title("Visualisasi Data Emas")
    
    if os.path.exists('gld_price_data.csv'):
        df = pd.read_csv('gld_price_data.csv')
        
        # Konversi kolom Date ke datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        st.markdown("""
        ### Informasi Dataset
        - **GLD**: Harga ETF emas SPDR dalam USD (Dollar Amerika)
        - **SPX**: Indeks S&P 500
        - **USO**: United States Oil Fund LP ETF
        - **SLV**: iShares Silver Trust ETF
        - **EUR/USD**: Nilai tukar Euro terhadap USD
        """)
        
        st.subheader("Grafik Historis Harga Emas (USD)")
        fig, ax = plt.subplots(figsize=(10, 6))
        if 'GLD' in df.columns:
            ax.plot(df.index, df['GLD'], label='Harga Emas (USD)')
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
            ax.set_title('Historis Harga Emas SPDR Gold Shares (GLD)')
            ax.set_xlabel('Periode')
            ax.set_ylabel('Harga (USD)')
            ax.legend()
            
            # Menambahkan anotasi untuk nilai tertinggi dan terendah
            max_price = df['GLD'].max()
            min_price = df['GLD'].min()
            max_date = df['GLD'].idxmax()
            min_date = df['GLD'].idxmin()
            
            plt.annotate(f'Tertinggi: ${max_price:.2f}',
                        xy=(max_date, max_price),
                        xytext=(10, 10),
                        textcoords='offset points')
            plt.annotate(f'Terendah: ${min_price:.2f}',
                        xy=(min_date, min_price),
                        xytext=(10, -10),
                        textcoords='offset points')
            
            st.pyplot(fig)
            
            # Menampilkan statistik harga
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Harga Tertinggi", f"${max_price:.2f}")
            with col2:
                st.metric("Harga Terendah", f"${min_price:.2f}")
            with col3:
                current_price = df['GLD'].iloc[-1]
                st.metric("Harga Terakhir", f"${current_price:.2f}")
        else:
            st.warning("Kolom 'GLD' tidak ditemukan dalam dataset")
        
        st.subheader("Statistik Deskriptif (dalam USD)")
        # Hanya tampilkan statistik untuk kolom numerik
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        stats_df = df[numeric_columns].describe()
        # Format nilai dalam statistik
        stats_df = stats_df.round(2)
        st.write(stats_df)
        
        st.subheader("Korelasi Antar Variabel")
        # Hitung korelasi hanya untuk kolom numerik
        correlation_matrix = df[numeric_columns].corr()
        
        # Plot heatmap korelasi
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
        plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
        plt.title('Peta Korelasi Antar Variabel')
        
        # Menambahkan nilai korelasi di dalam heatmap
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                        ha='center', va='center')
        
        st.pyplot(fig)
        
        # Tampilkan data mentah
        st.subheader("Data Mentah (5 Baris Pertama)")
        st.write(df.head())
    else:
        st.warning("File data tidak ditemukan. Silakan upload data terlebih dahulu di menu Prediksi.")

def prediction_page():
    st.title("Prediksi Harga Emas")
    
    # Informasi format data yang dibutuhkan
    st.info("""
    ℹ️ Format Data yang Dibutuhkan:
    - File CSV dengan minimal kolom tanggal dan harga
    - Kolom tanggal dapat berformat:
      * YYYY-MM-DD (contoh: 2024-03-20)
      * MM/DD/YYYY (contoh: 03/20/2024)
      * DD/MM/YYYY (contoh: 20/03/2024)
    - Kolom harga harus berisi nilai numerik positif
    - Data minimal 60 baris untuk prediksi
    - Data harus berurutan berdasarkan tanggal
    - Tidak boleh ada nilai yang hilang
    """)
    
    uploaded_file = st.file_uploader("Upload file CSV data harga emas", type=['csv'])
    
    if uploaded_file is None:
        st.warning("⚠️ Silakan upload file CSV terlebih dahulu untuk melakukan prediksi")
        return
    
    try:
        # Baca data
        df = pd.read_csv(uploaded_file)
        
        # Tampilkan preview data
        st.subheader("📋 Preview Data")
        st.write(df.head())
        
        # Validasi format data
        if len(df.columns) < 2:
            st.error("❌ File CSV harus memiliki minimal 2 kolom (tanggal dan harga)")
            return
            
        # Pilihan kolom
        date_column = st.selectbox("Pilih kolom tanggal:", df.columns)
        value_column = st.selectbox("Pilih kolom harga:", df.columns)
        
        # Validasi data
        is_valid, error_message = validate_data(df, date_column, value_column)
        if not is_valid:
            st.error(f"❌ {error_message}")
            return
            
        # Urutkan data berdasarkan tanggal
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(by=date_column)
        
        # Load model dan scaler
        model = load_model()
        scaler = load_scaler()
        
        if model is None or scaler is None:
            st.error("❌ Gagal memuat model atau scaler")
            return
            
        # Persiapkan data
        data, dates = prepare_prediction_data(df, value_column, date_column)
        if data is None or dates is None:
            st.error("❌ Gagal mempersiapkan data")
            return
            
        # Tampilkan informasi data
        st.info(f"ℹ️ Menggunakan {len(data)} data point untuk prediksi")
        
        # Tambahkan slider untuk jumlah hari prediksi
        days_to_predict = st.slider("Jumlah hari untuk prediksi:", 1, 90, 30)
        
        # Tambahkan tombol prediksi
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            predict_button = st.button("🔮 Mulai Prediksi", use_container_width=True)
        
        if predict_button:
            # Preprocessing
            sequence = preprocess_data(data, scaler)
            if len(sequence) == 0:
                st.error("❌ Gagal melakukan preprocessing data")
                return
                
            # Prediksi
            with st.spinner('🔄 Melakukan prediksi...'):
                predictions, future_dates = predict_future(model, sequence, scaler, days_to_predict)
                
            if len(predictions) == 0:
                st.error("❌ Gagal melakukan prediksi")
                return
                
            # Format hasil prediksi
            df_daily, df_yearly = format_prediction_results(predictions, future_dates, 
                                                          *calculate_yearly_predictions(predictions, data[-1][0]))
            
            # Buat tab untuk prediksi harian dan tahunan
            tab1, tab2 = st.tabs(["📈 Prediksi Harian", "📊 Prediksi Tahunan"])
            
            with tab1:
                st.subheader(f"Prediksi {days_to_predict} Hari Kedepan")
                
                # Tampilkan metrik harian
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Harga Awal", format_currency(predictions[0]))
                with col2:
                    st.metric("Harga Akhir", format_currency(predictions[-1]))
                with col3:
                    change = predictions[-1] - predictions[0]
                    st.metric("Perubahan Total", format_currency(change))
                with col4:
                    pct_change = (change / predictions[0]) * 100
                    st.metric("Perubahan %", f"{pct_change:.2f}%")
                
                # Tampilkan tabel prediksi harian
                st.dataframe(
                    df_daily.style
                    .format({
                        'Prediksi (USD)': format_currency,
                        'Perubahan (USD)': format_currency,
                        'Perubahan (%)': '{:.2f}%'
                    })
                    .background_gradient(subset=['Perubahan (%)'], cmap='RdYlGn')
                )
                
                # Plot prediksi harian
                fig_daily = create_daily_plot(df[date_column], data, predictions, future_dates, date_column, value_column)
                if fig_daily:
                    st.pyplot(fig_daily)
            
            with tab2:
                st.subheader("Prediksi 5 Tahun Kedepan")
                
                # Tampilkan metrik tahunan
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tahun Pertama", format_currency(df_yearly['Prediksi (USD)'].iloc[0]))
                with col2:
                    st.metric("Tahun Ketiga", format_currency(df_yearly['Prediksi (USD)'].iloc[2]))
                with col3:
                    st.metric("Tahun Kelima", format_currency(df_yearly['Prediksi (USD)'].iloc[4]))
                
                # Tampilkan tabel prediksi tahunan
                st.dataframe(
                    df_yearly.style
                    .format({
                        'Prediksi (USD)': format_currency,
                        'Perubahan (USD)': format_currency
                    })
                    .background_gradient(subset=['Perubahan (USD)'], cmap='RdYlGn')
                )
                
                # Plot prediksi tahunan
                fig_yearly = create_yearly_plot(df_yearly['Prediksi (USD)'].values, value_column)
                if fig_yearly:
                    st.pyplot(fig_yearly)
                
                # Tambahkan catatan yang diperbarui
                st.markdown("""
                ### ℹ️ Catatan:
                - Prediksi menggunakan model LSTM yang telah dilatih
                - Hasil prediksi sesuai dengan pola yang dipelajari model
                - Prediksi tahunan dihitung berdasarkan analisis tren prediksi harian
                - Warna hijau menunjukkan perubahan positif
                - Warna merah menunjukkan perubahan negatif
                - Model telah dilatih menggunakan data historis harga emas
                """)
                
    except Exception as e:
        st.error(f"❌ Terjadi error: {str(e)}")
        st.error("Silakan periksa kembali format data Anda")

def main():
    local_css()
    
    # Sidebar navigation
    st.sidebar.title("Navigasi")
    
    # Menggunakan radio untuk navigasi yang lebih baik
    page = st.sidebar.radio(
        "Pilih Halaman:",
        ["HOME", "PREDIKSI", "VISUALISASI MODEL"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='background: linear-gradient(120deg, #FFD700, #FFA500); padding: 10px; border-radius: 5px;'>
        <h3 style='color: white; margin: 0;'>Info Aplikasi</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    <div style='background-color: #FFF8E7; padding: 15px; border-radius: 5px; border-left: 4px solid #FFD700;'>
        <p style='margin: 0; color: #B8860B; font-size: 16px;'><b>Versi:</b> 1.0.0</p>
        <p style='margin: 5px 0 0 0; color: #B8860B; font-size: 16px;'><b>Update Terakhir:</b> 2025</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Page routing
    if page == "HOME":
        home_page()
    elif page == "PREDIKSI":
        prediction_page()
    else:
        visualization_page()

if __name__ == "__main__":
    main() 