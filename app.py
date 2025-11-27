import streamlit as st
import pandas as pd

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sistem Rekomendasi Produk", layout="wide")

# --- FUNGSI LOAD DATA (Dichache agar cepat) ---
@st.cache_data
def load_data():
    """
    Fungsi ini memuat data yang terpecah-pecah (chunks) 
    dan menggabungkannya kembali menjadi dataframe utuh.
    """
    
    # 1. LOAD PREDICTED RATINGS (Terpecah jadi 6 file)
    rating_parts = []
    # Loop dari 1 sampai 6 (sesuai jumlah file di screenshot)
    for i in range(1, 7):
        filename = f'app_data/predicted_ratings_part{i}.pkl'
        # compression='gzip' harus ada jika saat save menggunakan gzip
        part = pd.read_pickle(filename, compression='gzip')
        rating_parts.append(part)
    
    # Gabungkan kembali menjadi satu dataframe besar
    predictions = pd.concat(rating_parts)

    # 2. LOAD USER HISTORY (Terpecah jadi 6 file)
    history_parts = []
    for i in range(1, 7):
        filename = f'app_data/user_history_part{i}.pkl'
        # Cek try-except jaga-jaga jika file history kurang dari 6
        try:
            part = pd.read_pickle(filename, compression='gzip')
            history_parts.append(part)
        except FileNotFoundError:
            continue # Lanjut jika file tidak ditemukan
            
    # Gabungkan kembali
    history = pd.concat(history_parts)

    # 3. LOAD PRODUCT METADATA (Single file, tidak dipecah)
    products = pd.read_pickle('app_data/product_metadata.pkl')

    return predictions, products, history

# --- PROSES MEMUAT DATA ---
try:
    with st.spinner('Sedang menyiapkan data (Unpacking & Merging)...'):
        predicted_ratings_df, full_product, order_cust = load_data()
    st.success("✅ Data berhasil dimuat!")
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat data: {e}")
    st.stop()

# --- FUNGSI REKOMENDASI (Logika SVD) ---
def get_svd_recommendations(customer_id, n_recs=10):
    if customer_id not in predicted_ratings_df.index: 
        return []
    # Ambil row user tersebut, urutkan dari nilai tertinggi
    sorted_preds = predicted_ratings_df.loc[customer_id].sort_values(ascending=False)
    # Kembalikan list Product ID (mid)
    return [str(mid) for mid in sorted_preds.head(n_recs).index]

# --- USER INTERFACE (UI) ---
st.title("🛍️ Simulasi Rekomendasi Produk (SVD)")

# 1. Sidebar untuk memilih User
st.sidebar.header("Pilih Customer")

# Ambil list semua customer ID yang ada di data prediksi
available_users = predicted_ratings_df.index.unique().tolist()
selected_user_id = st.sidebar.selectbox("Masukkan/Pilih Customer ID:", available_users)

if st.sidebar.button("Generate Rekomendasi"):
    
    # --- PROSES UTAMA ---
    col1, col2 = st.columns(2)
    
    # A. Tampilkan History User (Apa yang pernah dibeli)
    with col1:
        st.subheader("📜 Riwayat Pembelian User")
        
        # Filter history berdasarkan user yang dipilih
        user_history_mids = order_cust[order_cust['customer_id'] == selected_user_id]['mid'].unique().tolist()
        
        if user_history_mids:
            # Join dengan tabel produk untuk dapat nama barang
            history_df = pd.DataFrame({'mid': user_history_mids})
            history_df['mid'] = history_df['mid'].astype(str) # Pastikan tipe data sama
            
            # Merge untuk ambil deskripsi
            history_display = history_df.merge(
                full_product, 
                on='mid', 
                how='left'
            )
            st.dataframe(history_display[['mid', 'mid_desc', 'desc2']], use_container_width=True)
        else:
            st.info("User ini belum memiliki riwayat pembelian di data history.")

    # B. Tampilkan Rekomendasi SVD
    with col2:
        st.subheader("✨ Top 10 Rekomendasi (SVD)")
        
        recs_mids = get_svd_recommendations(selected_user_id, n_recs=10)
        
        if recs_mids:
            recs_df = pd.DataFrame({'mid': recs_mids})
            recs_df['mid'] = recs_df['mid'].astype(str)
            
            # Merge dengan info produk
            recs_display = recs_df.merge(
                full_product, 
                on='mid', 
                how='left'
            )
            
            # Tampilkan
            st.dataframe(recs_display[['mid', 'mid_desc', 'desc2']], use_container_width=True)
        else:
            st.warning("User ini tidak ditemukan dalam data training SVD (Cold Start).")
