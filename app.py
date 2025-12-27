import streamlit as st
import pandas as pd
import re

# --- 1. CONFIGURATION & UTILS ---
st.set_page_config(
    page_title="FMCG Recommender System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def mask_product_name(name):
    """
    Mask product name: take first letter of each word until a number appears.
    """
    words = name.split()
    masked_letters = []
    remaining_words = []
    numeric_found = False
    
    for w in words:
        if not numeric_found:
            if re.search(r'\d', w):  
                numeric_found = True
                remaining_words.append(w)
            else:
                masked_letters.append(w[0]) 
        else:
            remaining_words.append(w)

    masked_part = "".join(masked_letters)
    
    if remaining_words:
        return masked_part + " " + " ".join(remaining_words)
    else:
        return masked_part

# --- 2. DATA LOADING FUNCTION ---
@st.cache_data
def load_data():
    """
    Loads data chunks (split files) and merges them back.
    """
    # 1. LOAD PREDICTED RATINGS (6 Parts)
    rating_parts = []
    for i in range(1, 7):
        filename = f'app_data/predicted_ratings_part{i}.pkl'
        part = pd.read_pickle(filename, compression='gzip')
        rating_parts.append(part)
    predictions = pd.concat(rating_parts)

    # 2. LOAD USER HISTORY (2 Parts based on your request)
    history_parts = []
    for i in range(1, 7): # Adjusted loop to check up to 6 just in case, typical safe approach
        filename = f'app_data/user_history_part{i}.pkl'
        try:
            part = pd.read_pickle(filename, compression='gzip')
            history_parts.append(part)
        except FileNotFoundError:
            continue
    history = pd.concat(history_parts)

    # 3. LOAD METADATA
    products = pd.read_pickle('app_data/product_metadata.pkl')

    return predictions, products, history

# Load data globally to ensure availability
try:
    with st.spinner('Menyiapkan database sistem (Loading & Merging)...'):
        predicted_ratings_df, full_product, order_cust = load_data()
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()


# --- 3. SESSION STATE FOR NAVIGATION ---
if 'page' not in st.session_state:
    st.session_state.page = "home"

def go_to_simulation():
    st.session_state.page = "simulation"

# Sidebar Navigation Logic
st.sidebar.title("Menu Aplikasi")
selection = st.sidebar.radio(
    "Pilih Halaman:", 
    ["🏠 Beranda & Panduan", "🚀 Simulasi Rekomendasi"],
    index=0 if st.session_state.page == "home" else 1
)

if selection == "🏠 Beranda & Panduan":
    st.session_state.page = "home"
else:
    st.session_state.page = "simulation"


# --- 4. PAGE: HOMEPAGE (PENJELASAN) ---
if st.session_state.page == "home":
    st.title("Sistem Rekomendasi Produk FMCG")
    st.markdown("#### *Optimasi Cross-Selling Berbasis Data Historis*")
    
    st.divider()

    col_info1, col_info2 = st.columns([1.5, 1])

    with col_info1:
        st.subheader("📌 Tentang Aplikasi Ini")
        st.write("""
        Aplikasi ini dirancang khusus untuk tim **Sales & Marketing** guna membantu mengidentifikasi peluang penjualan baru (*cross-selling*) kepada pelanggan yang sudah ada (existing customers).
        
        Daripada menebak-nebak produk apa yang harus ditawarkan, sistem ini menganalisis pola belanja ribuan pelanggan untuk memberikan rekomendasi yang presisi.
        """)

        st.subheader("🤖 Bagaimana Cara Kerjanya?")
        st.info("**Metode: Collaborative Filtering (Truncated SVD)**")
        st.write("""
        Bayangkan seorang sales senior yang hafal kebiasaan ribuan pelanggan. Sistem ini bekerja dengan cara serupa menggunakan matematika:
        
        1.  **Mempelajari Pola:** Sistem melihat riwayat transaksi. Jika Toko A membeli *Susu Coklat*, *Roti*, dan *Selai*, dan Toko B membeli *Susu Coklat* dan *Roti*...
        2.  **Mencari Kemiripan:** Sistem mendeteksi bahwa Toko A dan Toko B memiliki selera yang mirip.
        3.  **Memberi Saran:** Sistem akan merekomendasikan *Selai* kepada Toko B, karena Toko A (yang mirip) sudah membelinya.
        
        Teknik ini disebut **Matrix Factorization**, yang mampu menemukan hubungan tersembunyi (*latent features*) antar produk dan pelanggan.
        """)

    with col_info2:
        st.warning("💡 Manfaat untuk Sales")
        st.markdown("""
        * **Personalisasi:** Penawaran sesuai profil toko/customer.
        * **Efisiensi:** Tidak perlu cek manual history satu per satu.
        * **Discovery:** Menemukan produk yang mungkin terlupakan oleh customer tapi potensial laku.
        """)
        
        st.markdown("---")
        st.markdown("##### Siap mencoba?")
        st.write("Klik tombol di bawah atau pilih menu 'Simulasi' di samping.")
        if st.button("Mulai Simulasi Sekarang 🚀", type="primary", use_container_width=True):
            go_to_simulation()
            st.rerun()

# --- 5. PAGE: SIMULATION (APLIKASI UTAMA) ---
elif st.session_state.page == "simulation":
    
    # --- SIDEBAR CONTROLS ---
    st.sidebar.divider()
    st.sidebar.header("⚙️ Konfigurasi")
    
    # 1. Select User
    available_users = predicted_ratings_df.index.unique().tolist()
    selected_user_id = st.sidebar.selectbox(
        "1. Pilih Customer ID:", 
        available_users,
        help="Ketik atau pilih ID Pelanggan dari daftar."
    )

    # 2. Select Num Recs
    n_recs = st.sidebar.selectbox(
        "2. Jumlah Rekomendasi:",
        [5, 10, 15, 20, 25],
        index=1
    )
    
    # --- MAIN CONTENT ---
    st.title("🚀 Simulasi Rekomendasi")
    
    # --- USER GUIDE (PANDUAN PENGGUNAAN) ---
    with st.expander("📖 Panduan Penggunaan (User Guide)", expanded=True):
        col_guide1, col_guide2 = st.columns(2)
        with col_guide1:
            st.markdown("""
            **Langkah 1: Pilih Customer**
            👈 Lihat menu di sebelah kiri (Sidebar).
            * Cari ID Customer pada kolom **"Pilih Customer ID"**.
            * Ini adalah target customer yang ingin Anda analisis.
            """)
        with col_guide2:
            st.markdown("""
            **Langkah 2: Tentukan Jumlah**
            👈 Lihat menu di sebelah kiri.
            * Gunakan **"Jumlah Rekomendasi"** untuk mengatur berapa banyak produk yang ingin ditampilkan (Top 5, 10, dst).
            """)
        st.caption("Setelah memilih, klik tombol 'Generate Recommendations' di bawah ini untuk melihat hasil.")

    st.markdown(f"### Analisis untuk Customer ID: `{selected_user_id}`")

    # Tombol Eksekusi
    if st.button("Generate Recommendations", type="primary"):
        
        # LOGIKA SVD
        def get_svd_recommendations(customer_id, n=10):
            if customer_id not in predicted_ratings_df.index: 
                return []
            sorted_preds = predicted_ratings_df.loc[customer_id].sort_values(ascending=False)
            return [str(mid) for mid in sorted_preds.head(n).index]

        # 1. Ambil Data History
        user_history_mids = order_cust[order_cust['customer_id'] == selected_user_id]['mid'].unique().tolist()
        
        # 2. Ambil Rekomendasi
        recs_mids = get_svd_recommendations(selected_user_id, n=n_recs)

        # --- TAMPILAN METRICS ---
        st.markdown("---")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Total Produk Pernah Dibeli", f"{len(user_history_mids)} Item")
        with col_m2:
            st.metric("Rekomendasi Dihasilkan", f"{len(recs_mids)} Item")
        st.markdown("---")

        # --- TAMPILAN TABEL ---
        col_left, col_right = st.columns(2)

        # TABEL KIRI: HISTORY
        with col_left:
            st.subheader("📜 Riwayat Pembelian (Actual)")
            st.caption("Barang yang **sudah pernah** dibeli oleh toko ini sebelumnya.")
            
            if user_history_mids:
                history_df = pd.DataFrame({'mid': user_history_mids})
                history_df['mid'] = history_df['mid'].astype(str)
                
                # Merge info produk
                history_display = history_df.merge(full_product, on='mid', how='left')
                
                # Rename & Masking
                display_df = history_display[['mid', 'mid_desc', 'desc2']].rename(columns={
                    'mid': 'Product ID',
                    'mid_desc': 'Nama Produk',
                    'desc2': 'Kategori/Detail'
                })
                display_df['Nama Produk'] = display_df['Nama Produk'].apply(mask_product_name)

                st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)
            else:
                st.info("User ini tidak memiliki riwayat transaksi di dataset.")

        # TABEL KANAN: REKOMENDASI
        with col_right:
            st.subheader(f"✨ Top {len(recs_mids)} Rekomendasi (Prediction)")
            st.caption("Barang yang **diprediksi akan disukai** berdasarkan pola kemiripan.")
            
            if recs_mids:
                recs_df = pd.DataFrame({'mid': recs_mids})
                recs_df['mid'] = recs_df['mid'].astype(str)
                
                # Merge info produk
                recs_display = recs_df.merge(full_product, on='mid', how='left')
                
                # Rename & Masking
                display_recs = recs_display[['mid', 'mid_desc', 'desc2']].rename(columns={
                    'mid': 'Product ID',
                    'mid_desc': 'Nama Produk',
                    'desc2': 'Kategori/Detail'
                })
                display_recs['Nama Produk'] = display_recs['Nama Produk'].apply(mask_product_name)

                st.dataframe(display_recs, use_container_width=True, hide_index=True, height=500)
            else:
                st.warning("Cold Start: Sistem belum memiliki cukup data untuk memberikan rekomendasi pada user ini.")

    else:
        st.info("👈 Silakan atur konfigurasi di sidebar kiri, lalu klik tombol **'Generate Recommendations'**.")
