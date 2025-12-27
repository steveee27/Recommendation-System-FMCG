import streamlit as st
import pandas as pd
import re
import numpy as np

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
    Loads data chunks and merges them back.
    """
    # 1. LOAD PREDICTED RATINGS (6 Parts)
    rating_parts = []
    for i in range(1, 7):
        filename = f'app_data/predicted_ratings_part{i}.pkl'
        part = pd.read_pickle(filename, compression='gzip')
        rating_parts.append(part)
    predictions = pd.concat(rating_parts)

    # 2. LOAD USER HISTORY (Up to 6 parts safe check)
    history_parts = []
    for i in range(1, 7):
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

# Load data globally
try:
    with st.spinner('Menyiapkan database sistem...'):
        predicted_ratings_df, full_product, order_cust = load_data()
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()


# --- 3. SESSION STATE & NAVIGATION ---
if 'page' not in st.session_state:
    st.session_state.page = "simulation"

def go_to_docs():
    st.session_state.page = "documentation"

def go_to_simulation():
    st.session_state.page = "simulation"


# ==========================================
# PAGE 1: DOCUMENTATION (CARA KERJA MODEL)
# ==========================================
if st.session_state.page == "documentation":
    
    st.button("⬅️ Kembali ke Simulasi", on_click=go_to_simulation)
    
    st.title("📖 Cara Kerja Model & Aplikasi")
    st.markdown("Dokumentasi teknis tentang bagaimana sistem menghasilkan rekomendasi produk.")
    st.divider()

    # --- BAGIAN 1: KONSEP DASAR ---
    st.header("1. Konsep Utama: Collaborative Filtering")
    st.info("""
    **Prinsip Dasar:** "Customer yang menyukai hal yang sama di masa lalu, cenderung menyukai hal yang sama di masa depan."
    """)
    
    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        st.markdown("""
        Sistem ini tidak melihat *isi* produk (misal: rasa, warna, kemasan), melainkan melihat **pola interaksi/transaksi**.
        
        * **User:** Customer ID (Toko/Warung)
        * **Item:** Product ID (SKU)
        * **Interaksi:** Riwayat Pembelian
        """)
    with col_d2:
        st.markdown("""
        **Analogi Sederhana:**
        Jika **Toko A** membeli [Susu, Kopi, Gula] dan **Toko B** membeli [Susu, Kopi]...
        
        Maka sistem akan berpikir: *"Toko B mirip dengan Toko A. Karena Toko A membeli Gula, maka Toko B kemungkinan besar juga butuh Gula."*
        """)

    st.divider()

    # --- BAGIAN 2: ALGORITMA SVD ---
    st.header("2. Algoritma: Truncated SVD")
    st.markdown("""
    Kami menggunakan metode **Matrix Factorization** dengan algoritma *Truncated Singular Value Decomposition (SVD)*. 
    Berikut adalah tahapan teknis yang terjadi di balik layar (berdasarkan kode training model):
    """)

    # Step 1
    st.subheader("Langkah 1: Pembuatan User-Item Matrix")
    st.code("matrix_train = matrix[matrix['customer_id'].isin(train_final['customer_id'].unique())]", language="python")
    st.write("Sistem menyusun data transaksi menjadi tabel raksasa (Matriks) dimana baris adalah **Customer** dan kolom adalah **Produk**. Isi tabel adalah data interaksi.")

    # Step 2
    st.subheader("Langkah 2: Reduksi Dimensi (Decomposition)")
    st.code("svd = TruncatedSVD(n_components=50, random_state=42)", language="python")
    st.write("""
    Matriks awal sangat besar dan bolong-bolong (sparse). SVD memecah dan memampatkan matriks tersebut menjadi **50 Komponen Tersembunyi (Latent Features)** (`n_components=50`).
    
    * **Apa itu Komponen Tersembunyi?** Ini adalah pola abstrak yang ditemukan komputer. 
    * *Contoh:* Komponen 1 mungkin mewakili "Produk Sarapan", Komponen 2 mewakili "Minuman Dingin", dst.
    """)
    

    # Step 3
    st.subheader("Langkah 3: Prediksi Skor (Reconstruction)")
    st.code("predicted_ratings = np.dot(matrix_decomposed, svd.components_)", language="python")
    st.write("""
    Sistem mengalikan kembali komponen-komponen tersebut untuk menghasilkan **Matriks Prediksi**.
    
    Hasilnya adalah skor (rating) untuk **SETIAP** produk bagi **SETIAP** customer. Skor inilah yang menentukan seberapa besar kemungkinan customer akan membeli produk tersebut.
    """)

    st.divider()

    # --- BAGIAN 3: ALUR APLIKASI ---
    st.header("3. Bagaimana Rekomendasi Muncul di Layar Sales?")
    st.markdown("Saat Anda menekan tombol **'Tampilkan Analisis'**, inilah yang terjadi:")
    
    st.success("""
    1.  **Lookup:** Sistem mencari baris data milik Customer ID yang dipilih dalam tabel `predicted_ratings`.
    2.  **Sorting:** Sistem mengurutkan ribuan produk berdasarkan **skor prediksi tertinggi** ke terendah.
    3.  **Filtering:** Sistem mengambil Top-N produk teratas (misal: 10 produk terbaik).
    4.  **Display:** Kode Produk (ID) diterjemahkan menjadi Nama Produk dan ditampilkan ke tabel.
    """)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.button("⬅️ Paham, Kembali ke Aplikasi", on_click=go_to_simulation, type="primary")


# ==========================================
# PAGE 2: SIMULATION (APLIKASI UTAMA)
# ==========================================
elif st.session_state.page == "simulation":

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("⚙️ Filter Customer")

    # 1. Select Customer
    available_users = predicted_ratings_df.index.unique().tolist()
    selected_user_id = st.sidebar.selectbox(
        "1. Pilih Customer ID:", 
        available_users,
        help="Masukkan atau cari ID Customer yang ingin dianalisis."
    )

    # 2. Select Num Recs
    n_recs = st.sidebar.selectbox(
        "2. Filter Jumlah Saran Order:",
        [5, 10, 15, 20, 25],
        index=1,
        help="Tentukan berapa banyak produk rekomendasi yang ingin ditampilkan."
    )

    st.sidebar.divider()
    
    # NAVIGATION TO DOCS
    st.sidebar.markdown("### ℹ️ Informasi Sistem")
    st.sidebar.write("Ingin tahu bagaimana AI menentukan rekomendasi ini?")
    if st.sidebar.button("Pelajari Cara Kerja Model"):
        go_to_docs()
        st.rerun()


    # --- MAIN CONTENT ---
    st.title("Sistem Rekomendasi Produk FMCG")

    # --- USER GUIDE (PANDUAN) ---
    with st.expander("📖 Panduan Penggunaan Aplikasi", expanded=True):
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            st.markdown("**Langkah 1:**\n👉 Pilih **Customer ID** pada menu sebelah kiri (Sidebar).")
        with col_g2:
            st.markdown("**Langkah 2:**\n👉 Tentukan **Jumlah Saran Order** (misal: 10 item).")
        with col_g3:
            st.markdown("**Langkah 3:**\n👉 Klik tombol **'Tampilkan Analisis'** di bawah ini.")

    st.divider()

    # Header Status
    st.markdown(f"### Analisis untuk Customer ID: `{selected_user_id}`")

    # TOMBOL EKSEKUSI
    if st.button("Tampilkan Analisis & Rekomendasi", type="primary"):
        
        # --- LOGIKA SVD ---
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
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Total SKU Pernah Order", f"{len(user_history_mids)} Item")
        with col_m2:
            st.metric("Saran Produk Baru", f"{len(recs_mids)} Item")
        
        st.markdown("---")

        # --- TAMPILAN TABEL ---
        col_left, col_right = st.columns(2)

        # TABEL KIRI: HISTORY
        with col_left:
            st.subheader("📦 Riwayat Belanja (History)")
            st.caption("Daftar barang yang **sudah biasa** dibeli oleh Customer ini.")
            
            if user_history_mids:
                history_df = pd.DataFrame({'mid': user_history_mids})
                history_df['mid'] = history_df['mid'].astype(str)
                
                # Merge info produk
                history_display = history_df.merge(full_product, on='mid', how='left')
                
                # Rename & Masking
                display_df = history_display[['mid', 'mid_desc', 'desc2']].rename(columns={
                    'mid': 'Kode Produk',
                    'mid_desc': 'Nama Produk',
                    'desc2': 'Kategori'
                })
                display_df['Nama Produk'] = display_df['Nama Produk'].apply(mask_product_name)

                st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)
            else:
                st.info("Customer ini belum memiliki riwayat transaksi.")

        # TABEL KANAN: REKOMENDASI
        with col_right:
            st.subheader(f"✨ Saran Order (Rekomendasi)")
            st.caption("Barang yang **belum pernah dibeli**, tapi diprediksi **relevan** untuk Customer ini.")
            
            if recs_mids:
                recs_df = pd.DataFrame({'mid': recs_mids})
                recs_df['mid'] = recs_df['mid'].astype(str)
                
                # Merge info produk
                recs_display = recs_df.merge(full_product, on='mid', how='left')
                
                # Rename & Masking
                display_recs = recs_display[['mid', 'mid_desc', 'desc2']].rename(columns={
                    'mid': 'Kode Produk',
                    'mid_desc': 'Nama Produk',
                    'desc2': 'Kategori'
                })
                display_recs['Nama Produk'] = display_recs['Nama Produk'].apply(mask_product_name)

                st.dataframe(display_recs, use_container_width=True, hide_index=True, height=500)
            else:
                st.warning("Data belum cukup untuk memberikan rekomendasi spesifik.")

    else:
        # State awal sebelum tombol ditekan
        st.info("👋 Silakan ikuti panduan di atas untuk memulai analisis.")
