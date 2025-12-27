import streamlit as st
import pandas as pd
import re
import numpy as np

# ==========================================
# 1. CONFIGURATION & UTILITY FUNCTIONS
# ==========================================
st.set_page_config(
    page_title="FMCG Recommender System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def mask_product_name(name):
    """
    Masks the product name for privacy/display purposes.
    It takes the first letter of each word until a numeric character is encountered.
    Words containing numbers and subsequent words remain unchanged.
    
    Args:
        name (str): Original product name.
    
    Returns:
        str: Masked product name.
    """
    words = name.split()
    masked_letters = []
    remaining_words = []
    numeric_found = False
    
    for w in words:
        if not numeric_found:
            # Check if word contains a digit
            if re.search(r'\d', w):  
                numeric_found = True
                remaining_words.append(w)
            else:
                # Take only the first letter
                masked_letters.append(w[0]) 
        else:
            remaining_words.append(w)

    masked_part = "".join(masked_letters)
    
    if remaining_words:
        return masked_part + " " + " ".join(remaining_words)
    else:
        return masked_part

# ==========================================
# 2. DATA LOADING FUNCTION
# ==========================================
@st.cache_data
def load_data():
    """
    Loads data chunks from the 'app_data' directory and merges them back 
    into full pandas DataFrames.
    
    Returns:
        tuple: (predictions_df, products_df, history_df)
    """
    # 1. LOAD PREDICTED RATINGS (Split into 6 parts)
    rating_parts = []
    for i in range(1, 7):
        filename = f'app_data/predicted_ratings_part{i}.pkl'
        # compression='gzip' is required as data was saved with this compression
        part = pd.read_pickle(filename, compression='gzip')
        rating_parts.append(part)
    predictions = pd.concat(rating_parts)

    # 2. LOAD USER HISTORY (Check up to 6 parts)
    history_parts = []
    for i in range(1, 7):
        filename = f'app_data/user_history_part{i}.pkl'
        try:
            part = pd.read_pickle(filename, compression='gzip')
            history_parts.append(part)
        except FileNotFoundError:
            continue # Skip if file part doesn't exist
    history = pd.concat(history_parts)

    # 3. LOAD PRODUCT METADATA (Single file)
    products = pd.read_pickle('app_data/product_metadata.pkl')

    return predictions, products, history

# Execute load_data globally to ensure data is available on app start
try:
    with st.spinner('Menyiapkan database sistem...'):
        predicted_ratings_df, full_product, order_cust = load_data()
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()


# ==========================================
# 3. SESSION STATE & NAVIGATION LOGIC
# ==========================================
# Initialize session state for page navigation if it doesn't exist
if 'page' not in st.session_state:
    st.session_state.page = "simulation"

def go_to_docs():
    """Switch state to documentation page."""
    st.session_state.page = "documentation"

def go_to_simulation():
    """Switch state to simulation page."""
    st.session_state.page = "simulation"


# ==========================================
# PAGE 1: DOCUMENTATION (HOW IT WORKS)
# ==========================================
if st.session_state.page == "documentation":
    
    # Back button to return to the main app
    st.button("⬅️ Kembali ke Simulasi", on_click=go_to_simulation)
    
    st.title("📖 Cara Kerja Model & Aplikasi")
    st.markdown("Dokumentasi teknis mengenai metodologi sistem rekomendasi yang digunakan.")
    st.divider()

    # --- SECTION 1: CORE CONCEPT ---
    st.header("1. Konsep Utama: Collaborative Filtering")
    st.info("""
    **Prinsip Dasar:** Sistem memprediksi preferensi customer berdasarkan kemiripan pola transaksi dengan customer lain.
    """)
    
    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        st.markdown("""
        Pendekatan ini berfokus pada **pola interaksi historis**, bukan pada atribut fisik produk.
        
        * **User:** Customer ID
        * **Item:** Material ID (SKU)
        * **Interaksi:** Data Transaksi Pembelian
        """)
    with col_d2:
        st.markdown("""
        **Analogi:**
        Jika **Customer A** membeli produk [X, Y, Z] dan **Customer B** membeli produk [X, Y]...
        
        Sistem mengidentifikasi bahwa Customer B memiliki kemiripan profil dengan Customer A. Oleh karena itu, sistem akan merekomendasikan produk **Z** kepada Customer B, karena produk tersebut belum dibeli namun relevan dengan profilnya.
        """)

    st.divider()

    # --- SECTION 2: ALGORITHM DETAILS ---
    st.header("2. Algoritma: Truncated SVD")
    st.markdown("""
    Model ini dibangun menggunakan metode **Matrix Factorization** dengan algoritma *Truncated Singular Value Decomposition (SVD)*. 
    Berikut adalah tahapan pemrosesan data:
    """)

    # Step 1: Matrix Creation
    st.subheader("Langkah 1: Penyusunan User-Item Matrix")
    st.code("matrix_train = matrix[matrix['customer_id'].isin(train_final['customer_id'].unique())]", language="python")
    st.write("Data transaksi dikonversi menjadi sebuah tabel matriks besar (User-Item Matrix) di mana baris merepresentasikan **Customer** dan kolom merepresentasikan **Produk**.")

    # Step 2: Decomposition
    st.subheader("Langkah 2: Reduksi Dimensi (Decomposition)")
    st.code("svd = TruncatedSVD(n_components=50, random_state=42)", language="python")
    st.write("""
    Matriks awal memiliki dimensi yang sangat besar dan bersifat *sparse* (banyak nilai kosong karena satu customer tidak mungkin membeli semua produk).
    
    SVD memproses matriks ini dengan memecahnya menjadi **50 Komponen Laten (Latent Features)**. Komponen ini merepresentasikan pola tersembunyi atau karakteristik abstrak dari interaksi user dan item.
    """)

    # Step 3: Reconstruction
    st.subheader("Langkah 3: Kalkulasi Skor Prediksi")
    st.code("predicted_ratings = np.dot(matrix_decomposed, svd.components_)", language="python")
    st.write("""
    Melalui perkalian matriks hasil dekomposisi, sistem menghasilkan **Matriks Prediksi**.
    
    Outputnya adalah nilai skor (rating) untuk setiap pasangan Customer dan Produk. Skor ini mengindikasikan tingkat probabilitas atau relevansi produk tersebut bagi customer tertentu.
    """)

    st.divider()

    # --- SECTION 3: INFERENCE FLOW ---
    st.header("3. Alur Proses Rekomendasi pada Aplikasi")
    st.markdown("Saat pengguna meminta rekomendasi, sistem melakukan langkah-langkah berikut secara *real-time*:")
    
    st.success("""
    1.  **Lookup:** Mengidentifikasi data vektor milik Customer ID yang dipilih.
    2.  **Sorting:** Mengurutkan seluruh produk berdasarkan nilai prediksi tertinggi (High Probability).
    3.  **Filtering:** Mengambil sejumlah *N* produk teratas sesuai filter yang diinginkan.
    4.  **Serving:** Menampilkan hasil rekomendasi dalam bentuk tabel interaktif.
    """)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.button("⬅️ Mengerti, Kembali ke Aplikasi", on_click=go_to_simulation, type="primary")


# ==========================================
# PAGE 2: SIMULATION (MAIN APP)
# ==========================================
elif st.session_state.page == "simulation":

    # --- SIDEBAR: FILTER CONTROLS ---
    st.sidebar.header("⚙️ Filter Customer")

    # 1. Customer Selection
    available_users = predicted_ratings_df.index.unique().tolist()
    selected_user_id = st.sidebar.selectbox(
        "1. Pilih Customer ID:", 
        available_users,
        help="Cari atau pilih ID Customer dari daftar."
    )

    # 2. Number of Recommendations Selection
    n_recs = st.sidebar.selectbox(
        "2. Jumlah Rekomendasi:",
        [5, 10, 15, 20, 25],
        index=1,
        help="Tentukan jumlah produk yang ingin ditampilkan."
    )

    st.sidebar.divider()
    
    # --- SIDEBAR: LINK TO DOCS ---
    st.sidebar.markdown("### ℹ️ Informasi Sistem")
    st.sidebar.write("Pelajari metodologi di balik rekomendasi ini.")
    if st.sidebar.button("Pelajari Cara Kerja Model"):
        go_to_docs()
        st.rerun()


    # --- MAIN CONTENT AREA ---
    st.title("Sistem Rekomendasi Produk FMCG")

    # --- USER GUIDE EXPANDER ---
    with st.expander("📖 Panduan Penggunaan Aplikasi", expanded=True):
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            st.markdown("**Langkah 1:**\n👉 Pilih **Customer ID** pada menu di sebelah kiri.")
        with col_g2:
            st.markdown("**Langkah 2:**\n👉 Tentukan *Pilih **Jumlah Rekomendasi** pada menu sebelah kiri.")
        with col_g3:
            st.markdown("**Langkah 3:**\n👉 Klik tombol **'Tampilkan Analisis'** di bawah ini.")

    st.divider()

    # Selected Customer Status
    st.markdown(f"### Analisis untuk Customer ID: `{selected_user_id}`")

    # EXECUTION BUTTON
    if st.button("Tampilkan Analisis & Rekomendasi", type="primary"):
        
        # --- INFERENCE LOGIC ---
        def get_svd_recommendations(customer_id, n=10):
            """
            Retrieves top N recommendations for a specific customer based on SVD predictions.
            """
            if customer_id not in predicted_ratings_df.index: 
                return []
            # Sort predictions descending
            sorted_preds = predicted_ratings_df.loc[customer_id].sort_values(ascending=False)
            # Return Top N indices
            return [str(mid) for mid in sorted_preds.head(n).index]

        # 1. Fetch User History
        user_history_mids = order_cust[order_cust['customer_id'] == selected_user_id]['mid'].unique().tolist()
        
        # 2. Fetch Recommendations
        recs_mids = get_svd_recommendations(selected_user_id, n=n_recs)

        # --- METRICS DISPLAY ---
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Total SKU yang Pernah Diorder", f"{len(user_history_mids)} Item")
        with col_m2:
            st.metric("Jumlah Rekomendasi Produk", f"{len(recs_mids)} Item")
        
        st.markdown("---")

        # --- DATAFRAME DISPLAY ---
        col_left, col_right = st.columns(2)

        # LEFT COLUMN: HISTORY TABLE
        with col_left:
            st.subheader("📦 Riwayat Belanja (History)")
            st.caption("Daftar produk yang **sudah pernah** dibeli oleh Customer ini.")
            
            if user_history_mids:
                # Create DataFrame for history
                history_df = pd.DataFrame({'mid': user_history_mids})
                history_df['mid'] = history_df['mid'].astype(str)
                
                # Merge with metadata
                history_display = history_df.merge(full_product, on='mid', how='left')
                
                # Rename columns and apply masking
                display_df = history_display[['mid', 'mid_desc', 'desc2']].rename(columns={
                    'mid': 'Kode Produk',
                    'mid_desc': 'Nama Produk',
                    'desc2': 'Kategori'
                })
                display_df['Nama Produk'] = display_df['Nama Produk'].apply(mask_product_name)

                # Render Table
                st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)
            else:
                st.info("Customer ini belum memiliki riwayat transaksi.")

        # RIGHT COLUMN: RECOMMENDATION TABLE
        with col_right:
            st.subheader(f"✨ Saran Order (Rekomendasi)")
            st.caption("Daftar produk yang **direkomendasikan** dan memiliki **relevansi tinggi**.")
            
            if recs_mids:
                # Create DataFrame for recommendations
                recs_df = pd.DataFrame({'mid': recs_mids})
                recs_df['mid'] = recs_df['mid'].astype(str)
                
                # Merge with metadata
                recs_display = recs_df.merge(full_product, on='mid', how='left')
                
                # Rename columns and apply masking
                display_recs = recs_display[['mid', 'mid_desc', 'desc2']].rename(columns={
                    'mid': 'Kode Produk',
                    'mid_desc': 'Nama Produk',
                    'desc2': 'Kategori'
                })
                display_recs['Nama Produk'] = display_recs['Nama Produk'].apply(mask_product_name)

                # Render Table
                st.dataframe(display_recs, use_container_width=True, hide_index=True, height=500)
            else:
                st.warning("Data belum cukup untuk memberikan rekomendasi spesifik.")

    else:
        # Initial State (Before button click)
        st.info("👋 Silakan ikuti panduan di atas untuk memulai analisis.")
