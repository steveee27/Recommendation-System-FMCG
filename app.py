import streamlit as st
import pandas as pd
import re
import numpy as np
import pickle

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

# ==========================================
# 2. DATA LOADING FUNCTION (FIXED)
# ==========================================
@st.cache_data
def load_data():
    """
    Loads Two-Tower artifacts: Embeddings (.npy) and ID Mappings (.pkl).
    Also loads user history and product metadata (Single Files).
    """
    # 1. LOAD EMBEDDINGS (Vektor)
    # File .npy sangat cepat dimuat dan ringan
    user_vecs = np.load('app_data/user_embeddings.npy')
    item_vecs = np.load('app_data/item_embeddings.npy')
    
    # 2. LOAD MAPPINGS (Kamus ID)
    with open('app_data/twotower_maps.pkl', 'rb') as f:
        maps = pickle.load(f)
        
    # Buat dictionary agar pencarian cepat: ID -> Index Baris
    # Pastikan ID dikonversi ke string agar konsisten dengan input user
    user_map = {str(uid): i for i, uid in enumerate(maps['user_ids'])}
    
    # Buat dictionary kebalikannya untuk Item: Index Baris -> ID
    item_inv_map = {i: str(mid) for i, mid in enumerate(maps['item_ids'])}

    # 3. LOAD USER HISTORY (SINGLE FILE - FIXED)
    # Sesuai dengan kode save Anda yang tidak memecah file
    history = pd.read_pickle('app_data/user_history.pkl', compression='gzip')

    # 4. LOAD PRODUCT METADATA (Single file)
    products = pd.read_pickle('app_data/product_metadata.pkl')

    return user_vecs, item_vecs, user_map, item_inv_map, products, history

# Execute load_data globally
try:
    with st.spinner('Menyiapkan database sistem (Loading Embeddings)...'):
        user_vecs, item_vecs, user_map, item_inv_map, full_product, order_cust = load_data()
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()


# ==========================================
# 3. SESSION STATE & NAVIGATION LOGIC
# ==========================================
if 'page' not in st.session_state:
    st.session_state.page = "simulation"

def go_to_docs():
    st.session_state.page = "documentation"

def go_to_simulation():
    st.session_state.page = "simulation"


# ==========================================
# PAGE 1: DOCUMENTATION (TWO-TOWER EXPLANATION)
# ==========================================
if st.session_state.page == "documentation":
    
    st.button("⬅️ Kembali ke Simulasi", on_click=go_to_simulation)
    
    st.title("📖 Cara Kerja Model: Two-Tower Architecture")
    st.markdown("Dokumentasi teknis mengenai model *Deep Learning* yang digunakan untuk rekomendasi.")
    st.divider()

    # --- SECTION 1: CORE CONCEPT ---
    st.header("1. Konsep Utama: Neural Collaborative Filtering")
    st.info("""
    **Prinsip Dasar:** Menggunakan dua jaringan saraf (Neural Networks) terpisah untuk mempelajari representasi 'User' dan 'Barang' dalam ruang vektor yang sama.
    """)
    
    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        st.markdown("""
        Berbeda dengan metode statistik biasa, model ini mempelajari **fitur kompleks**:
        
        * **User Tower:** Mempelajari profil customer (ID, demografi, history belanja).
        * **Item Tower:** Mempelajari karakteristik produk (ID, kategori, harga, teks deskripsi).
        """)
    with col_d2:
        st.markdown("""
        **Analogi:**
        Bayangkan biro jodoh yang canggih.
        * **Tower A** mewawancarai Pria (User) untuk memahami kepribadiannya.
        * **Tower B** mewawancarai Wanita (Item) untuk memahami kepribadiannya.
        
        Sistem kemudian memetakan mereka ke dalam "Peta Kepribadian". Jika posisi mereka berdekatan di peta, berarti mereka **cocok** (skor tinggi).
        """)
    st.markdown("")

    st.divider()

    # --- SECTION 2: ALGORITHM DETAILS ---
    st.header("2. Arsitektur: Two-Tower Neural Network")
    st.markdown("Model ini terdiri dari dua menara (*towers*) independen yang bertemu di ujung untuk menghitung skor.")

    # Step 1: Embedding Generation
    st.subheader("Langkah 1: Embedding Layers (Representasi Vektor)")
    st.write("""
    Setiap Customer ID dan Product ID diubah menjadi vektor angka (Embedding). 
    Model ini tidak hanya melihat ID, tapi juga fitur lain (seperti Kategori Produk atau Lokasi User) yang digabungkan (*Concatenate*) menjadi satu vektor informasi yang kaya.
    """)

    # Step 2: Training
    st.subheader("Langkah 2: Proses Training (Dot Product)")
    st.code("""
    # Pseudo-code Arsitektur
    user_embedding = user_tower(user_features)
    item_embedding = item_tower(item_features)
    
    # Menghitung kecocokan
    score = dot_product(user_embedding, item_embedding)
    """, language="python")
    st.write("""
    Tujuan model saat training adalah memaksimalkan skor (Dot Product) untuk pasangan user-item yang benar-benar terjadi (transaksi positif) dan meminimalkannya untuk yang tidak pernah terjadi.
    """)
    st.markdown("")

    # Step 3: Output
    st.subheader("Langkah 3: Output Embeddings")
    st.write("""
    Hasil akhir dari training bukanlah "Tabel Skor" (seperti SVD), melainkan **Kamus Vektor**.
    * Setiap User memiliki koordinat vektor (misal: 16 dimensi).
    * Setiap Produk memiliki koordinat vektor (16 dimensi).
    * Kemiripan dihitung secara *real-time* menggunakan jarak antar vektor ini.
    """)

    st.divider()

    # --- SECTION 3: INFERENCE FLOW ---
    st.header("3. Alur Proses Rekomendasi pada Aplikasi")
    st.markdown("Aplikasi ini menggunakan teknik *Fast Retrieval* tanpa perlu menjalankan model TensorFlow yang berat:")
    
    st.success("""
    1.  **Load Vectors:** Memuat vektor User dan Item yang sudah dihitung sebelumnya (`.npy`).
    2.  **User Lookup:** Mengambil vektor milik Customer yang dipilih.
    3.  **Similarity Search:** Melakukan perkalian matriks (*Dot Product*) antara Vektor User tersebut dengan **SEMUA** Vektor Item.
    4.  **Ranking:** Barang dengan hasil perkalian tertinggi dianggap paling relevan.
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
    # Mengambil list user ID dari user_map dictionary
    available_users = list(user_map.keys())
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
    st.sidebar.write("Pelajari model Two-Tower di balik aplikasi ini.")
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
            st.markdown("**Langkah 2:**\n👉 Pilih **Jumlah Rekomendasi** pada menu sebelah kiri.")
        with col_g3:
            st.markdown("**Langkah 3:**\n👉 Klik tombol **'Tampilkan Analisis'** di bawah ini.")

    st.divider()

    # Selected Customer Status
    st.markdown(f"### Analisis untuk Customer ID: `{selected_user_id}`")

    # EXECUTION BUTTON
    if st.button("Tampilkan Analisis & Rekomendasi", type="primary"):
        
        # --- INFERENCE LOGIC (TWO-TOWER / VECTOR SEARCH) ---
        def get_twotower_recommendations(customer_id, n=10):
            """
            Melakukan pencarian vektor (Dot Product) untuk menemukan item paling mirip dengan user.
            """
            # 1. Cek User ID
            if str(customer_id) not in user_map:
                return []
            
            # 2. Ambil Index & Vektor User
            u_idx = user_map[str(customer_id)]
            target_user_vec = user_vecs[u_idx] # Shape: (Embedding_Dim,)
            
            # 3. Hitung Skor (Dot Product) vs Semua Item
            # item_vecs shape: (N_Items, Embedding_Dim)
            # scores shape: (N_Items,)
            scores = np.dot(item_vecs, target_user_vec)
            
            # 4. Ambil Top-N Index Terbaik
            # np.argsort mengurutkan dari kecil ke besar, ambil n terakhir, lalu balik urutannya
            # Kita ambil lebih banyak dulu (n + 50) untuk jaga-jaga jika ada yg difilter
            top_indices_candidates = np.argsort(scores)[-(n + 100):][::-1]
            
            # 5. Filter & Mapping
            # Filter history untuk menampilkan 'Discovery' (Barang baru saja)
            # Pastikan kolom customer_id di order_cust tipe-nya string agar match
            already_bought = order_cust[order_cust['customer_id'].astype(str) == str(customer_id)]['mid'].unique().tolist()
            already_bought_set = set([str(x) for x in already_bought])
            
            final_recs = []
            for idx in top_indices_candidates:
                mid = item_inv_map[idx]
                
                # JIKA ingin menampilkan hanya barang baru (Discovery):
                if mid not in already_bought_set:
                    final_recs.append(mid)
                
                if len(final_recs) >= n:
                    break
            
            return final_recs

        # 1. Fetch User History
        # Pastikan tipe data sama (string)
        user_history_mids = order_cust[order_cust['customer_id'].astype(str) == str(selected_user_id)]['mid'].unique().tolist()
        
        # 2. Fetch Recommendations (Using Two Tower Logic)
        recs_mids = get_twotower_recommendations(selected_user_id, n=n_recs)

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

                st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)
            else:
                st.info("Customer ini belum memiliki riwayat transaksi.")

        # RIGHT COLUMN: RECOMMENDATION TABLE
        with col_right:
            st.subheader(f"✨ Saran Order (Rekomendasi)")
            st.caption("Daftar produk yang **direkomendasikan** dan memiliki **relevansi tinggi**.")
            
            if recs_mids:
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

                st.dataframe(display_recs, use_container_width=True, hide_index=True, height=500)
            else:
                st.warning("Data belum cukup untuk memberikan rekomendasi spesifik.")

    else:
        # Initial State
        st.info("👋 Silakan ikuti panduan di atas untuk memulai analisis.")
