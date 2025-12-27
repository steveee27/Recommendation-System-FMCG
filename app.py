import streamlit as st
import pandas as pd
import re

# --- 1. CONFIGURATION & UTILS ---
st.set_page_config(
    page_title="Sales Assist: FMCG General Trade",
    page_icon="🛒",
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
    with st.spinner('Menyiapkan database outlet & produk...'):
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
    "Navigasi:", 
    ["🏠 Beranda (Home)", "📋 Cek Toko & Rekomendasi"],
    index=0 if st.session_state.page == "home" else 1
)

if selection == "🏠 Beranda (Home)":
    st.session_state.page = "home"
else:
    st.session_state.page = "simulation"


# --- 4. PAGE: HOMEPAGE (PENJELASAN KHUSUS SALES) ---
if st.session_state.page == "home":
    # Header yang lebih spesifik ke use-case Sales GT
    st.title("Sistem Pendukung Keputusan: FMCG General Trade")
    st.markdown("### *Alat Bantu Sales untuk Analisis History & Cross-Selling Outlet*")
    
    st.divider()

    col_info1, col_info2 = st.columns([1.5, 1])

    with col_info1:
        st.subheader("📌 Fungsi Aplikasi untuk Salesman")
        st.write("""
        Aplikasi ini dirancang untuk membantu Tim Sales saat melakukan kunjungan (*visiting*) ke outlet General Trade (GT). 
        Tujuan utamanya adalah memastikan Sales tidak hanya mencatat order rutin, tetapi juga menawarkan produk baru yang **potensial laku** di toko tersebut.
        
        **Apa yang bisa Anda lakukan di sini?**
        1.  **Cek Riwayat Belanja (Purchase History):** Melihat kembali apa yang biasa dibeli oleh toko tersebut agar tidak ada SKU rutin yang terlewat (Pareto Item).
        2.  **Dapatkan Saran Order (Recommendation):** Mendapatkan daftar produk yang *belum* pernah dibeli toko tersebut, namun diprediksi akan laku berdasarkan data penjualan toko-toko lain yang serupa.
        """)

        st.info("""
        **🔍 Cara Kerja Sistem (Collaborative Filtering):**
        Sistem ini menggunakan algoritma cerdas yang membandingkan profil belanja satu toko dengan ribuan toko lain.
        
        *Contoh:* Jika **Toko A** dan **Toko B** memiliki kemiripan pola belanja, namun Toko B belum membeli *Produk Baru X* yang laris di Toko A, maka sistem akan menyarankan Sales untuk menawarkan *Produk X* tersebut ke Toko B.
        """)

    with col_info2:
        st.success("💡 Target Bisnis")
        st.markdown("""
        * **Meningkatkan SKU Aktif:** Menambah variasi produk yang dijual di outlet.
        * **Cross-Selling:** Menawarkan produk komplementer yang relevan.
        * **Efisiensi Kunjungan:** Sales memiliki bahan percakapan berbasis data saat negosiasi dengan pemilik toko.
        """)
        
        st.markdown("---")
        st.markdown("##### Mulai Kunjungan?")
        st.write("Klik tombol di bawah untuk masuk ke menu simulasi outlet.")
        if st.button("Mulai Analisis Outlet 🚀", type="primary", use_container_width=True):
            go_to_simulation()
            st.rerun()

# --- 5. PAGE: SIMULATION (APLIKASI UTAMA) ---
elif st.session_state.page == "simulation":
    
    # --- SIDEBAR CONTROLS ---
    st.sidebar.divider()
    st.sidebar.header("⚙️ Filter Outlet")
    
    # 1. Select User
    available_users = predicted_ratings_df.index.unique().tolist()
    selected_user_id = st.sidebar.selectbox(
        "1. Pilih Kode Outlet / Customer ID:", 
        available_users,
        help="Masukkan kode outlet yang sedang dikunjungi."
    )

    # 2. Select Num Recs
    n_recs = st.sidebar.selectbox(
        "2. Jumlah Saran Order (SKU):",
        [5, 10, 15, 20, 25],
        index=1,
        help="Berapa banyak produk rekomendasi yang ingin ditampilkan."
    )
    
    # --- MAIN CONTENT ---
    st.title("📋 Cek Toko & Rekomendasi")
    
    # --- USER GUIDE (PANDUAN VISUAL) ---
    with st.expander("📖 Panduan Penggunaan Sales (Klik untuk menutup)", expanded=True):
        col_guide1, col_guide2 = st.columns(2)
        with col_guide1:
            st.markdown("""
            **Langkah 1: Pilih Outlet**
            👈 Pada menu kiri, masukkan **Kode Outlet** (Customer ID) yang sedang Anda kunjungi.
            """)
        with col_guide2:
            st.markdown("""
            **Langkah 2: Generate Data**
            Klik tombol **'Tampilkan Analisis Outlet'** di bawah ini untuk melihat History Belanja & Saran Order.
            """)

    st.markdown(f"### Outlet yang Sedang Dianalisis: `{selected_user_id}`")

    # Tombol Eksekusi
    if st.button("Tampilkan Analisis Outlet", type="primary"):
        
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
            st.metric("Total SKU Pernah Order", f"{len(user_history_mids)} Item")
        with col_m2:
            st.metric("Potensi SKU Baru (Rekomendasi)", f"{len(recs_mids)} Item")
        st.markdown("---")

        # --- TAMPILAN TABEL ---
        col_left, col_right = st.columns(2)

        # TABEL KIRI: HISTORY
        with col_left:
            st.subheader("📦 History Belanja (Rutin)")
            st.caption("Daftar barang yang **sudah biasa** dibeli oleh toko ini. Pastikan stok aman.")
            
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
                st.info("Outlet ini belum memiliki riwayat transaksi (New Outlet).")

        # TABEL KANAN: REKOMENDASI
        with col_right:
            st.subheader(f"✨ Saran Order Baru (Cross-Sell)")
            st.caption("Barang ini **belum pernah dibeli**, tapi diprediksi **LAKU** di toko ini. Tawarkan ini!")
            
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
                st.warning("Data belum cukup untuk memberikan rekomendasi spesifik pada outlet ini.")

    else:
        st.info("👈 Silakan pilih Kode Outlet di menu kiri, lalu klik tombol **'Tampilkan Analisis Outlet'**.")
