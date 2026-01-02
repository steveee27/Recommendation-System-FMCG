import streamlit as st
import pandas as pd
import re
import numpy as np
import pickle

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

@st.cache_data
def load_data():
    """
    Loads Two-Tower artifacts: Embeddings (.npy) and ID Mappings (.pkl).
    Also loads user history and product metadata (Single Files).
    """
    user_vecs = np.load('app_data/user_embeddings.npy')
    item_vecs = np.load('app_data/item_embeddings.npy')
    
    with open('app_data/twotower_maps.pkl', 'rb') as f:
        maps = pickle.load(f)
        
    user_map = {str(uid): i for i, uid in enumerate(maps['user_ids'])}
    item_inv_map = {i: str(mid) for i, mid in enumerate(maps['item_ids'])}

    history = pd.read_pickle('app_data/user_history.pkl', compression='gzip')
    products = pd.read_pickle('app_data/product_metadata.pkl')

    return user_vecs, item_vecs, user_map, item_inv_map, products, history

try:
    with st.spinner('Menyiapkan database sistem (Loading Embeddings)...'):
        user_vecs, item_vecs, user_map, item_inv_map, full_product, order_cust = load_data()
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

if 'page' not in st.session_state:
    st.session_state.page = "simulation"

def go_to_docs():
    st.session_state.page = "documentation"

def go_to_simulation():
    st.session_state.page = "simulation"

if st.session_state.page == "documentation":
    st.button("⬅️ Kembali ke Simulasi", on_click=go_to_simulation)
    
    st.title("Dokumentasi Teknis: Sistem Rekomendasi Two-Tower")
    st.markdown("""
    Dokumentasi ini menguraikan landasan teoritis, spesifikasi arsitektur terbaik, 
    dan mekanisme inferensi *high-speed retrieval* yang diimplementasikan dalam sistem rekomendasi ini.
    """)
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📚 Landasan Teori", 
        "🏗️ Arsitektur Model", 
        "⚙️ Algoritma Training",
        "🚀 Mekanisme Inferensi (FAISS)"
    ])

    with tab1:
        st.header("1. Konsep Dasar:")
        
        st.write("""
        Sistem ini dibangun di atas paradigma *Representation Learning*. Berbeda dengan pendekatan *Collaborative Filtering* tradisional (seperti *Matrix Factorization*) 
        yang hanya mengandalkan interaksi ID pengguna dan item, arsitektur **Two-Tower** memungkinkan integrasi **fitur sampingan (*side features*)** yang kaya.
        """)

        st.info("""
        **Prinsip Kerja Utama:**
        Sistem bertujuan memetakan entitas **Pengguna (User)** dan **Barang (Item)** ke dalam **Ruang Vektor Laten Bersama (*Shared Latent Vector Space*)**.
        Dalam ruang ini, relevansi diukur berdasarkan kedekatan geometris antar vektor.
        """)

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.subheader("Keunggulan Arsitektur")
            st.markdown("""
            * **Hybrid Input:** Mampu memproses data numerik (misal: *recency*) dan kategorikal (misal: *brand*) secara bersamaan.
            * **Cold-Start Handling:** Tetap dapat memberikan rekomendasi untuk user/item baru berdasarkan kesamaan fitur, meskipun belum ada riwayat transaksi.
            * **Scalability:** Proses inferensi sangat cepat karena vektor item dapat dihitung sebelumnya (*pre-computed*).
            """)
        
        with col_t2:
            st.subheader("Formulasi Matematis")
            st.write("Skor relevansi ($S$) didefinisikan sebagai *Dot Product* antara vektor user ($u$) dan item ($v$):")
            st.latex(r'''
            S(u, v) = \langle u, v \rangle = \sum_{i=1}^{d} u_i v_i
            ''')
            st.caption("Dimana $d$ adalah dimensi embedding akhir (8 dimensi).")

    with tab2:
        st.header("2. Spesifikasi Arsitektur Two Tower")
        st.write("""
        Berdasarkan eksperimen penelitian, arsitektur berikut adalah arsitektur terbaik dari hasil hyperparameter tuning dengan memberikan keseimbangan optimal 
        antara metrik **Precision@10**, **Recall@10**, **NDCG@10**, dan **Coverage**.
        """)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Embedding Dim", "32", "Latent Size")
            st.metric("Output Dim", "8", "Final Vector")
        with c2:
            st.metric("Hidden Layers", "[32, 16, 8]", "Deep Structure")
            st.metric("Activation", "ReLU", "Non-Linearity")
        with c3:
            st.metric("Regularization", "Dropout 0.3", "Mencegah Overfitting")
            st.metric("Learning Rate", "0.001", "Adam Optimizer")

        st.divider()
        
        st.subheader("Detail Implementasi Layer")
        st.markdown("Arsitektur terdiri dari dua neural network independen (*Twin Towers*) yang simetris:")

        with st.expander("🅰️ User Tower Specification", expanded=True):
            st.markdown("""
            1.  **Input Layer:** Menerima `Customer ID` (Embedding) + Fitur Numerik (Normalized) + Fitur Kategori (Embedding).
            2.  **Concatenation:** Penggabungan seluruh fitur user menjadi satu vektor densitas tinggi.
            3.  **Dense Block:**
                * Layer 1: 32 Neuron (ReLU) + Dropout 0.3
                * Layer 2: 16 Neuron (ReLU) + Dropout 0.3
            4.  **Projection Head:** Layer Dense akhir dengan 8 Neuron (menghasilkan vektor $u$).
            """)

        with st.expander("🅱️ Item Tower Specification", expanded=True):
            st.markdown("""
            1.  **Input Layer:** Menerima `Material ID` (Embedding) + Fitur Numerik (Normalized) + Fitur Kategori (Embedding).
            2.  **Concatenation:** Penggabungan seluruh fitur item menjadi satu vektor densitas tinggi.
            3.  **Dense Block:**
                * Layer 1: 32 Neuron (ReLU) + Dropout 0.3
                * Layer 2: 16 Neuron (ReLU) + Dropout 0.3
            4.  **Projection Head:** Layer Dense akhir dengan 8 Neuron (menghasilkan vektor $u$).
            """)

    with tab3:
        st.header("3. Mekanisme Pembelajaran (Training)")
        st.write("""
        Selama fase pelatihan, model belajar untuk mendekatkan vektor pengguna dan item yang memiliki interaksi positif, 
        serta menjauhkan mereka yang tidak berinteraksi.
        """)

        st.code("""
# Snippet Logika Training (Keras Functional API)

# 1. Forward Pass
user_vec = user_tower(user_inputs)
item_vec = item_tower(item_inputs)

# 2. Interaction Layer (Dot Product)
# Menghitung kesamaan kosinus/dot product
dot_interaction = Dot(axes=1, normalize=True)([user_vec, item_vec])

# 3. Output Layer (Probability)
# Activation Sigmoid mengubah skor menjadi probabilitas (0-1)
output = Dense(1, activation='sigmoid')(dot_interaction)

# 4. Optimization
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))
        """, language="python")

        st.info("""
        **Penjelasan:**
        Lapisan `Dot` bertindak sebagai pengukur kesamaan. Fungsi aktivasi `Sigmoid` digunakan karena kita memodelkan masalah ini 
        sebagai *Binary Classification* (Interaksi vs Non-Interaksi).
        """)

    # --- TAB 4: MEKANISME FAISS ---
    with tab4:
        st.header("4. Mekanisme Inferensi: Vector Search Engine")
        st.write("""
        Untuk kebutuhan aplikasi *real-time* dengan latensi rendah, sistem ini **tidak melakukan prediksi model satu-per-satu**.
        Sistem menggunakan pendekatan **Approximate Nearest Neighbor (ANN)** menggunakan pustaka **FAISS (Facebook AI Similarity Search)**.
        """)

        st.subheader("Tahapan Proses:")
        
        st.markdown("""
        **Tahap 1: Pre-Computation (Batch Processing)**
        
        Seluruh katalog produk diproses melalui *Item Tower* untuk menghasilkan vektor embedding statis. Proses ini dilakukan dalam *batch* besar untuk efisiensi.
        """)
        st.code("""
# Generate embeddings untuk seluruh katalog (sekali jalan)
item_tower_input = prepare_item_input(all_item_data)
all_item_embeddings = item_tower.predict(item_tower_input, batch_size=4096)
        """, language="python")

        st.markdown("""
        **Tahap 2: Indexing (FAISS Construction)**
        
        Vektor-vektor tersebut disimpan ke dalam struktur data indeks. Kami menggunakan `IndexFlatIP` (*Inner Product*) 
        yang secara matematis ekuivalen dengan operasi *Dot Product* pada model.
        """)
        st.code("""
import faiss

# Inisialisasi Index berdasarkan Inner Product (Dot Product)
dimension = 8  # Sesuai output model
index = faiss.IndexFlatIP(dimension)

# Menambahkan vektor ke memori
index.add(all_item_embeddings.astype('float32'))
        """, language="python")

        st.markdown("""
        **Tahap 3: Real-Time Retrieval**
        
        Saat simulasi dijalankan:
        1. Data user diproses oleh *User Tower* dan menghasilkan 1 vektor user.
        2. Vektor user digunakan untuk *query* ke FAISS Index.
        3. FAISS mengembalikan $K$ item dengan nilai *Inner Product* tertinggi.
        """)

        st.success("""
        **Implikasi:**
        Metode ini memisahkan beban komputasi. *Item Tower* yang berat hanya dijalankan sekali (offline), 
        sedangkan saat *serving* hanya *User Tower* yang aktif, membuat respon aplikasi sangat cepat (< 50ms).
        """)

    st.markdown("<br>", unsafe_allow_html=True)

elif st.session_state.page == "simulation":

    st.sidebar.header("⚙️ Filter Customer")

    available_users = list(user_map.keys())
    selected_user_id = st.sidebar.selectbox(
        "1. Pilih Customer ID:", 
        available_users,
        help="Cari atau pilih ID Customer dari daftar."
    )

    n_recs = st.sidebar.selectbox(
        "2. Jumlah Rekomendasi:",
        [5, 10, 15, 20, 25],
        index=1,
        help="Tentukan jumlah produk yang ingin ditampilkan."
    )

    st.sidebar.divider()
    
    st.sidebar.markdown("### ℹ️ Informasi Sistem")
    st.sidebar.write("Pelajari model Two-Tower di balik aplikasi ini.")
    if st.sidebar.button("Pelajari Cara Kerja Model"):
        go_to_docs()
        st.rerun()

    st.title("Sistem Rekomendasi Produk FMCG")

    with st.expander("📖 Panduan Penggunaan Aplikasi", expanded=True):
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            st.markdown("**Langkah 1:**\n👉 Pilih **Customer ID** pada menu di sebelah kiri.")
        with col_g2:
            st.markdown("**Langkah 2:**\n👉 Pilih **Jumlah Rekomendasi** pada menu sebelah kiri.")
        with col_g3:
            st.markdown("**Langkah 3:**\n👉 Klik tombol **'Tampilkan Analisis'** di bawah ini.")

    st.divider()

    st.markdown(f"### Analisis untuk Customer ID: `{selected_user_id}`")

    if st.button("Tampilkan Analisis & Rekomendasi", type="primary"):
        def get_twotower_recommendations(customer_id, n=10):
            """
            Melakukan pencarian vektor (Dot Product) untuk menemukan item paling mirip dengan user.
            """
            if str(customer_id) not in user_map:
                return []
            
            u_idx = user_map[str(customer_id)]
            target_user_vec = user_vecs[u_idx] # Shape: (Embedding_Dim,)
            
            scores = np.dot(item_vecs, target_user_vec)
            
            top_indices_candidates = np.argsort(scores)[-(n + 100):][::-1]
            
            already_bought = order_cust[order_cust['customer_id'].astype(str) == str(customer_id)]['mid'].unique().tolist()
            already_bought_set = set([str(x) for x in already_bought])
            
            final_recs = []
            for idx in top_indices_candidates:
                mid = item_inv_map[idx]
                if mid not in already_bought_set:
                    final_recs.append(mid)
                
                if len(final_recs) >= n:
                    break
            
            return final_recs

        user_history_mids = order_cust[order_cust['customer_id'].astype(str) == str(selected_user_id)]['mid'].unique().tolist()
        
        recs_mids = get_twotower_recommendations(selected_user_id, n=n_recs)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Total SKU yang Pernah Diorder", f"{len(user_history_mids)} Item")
        with col_m2:
            st.metric("Jumlah Rekomendasi Produk", f"{len(recs_mids)} Item")
        
        st.markdown("---")

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("📦 Riwayat Belanja (History)")
            st.caption("Daftar produk yang **sudah pernah** dibeli oleh Customer ini.")
            
            if user_history_mids:
                history_df = pd.DataFrame({'mid': user_history_mids})
                history_df['mid'] = history_df['mid'].astype(str)
                
                history_display = history_df.merge(full_product, on='mid', how='left')
                
                display_df = history_display[['mid', 'mid_desc', 'desc2']].rename(columns={
                    'mid': 'Kode Produk',
                    'mid_desc': 'Nama Produk',
                    'desc2': 'Kategori'
                })
                display_df['Nama Produk'] = display_df['Nama Produk'].apply(mask_product_name)

                st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)
            else:
                st.info("Customer ini belum memiliki riwayat transaksi.")

        with col_right:
            st.subheader(f"✨ Saran Order (Rekomendasi)")
            st.caption("Daftar produk yang **direkomendasikan** dan memiliki **relevansi tinggi**.")
            
            if recs_mids:
                recs_df = pd.DataFrame({'mid': recs_mids})
                recs_df['mid'] = recs_df['mid'].astype(str)
                
                recs_display = recs_df.merge(full_product, on='mid', how='left')
                
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
        st.info("👋 Silakan ikuti panduan di atas untuk memulai analisis.")
