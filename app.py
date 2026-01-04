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
    Dokumentasi ini menjelaskan dasar konseptual, arsitektur model terbaik, serta mekanisme inferensi berbasis 
    *vector search* yang diterapkan pada sistem rekomendasi Two-Tower Neural Network. 
    Fokus pembahasan mencakup bagaimana representasi pelanggan dan produk dibangun, dilatih, 
    serta dimanfaatkan untuk menghasilkan rekomendasi secara efisien dalam skala besar.
    """)
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📚 Landasan Teori", 
        "🏗️ Arsitektur Model", 
        "⚙️ Algoritma Training",
        "🚀 Mekanisme Inferensi (FAISS)"
    ])

    # TAB 1: LANDASAN TEORI
    with tab1:
        st.header("1. Landasan Teori")
        
        st.write("""
        Sistem rekomendasi ini dibangun menggunakan pendekatan *representation learning*. 
        Berbeda dengan metode *collaborative filtering* tradisional yang hanya memanfaatkan interaksi 
        antara ID pelanggan dan produk, arsitektur **Two-Tower Neural Network** memungkinkan integrasi 
        fitur tambahan (*enriched features*) dari sisi pelanggan maupun produk.
        """)

        st.info("""
        **Prinsip Utama:**
        Pendekatan ini bertujuan untuk mempelajari representasi laten pelanggan dan produk dalam satu ruang vektor bersama, 
        sehingga tingkat relevansi dapat diukur berdasarkan kedekatan vektor di ruang tersebut.
        """)

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.subheader("Keunggulan Arsitektur Two-Tower")
            st.markdown("""
            * **Integrasi fitur heterogen:** Memproses fitur numerik dan kategorikal secara bersamaan.
            * **Menangani cold-start:** Tetap dapat memberikan rekomendasi untuk user/item baru berdasarkan kesamaan fitur, meskipun belum ada riwayat transaksi.
            * **Skalabilitas tinggi:** Proses inferensi sangat cepat karena vektor item dapat dihitung sebelumnya (*pre-computed*).
            """)
        
        with col_t2:
            st.subheader("Formulasi Relevansi")
            st.write("""
            Relevansi antara pelanggan dan produk dihitung menggunakan operasi *dot product* 
            pada embedding masing-masing.
            """)
            st.latex(r'''
            S(u, v) = \langle u, v \rangle = \sum_{i=1}^{d} u_i v_i
            ''')
            st.caption("di mana $u$ adalah embedding pelanggan,  $v$ adalah embedding produk, $d$ adalah dimensi embedding akhir")

    # TAB 2: ARSITEKTUR MODEL
    with tab2:
        st.header("2. Arsitektur Model Two-Tower")
        st.write("""
        Arsitektur Two-Tower yang digunakan merupakan hasil dari proses *hyperparameter tuning* 
        dan menunjukkan performa terbaik pada metrik **Precision@10**, **Recall@10**, **NDCG@10**, dan **Coverage**.
        """)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Embedding Dim", "32", "Dimensi Awal")
            st.metric("Output Embedding", "8", "Dimensi Akhir")
        with c2:
            st.metric("Hidden Layers", "[32, 16, 8]", "Struktur")
            st.metric("Activation", "ReLU", "Fungsi Aktivasi")
        with c3:
            st.metric("Regularization", "Dropout 0.3", "Mencegah Overfitting")
            st.metric("Learning Rate", "0.001", "Adam Optimizer")

        st.divider()
        
        st.subheader("Detail Arsitektur")
        st.markdown("Arsitektur terdiri dari dua neural network independen yang simetris:")

        with st.expander("🅰️ Struktur User Tower", expanded=True):
            st.markdown("""
            1.  **Input Layer:** Menerima `Customer ID` (Embedding) + Fitur Numerik (Normalized) + Fitur Kategori (Embedding).
            2.  **Concatenation:** Penggabungan seluruh fitur user menjadi satu vektor densitas tinggi.
            3.  **Dense Block:**
                * Layer 1: 32 Neuron (ReLU) + Dropout 0.3
                * Layer 2: 16 Neuron (ReLU) + Dropout 0.3
            4.  **Projection Head:** Layer Dense akhir dengan 8 Neuron (ReLU) (menghasilkan vektor $u$).
            """)

        with st.expander("🅱️ Struktur Item Tower", expanded=True):
            st.markdown("""
            1.  **Input Layer:** Menerima `Material ID` (Embedding) + Fitur Numerik (Normalized) + Fitur Kategori (Embedding).
            2.  **Concatenation:** Penggabungan seluruh fitur item menjadi satu vektor densitas tinggi.
            3.  **Dense Block:**
                * Layer 1: 32 Neuron (ReLU) + Dropout 0.3
                * Layer 2: 16 Neuron (ReLU) + Dropout 0.3
            4.  **Projection Head:** Layer Dense akhir dengan 8 Neuron (ReLU) (menghasilkan vektor $v$).
            """)

    # TAB 3: TRAINING
    with tab3:
        st.header("3. Mekanisme Training")

        st.write("""
        Pada tahap pelatihan, model belajar untuk memaksimalkan kesamaan antara pasangan pelanggan–produk 
        yang benar-benar berinteraksi, serta meminimalkan kesamaan untuk pasangan yang tidak relevan.
        """)

        st.code("""
# Interaksi antara embedding pelanggan dan produk
dot_product = Dot(axes=1, normalize=True)([user_embedding, item_embedding])

# Lapisan klasifikasi
x = Dense(8, activation='relu')(dot_product)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

# Kompilasi model
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='binary_crossentropy'
)
        """, language="python")

        st.info("""
        Lapisan `Dot` bertindak sebagai pengukur kesamaan.
        Model dilatih sebagai masalah klasifikasi biner menggunakan *binary cross-entropy* (Interaksi vs Non-Interaksi). 
        """)

    # TAB 4: FAISS
    with tab4:
        st.header("4. Mekanisme Inferensi Berbasis Vector Similarity Search (FAISS)")
        st.write("""
        Untuk mendukung proses rekomendasi yang efisien pada skala besar, sistem ini menggunakan 
        **FAISS (Facebook AI Similarity Search)** sebagai *vector similarity search engine* 
        dengan *IndexFlatIP* untuk melakukan pencarian produk paling relevan berdasarkan *inner product* antar embedding hasil model Two-Tower Neural Network.
        """)

        st.subheader("Tahapan Proses:")
        
        st.markdown("""
        **Tahap 1: Pre-Computation**
        
        Seluruh produk diproses satu kali melalui **Item Tower** untuk menghasilkan embedding produk.
        Proses ini dilakukan secara *offline* dan bersifat *batch*, sehingga tidak membebani proses inferensi.
        """)
        st.code("""
# Generate embedding produk (offline)
item_tower_input = prepare_item_input(all_item_data)
all_item_embeddings = item_tower.predict(item_tower_input, batch_size=4096)
        """, language="python")

        st.markdown("""
        **Tahap 2: Pembuatan FAISS Index**
        
        Embedding produk disimpan ke dalam **FAISS Index** menggunakan 
        **IndexFlatIP (Inner Product)** sebagai metrik kesamaan.
        
        Penggunaan *inner product* sesuai dengan mekanisme perhitungan relevansi 
        pada model Two-Tower yang berbasis *dot product* antar embedding.
        """)
        st.code("""
import faiss

# Inisialisasi Index berdasarkan Inner Product (Dot Product)
dimension = 8  # Sesuai output model
index = faiss.IndexFlatIP(dimension)

# Menambahkan embedding ke index
index.add(all_item_embeddings.astype('float32'))
        """, language="python")

        st.markdown("""
        **Tahap 3: Real-Time Retrieval**
        
        Saat sistem dijalankan:
        1. Data pelanggan diproses melalui **User Tower** untuk menghasilkan embedding pelanggan.
        2. Embedding tersebut digunakan sebagai *query vector* ke FAISS Index.
        3. FAISS mengembalikan sejumlah produk dengan nilai *inner product* tertinggi 
        sebagai hasil rekomendasi.
        """)

        st.success("""
        Pendekatan ini memisahkan proses komputasi offline dan online secara jelas.
        Item Tower hanya dijalankan sekali untuk seluruh katalog produk, 
        sedangkan saat inferensi hanya User Tower yang dieksekusi.
        
        Dengan demikian, sistem mampu menghasilkan rekomendasi secara konsisten 
        dengan model pelatihan dan memiliki latensi yang sangat rendah.
        """)

    st.markdown("<br>", unsafe_allow_html=True)

elif st.session_state.page == "simulation":

    st.sidebar.header("⚙️ Filter Customer")

    available_users = sorted(user_map.keys(), key=lambda x: int(x))
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
    # st.sidebar.write("Pelajari model Two-Tower di balik aplikasi ini.")
    st.sidebar.write("Sistem rekomendasi ini dibangun menggunakan model Two-Tower Neural Network")    
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
