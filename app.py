import streamlit as st
import pandas as pd
import re

def mask_product_name(name):
    """
    Convert product name by taking the first alphabet of each word 
    until the first numeric-containing word. After that, keep words unchanged.
    
    Example:
    MILKU ORIGINAL 200 ML ISI 12 -> MO 200 ML ISI 12
    MILKU COKLAT PREMIUM 200 ML -> MCP 200 ML
    FLORIDINA ORANGE BTL 350ML HRG PROMO -> FOB 350ML HRG PROMO
    NOODLE SEDAAP MIE CUP KARI MERCON 79GR -> NSMCKM 79GR
    """
    
    words = name.split()
    masked_letters = []
    remaining_words = []
    numeric_found = False
    
    for w in words:
        if not numeric_found:
            if re.search(r'\d', w):  
                # word contains a number → stop masking
                numeric_found = True
                remaining_words.append(w)
            else:
                masked_letters.append(w[0])  # take first letter
        else:
            remaining_words.append(w)

    masked_part = "".join(masked_letters)
    
    if remaining_words:
        return masked_part + " " + " ".join(remaining_words)
    else:
        return masked_part


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Product Recommender System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD DATA FUNCTION (Cached for performance) ---
@st.cache_data
def load_data():
    """
    Loads data chunks (split files) and merges them back into full dataframes.
    """
    
    # 1. LOAD PREDICTED RATINGS (Split into 6 files)
    rating_parts = []
    # Loop from 1 to 6
    for i in range(1, 7):
        filename = f'app_data/predicted_ratings_part{i}.pkl'
        # compression='gzip' is required as we saved it using gzip
        part = pd.read_pickle(filename, compression='gzip')
        rating_parts.append(part)
    
    # Merge back into one large dataframe
    predictions = pd.concat(rating_parts)

    # 2. LOAD USER HISTORY (Split into 2 files)
    history_parts = []
    for i in range(1, 3):
        filename = f'app_data/user_history_part{i}.pkl'
        # Try-except block in case history files are fewer than 2
        try:
            part = pd.read_pickle(filename, compression='gzip')
            history_parts.append(part)
        except FileNotFoundError:
            continue # Skip if file not found
            
    # Merge back
    history = pd.concat(history_parts)

    # 3. LOAD PRODUCT METADATA (Single file)
    products = pd.read_pickle('app_data/product_metadata.pkl')

    return predictions, products, history

# --- LOADING PROCESS ---
try:
    with st.spinner('Loading data (Unpacking & Merging)...'):
        predicted_ratings_df, full_product, order_cust = load_data()
    # Toast notification is less intrusive than a big green bar
    st.toast("✅ Data loaded successfully!", icon="🚀")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- RECOMMENDATION LOGIC (SVD) ---
def get_svd_recommendations(customer_id, n_recs=10):
    if customer_id not in predicted_ratings_df.index: 
        return []
    # Get user row, sort by highest predicted rating
    sorted_preds = predicted_ratings_df.loc[customer_id].sort_values(ascending=False)
    # Return list of Product IDs (mid)
    return [str(mid) for mid in sorted_preds.head(n_recs).index]

# --- USER INTERFACE (UI) ---

# Sidebar
with st.sidebar:
    st.title("🛒 Control Panel")
    
    st.markdown("### 1. Select User")
    available_users = predicted_ratings_df.index.unique().tolist()
    selected_user_id = st.selectbox(
        "Search or Select Customer ID:", 
        available_users,
        help="Type to search for a specific user ID"
    )

    st.markdown("### 2. Number of Recommendations")
    n_recs = st.selectbox(
        "Select how many items to recommend:",
        [5, 10, 15, 20, 25],
        index=1  # default: 10
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ About Model")
    st.info(
        """
        **Method:** Truncated SVD (Matrix Factorization)
        **Goal:** Predict latent preferences based on past interactions.
        """
    )


# Main Page
st.title("🛍️ Product Recommendation Simulation")
st.markdown(f"Analyzed behavior for Customer ID: **{selected_user_id}**")

# Button to trigger logic
if st.button("Generate Recommendations", type="primary"):
    
    # --- DATA PROCESSING ---
    
    # 1. Get History
    user_history_mids = order_cust[order_cust['customer_id'] == selected_user_id]['mid'].unique().tolist()
    
    # 2. Get Recommendations
    recs_mids = get_svd_recommendations(selected_user_id, n_recs=n_recs)


    # --- DISPLAY METRICS ---
    col_metric1, col_metric2 = st.columns(2)
    with col_metric1:
        st.metric(label="Total Items Purchased", value=len(user_history_mids))
    with col_metric2:
        st.metric(label="Recommendations Generated", value=len(recs_mids))

    st.markdown("---")

    # --- DISPLAY TABLES ---
    
    col1, col2 = st.columns(2)
    
    # LEFT COLUMN: HISTORY
    with col1:
        st.subheader("📜 Purchase History")
        st.caption("Items this user has actually bought/interacted with.")
        
        if user_history_mids:
            history_df = pd.DataFrame({'mid': user_history_mids})
            history_df['mid'] = history_df['mid'].astype(str)
            
            # Merge to get descriptions
            history_display = history_df.merge(
                full_product, 
                on='mid', 
                how='left'
            )
            
            # Clean up columns for display
            display_df = history_display[['mid', 'mid_desc', 'desc2']].rename(columns={
                'mid': 'Product ID',
                'mid_desc': 'Product Name',
                'desc2': 'Category/Details'
            })

            display_df['Product Name'] = display_df['Product Name'].apply(mask_product_name)

            
            st.dataframe(
                display_df, 
                use_container_width=True, 
                hide_index=True,
                height=400
            )
        else:
            st.info("This user has no purchase history in the dataset.")

    # RIGHT COLUMN: RECOMMENDATIONS
    with col2:
        st.subheader(f"✨ Top {len(recs_mids)} Recommendations")
        st.caption("Predicted items based on SVD Latent Features.")
        
        if recs_mids:
            recs_df = pd.DataFrame({'mid': recs_mids})
            recs_df['mid'] = recs_df['mid'].astype(str)
            
            # Merge to get descriptions
            recs_display = recs_df.merge(
                full_product, 
                on='mid', 
                how='left'
            )
            
            # Clean up columns for display
            display_recs = recs_display[['mid', 'mid_desc', 'desc2']].rename(columns={
                'mid': 'Product ID',
                'mid_desc': 'Product Name',
                'desc2': 'Category/Details'
            })
            display_recs['Product Name'] = display_recs['Product Name'].apply(mask_product_name)
            
            st.dataframe(
                display_recs, 
                use_container_width=True, 
                hide_index=True,
                height=400
            )
        else:
            st.warning("Cold Start: No recommendations available for this user.")

else:
    st.info("👈 Please select a Customer ID from the sidebar and click 'Generate Recommendations'")
