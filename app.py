import streamlit as st
from PIL import Image
from src.engine import ShopEngine
from src.utils import convert_to_url
import os

st.set_page_config(page_title='Shop The Look AI', layout='wide', page_icon='✨')

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 20px; border: 1px solid #ff4b4b; color: #ff4b4b; background: white; }
    .stButton>button:hover { background: #ff4b4b; color: white; }
    .product-card { border-radius: 15px; background: white; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .match-label { color: #6c757d; font-size: 0.85rem; text-align: center; margin-top: 5px; }
    </style>
""", unsafe_allow_html=True)

st.title("✨ Fabulous Shop The Look AI")
st.write("Upload an image and let our AI find the hottest matching products from our catalog.")

@st.cache_resource
def load_engine():
    # Ensure these file names match the exact artifacts in your repository root
    return ShopEngine('catalog.index', 'embeddings.pkl', 'catalog_hashes.pkl')

if os.path.exists('catalog.index') and os.path.exists('embeddings.pkl') and os.path.exists('catalog_hashes.pkl'):
    engine = load_engine()
    with st.sidebar:
        st.header("Configuration")
        top_k = st.slider("Matches per item", 1, 5, 3)
        st.info("Detection is optimized for clothing and accessories.")

    uploaded_file = st.file_uploader("Drop an inspiration image here...", type=['jpg', 'png'])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Your Inspiration', use_container_width=True)
        
        if st.button('✨ ANALYZE THIS LOOK'):
            with st.spinner('Curating matches...'):
                results = engine.detect_products(image, top_k=top_k)
                if not results:
                    st.warning("No specific fashion products found. Try another image!")
                else:
                    for idx, res in enumerate(results):
                        with st.container():
                            st.markdown(f'<div class="product-card">', unsafe_allow_html=True)
                            col_info, col_m1, col_m2, col_m3 = st.columns([1.5, 2, 2, 2])
                            
                            with col_info:
                                st.image(res['crop'], use_container_width=True)
                                st.write(f"**Detected:** {res['label'].title()}")
                                st.write(f"*Confidence:* {res['conf']:.1%}")
                            
                            match_cols = [col_m1, col_m2, col_m3]
                            for i, match in enumerate(res['matches'][:3]):
                                with match_cols[i]:
                                    url = convert_to_url(match['product_id'])
                                    st.image(url, use_container_width=True)
                                    st.markdown(f'<div class="match-label">Match {int(match["similarity"]*100)}%</div>', unsafe_allow_html=True)
                                    st.button('View Details', key=f'view_{idx}_{i}')
                            st.markdown('</div>', unsafe_allow_html=True)
else:
    st.error("System artifacts missing. Please ensure 'catalog.index', 'embeddings.pkl', and 'catalog_hashes.pkl' are uploaded and tracked via Git LFS in the root directory.")