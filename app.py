import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="PDF AI Chat", layout="centered")
st.title("📚 Chat with your PDF")

# 1. Load Model (Cached so it only loads once)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# 2. File Uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Extract Text
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
    
    if sentences:
        # Pre-calculate embeddings for the whole PDF (Cached for speed)
        @st.cache_data
        def get_embeddings(_sentences):
            return model.encode(_sentences)
        
        sentence_embeddings = get_embeddings(sentences)
        st.success(f"PDF Loaded! Ready to chat.")

        # 3. Chat System
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if question := st.chat_input("Ask me anything about the PDF..."):
            st.chat_message("user").write(question)
            st.session_state.messages.append({"role": "user", "content": question})

            # AI Logic 
            question_embedding = model.encode(question)
            scores = util.cos_sim(question_embedding, sentence_embeddings)
            best_match_idx = scores.argmax()
            confidence = float(scores.max())
            answer = sentences[best_match_idx]

            full_res = f"{answer}\n\n(Match Confidence: {confidence:.2%})"
            
            with st.chat_message("assistant"):
                st.write(full_res)
            st.session_state.messages.append({"role": "assistant", "content": full_res})

