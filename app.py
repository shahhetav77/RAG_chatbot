import os
import streamlit as st

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter

# -------------------------------------------------
# Streamlit Page Config (MUST BE FIRST)
# -------------------------------------------------
st.set_page_config(
    page_title="RAG LLM Chatbot",
    page_icon="🤖",
    layout="centered"
)

# -------------------------------------------------
# Custom CSS
# -------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
}

div[data-testid="stChatMessage"][aria-label="user"] {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
    border-radius: 12px;
    padding: 12px;
}

div[data-testid="stChatMessage"][aria-label="assistant"] {
    background-color: #e8f5e9;
    border-left: 4px solid #4caf50;
    border-radius: 12px;
    padding: 12px;
}

.stChatInput > div > div > input {
    border-radius: 20px !important;
    padding: 12px 20px !important;
}

h1 {
    text-align: center;
    color: #1e88e5;
}

.footer {
    text-align: center;
    margin-top: 40px;
    color: #777;
    font-size: 0.9em;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# App Title
# -------------------------------------------------
st.title("🤖 RAG LLM Chatbot")
st.markdown("### Ask questions about *llm.pdf*")
st.caption("Powered by LlamaIndex • Ollama (gemma3:1b) • sentence-transformers")

# -------------------------------------------------
# Load Models & Index (CACHED)
# -------------------------------------------------
@st.cache_resource
def load_models():
    # Embedding model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Optimized chunking for small LLMs
    Settings.node_parser = SentenceSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    # Gemma 3 (1B)
    Settings.llm = Ollama(
        model="gemma3:1b",
        request_timeout=120
    )

    # Absolute path (Windows-safe)
    DATA_PATH = r"D:\IITB_llm_with_python\data\llm.pdf"

    documents = SimpleDirectoryReader(
        input_files=[DATA_PATH]
    ).load_data()

    index = VectorStoreIndex.from_documents(documents)
    return index

index = load_models()

query_engine = index.as_query_engine(
    similarity_top_k=2,
    response_mode="compact"
)

# -------------------------------------------------
# Chat History
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! I'm a RAG-powered chatbot. Ask me anything about the document."
    }]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------------------------
# User Input
# -------------------------------------------------
DAD_JOKE_SYSTEM_PROMPT = """
You are a cheerful, expressive dad joke chatbot.

Your personality:
- Friendly
- Playful
- Slightly goofy
- Confident dad energy

Rules:
- ONLY respond with dad jokes
- Add light emotion using emojis (😄 😂 🤦‍♂️)
- React like a dad who is proud of his joke
- Never explain the joke
- Keep responses short (1–3 lines max)
- Even serious questions should get a dad joke

Style examples:
- Use phrases like: "Alright, here we go 😄", "Classic one!", "You’ll love this 🤓"
"""

if prompt := st.chat_input("Type your question here..."):
    # User message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            full_prompt = f"""
{DAD_JOKE_SYSTEM_PROMPT}

User asked: {prompt}
Respond with energy and humor.
"""
            response = query_engine.query(full_prompt)
            st.markdown(response.response)

            # Sources (optional – can remove for jokes)
            with st.expander("📄 Sources"):
                for node in response.source_nodes:
                    source = node.metadata.get("file_name", "Unknown source")
                    st.markdown(f"- {source}")

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.response
    })

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown(
    "<div class='footer'>Built with ❤️ using Streamlit & LlamaIndex</div>",
    unsafe_allow_html=True
)
