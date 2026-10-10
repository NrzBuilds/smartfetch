import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun

# Load environment variables from .env file
load_dotenv()

# 1. Page Configuration
st.set_page_config(page_title="SmartFetch", layout="wide", page_icon="🔬")

# Custom CSS for a cleaner, premium look
st.markdown("""<style>
    .stChatMessage { border-radius: 12px; margin-bottom: 15px; }
    .stDownloadButton>button { width: 100%; background-color: #4CAF50; color: white; border-radius: 8px; }
    .stButton>button { border-radius: 8px; font-weight: 600; transition: all 0.3s ease; }
    .stButton>button:hover { transform: scale(1.02); }
</style>""", unsafe_allow_html=True)

# 2. Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_key_idx" not in st.session_state:
    st.session_state.current_key_idx = 0
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "trigger_web_search" not in st.session_state:
    st.session_state.trigger_web_search = False

def generate_export():
    """Generates a formatted string of the chat history for export."""
    return "\n\n".join([f"[{m['role'].upper()}]\n{m['content']}" for m in st.session_state.messages])

# 3. Sidebar Setup
with st.sidebar:
    st.title("🔬 SmartFetch")

    # Load API keys securely from the environment (.env file)
    env_keys = os.getenv("GROQ_API_KEYS", "")

    if env_keys:
        keys_input = env_keys
    else:
        # Fallback if environment keys are not found
        keys_input = st.text_input("Groq API Keys (comma separated):", type="password")

    groq_keys = [k.strip() for k in keys_input.split(",") if k.strip()]

    # Premium subtle status indicator instead of large boxes
    st.markdown("---")
    if groq_keys:
        st.caption("🟢 **System Online:** API Connected")
        st.caption(f"🔑 **Active Slot:** #{(st.session_state.current_key_idx % len(groq_keys)) + 1}")
    else:
        st.error("⚠️ API Key Missing!")

    # Active Models Dropdown
    st.markdown("---")
    model_options = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b"
]
    selected_model = st.selectbox("🤖 Select AI Model", model_options)
    st.markdown("---")

    # File and URL upload sections
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    url_input = st.text_input("Web/YouTube URL")

    # Export and Reset options
    if st.session_state.messages:
        st.download_button("📄 Export Session", data=generate_export(), file_name="Nexus_Report.txt")

    if st.session_state.vector_store is not None:
        if st.button("🧹 Clear Knowledge Base", use_container_width=True):
            st.session_state.vector_store = None
            st.rerun()

    # Upgraded Reset Button
    if st.button("🗑️ Clear Chat & Reset", type="primary", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# 4. Document Indexing Logic
if (uploaded_files or url_input) and groq_keys and not st.session_state.vector_store:
    all_chunks = []
    with st.spinner("Indexing Knowledge Base..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        # Process uploaded PDF files
        if uploaded_files:
            for f in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.getvalue())
                    loader = PyPDFLoader(tmp.name)
                    all_chunks.extend(text_splitter.split_documents(loader.load()))

        # Process Web or YouTube URLs
        if url_input:
            try:
                if "youtu" in url_input:
                    loader = YoutubeLoader.from_youtube_url(url_input, add_video_info=True)
                else:
                    loader = WebBaseLoader(url_input)
                all_chunks.extend(text_splitter.split_documents(loader.load()))
            except Exception as e:
                st.error(f"URL Error: {e}")

        # Create FAISS Vector Store if chunks are available
        if all_chunks:
            st.session_state.vector_store = FAISS.from_documents(
                all_chunks,
                HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            )
            st.toast("Knowledge Base completely indexed and ready!", icon="🧠")

# 5. Chat Display Setup
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

        # BULLETPROOF Web Search fallback button
        if m["role"] == "assistant" and i == len(st.session_state.messages)-1:
            ans_text = m["content"].lower()
            trigger_words = ["not present", "cannot answer", "does not mention", "doesn't mention", "no information"]

            if any(word in ans_text for word in trigger_words):
                if st.button("🌐 Search Web instead?", key=f"web_btn_{i}", use_container_width=True):
                    st.session_state.trigger_web_search = True
                    st.rerun()

# 6. Chat Input & Execution Logic
prompt = st.chat_input("Ask anything...")

if prompt or st.session_state.trigger_web_search:
    if not groq_keys:
        st.error("Please add Groq API Keys to proceed!")
        st.stop()

    # Determine the active query based on whether it's a fresh prompt or a web search fallback
    if st.session_state.trigger_web_search:
        active_query = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
    else:
        active_query = prompt
        st.session_state.messages.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

    # Assistant Response Handling
    with st.chat_message("assistant"):
        res_box = st.empty()
        full_res = ""
        current_key = groq_keys[st.session_state.current_key_idx % len(groq_keys)]

        try:
            llm = ChatGroq(groq_api_key=current_key, model_name=selected_model, streaming=True)

            # Define System Message based on context (Web Search vs Document QA vs General)
            if st.session_state.trigger_web_search:
                res_box.info(f"🌐 Searching Web for: {active_query}...")

                # SAFE WEB SEARCH LOGIC
                try:
                    search_data = DuckDuckGoSearchRun().run(active_query)
                    if not search_data:
                        search_data = "No relevant results found."
                    sys_msg = f"Provide a detailed answer based ONLY on these web search results: {search_data}"
                except Exception as e:
                    sys_msg = "Apologize to the user and tell them that the web search engine is currently down or blocked, so you cannot fetch the answer right now."

                st.session_state.trigger_web_search = False # Reset flag

            elif st.session_state.vector_store:
                docs = st.session_state.vector_store.similarity_search(active_query, k=4)
                context = "\n\n".join([d.page_content for d in docs])
                sys_msg = f"Answer ONLY using the following context: {context}. If the answer is not in the context, strictly reply with: 'I cannot answer this as the information is not present in the uploaded document.'"
            else:
                sys_msg = "You are a helpful AI assistant."

            # SMART CONVERSATION MEMORY (Last 4 messages)
            chat_history = [("system", sys_msg)]
            recent_msgs = st.session_state.messages[-5:-1] if len(st.session_state.messages) > 1 else []
            for m in recent_msgs:
                role = "human" if m["role"] == "user" else "assistant"
                chat_history.append((role, m["content"]))

            chat_history.append(("human", active_query))

            # Stream the response
            for chunk in llm.stream(chat_history):
                full_res += chunk.content
                res_box.markdown(full_res + "▌")

            res_box.markdown(full_res)
            st.session_state.messages.append({"role": "assistant", "content": full_res})

            # MAGIC REFRESH (To load web search button instantly without waiting for next prompt)
            if not st.session_state.trigger_web_search:
                st.rerun()

        except Exception as e:
            # Handle API Rate Limits (429) by rotating keys
            if "429" in str(e):
                st.session_state.current_key_idx += 1
                st.warning("Rate limit hit. Rotating API key...")
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"An error occurred: {e}")