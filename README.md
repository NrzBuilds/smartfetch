# 🔬 SmartFetch

**SmartFetch** is an AI-powered chatbot application that lets you interact with your own documents, web pages, and YouTube videos using cutting-edge Large Language Models (LLMs) via **Groq's ultra-fast inference API**.

Built with **Streamlit**, **LangChain**, and **FAISS**, SmartFetch combines Retrieval-Augmented Generation (RAG) with an intelligent web search fallback — ensuring you always get an answer, whether it's inside your uploaded documents or out on the web.

---

## ✨ Features

- 📄 **PDF Upload & Chat** — Upload one or multiple PDFs and ask questions directly about their content.
- 🌐 **Web & YouTube Ingestion** — Paste any website URL or YouTube link to extract and index its content.
- 🧠 **RAG-powered Q&A** — Uses FAISS vector search + HuggingFace embeddings to fetch the most relevant context before answering.
- 🔍 **Smart Web Search Fallback** — If the answer isn't found in your documents, a "Search Web Instead?" button appears instantly, powered by DuckDuckGo.
- 💬 **Conversational Memory** — Remembers the last few exchanges for more natural, context-aware conversations.
- 🔑 **Multi-Key API Rotation** — Automatically rotates between multiple Groq API keys if rate limits (429 errors) are hit.
- 📤 **Session Export** — Download your entire chat history as a text file.
- 🎨 **Clean, Modern UI** — Custom-styled Streamlit interface for a smooth user experience.

---

## 🛠️ Tech Stack

| Component          | Technology                          |
|---------------------|--------------------------------------|
| Frontend / UI       | Streamlit                           |
| LLM Provider        | Groq (openai/gpt-oss-120b, gpt-oss-20b) |
| Orchestration       | LangChain                           |
| Vector Store        | FAISS                               |
| Embeddings          | HuggingFace (`all-MiniLM-L6-v2`)    |
| Document Loaders    | PyPDFLoader, WebBaseLoader, YoutubeLoader |
| Web Search          | DuckDuckGo Search (`ddgs`)          |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/NrzBuilds/smartfetch.git
cd smartfetch