import os
import tempfile
import streamlit as st
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

# -----------------------------------------------------------
# 🎯 Streamlit Chat‑with‑PDF App
# -----------------------------------------------------------
# 1. User drops a PDF → we build a FAISS index on‑the‑fly.
# 2. Questions are routed through a LangGraph ReAct agent that
#    calls `rag_tool`, which pulls context from the index.
# -----------------------------------------------------------

st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="🧠", layout="wide")
st.title("🧠 Chat with your PDF")

# -- 1️⃣ API key input -------------------------------------------------------
api_key = st.secrets["api_keys"]["api_key"]


# -- 2️⃣ Upload PDF ----------------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)
if uploaded_file is None:
    st.stop()

# -- Helpers ----------------------------------------------------------------

def build_vector_store(pdf_bytes: bytes):
    """Create a FAISS index from raw PDF bytes and return a retriever."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    text = " ".join(page.page_content for page in documents)

    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    chunks = splitter.create_documents([text])

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key
    )
    vector_store = FAISS.from_documents(chunks, embedding_model)
    return vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 7, "lambda_mult": 0.2}
    )


def create_agent(retriever):
    """Wire the retriever into a LangGraph ReAct agent and return it."""

    @tool
    def rag_tool(question: str) -> str:
        """Return PDF context relevant to *question*."""
        docs = retriever.invoke(question)
        return "\n\n".join(doc.page_content for doc in docs)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", api_key=api_key, temperature=0.8
    )

    return create_react_agent(
        model=llm,
        tools=[rag_tool],
        prompt="Answer ONLY from the provided PDF context, using `rag_tool` to pull passages.",
    )

# -- 3️⃣ (Re)index if this is a *new* PDF -----------------------------------
# Create a stable hash so even files with the same name but different content are detected.
pdf_bytes = uploaded_file.getbuffer()
pdf_hash = hashlib.md5(pdf_bytes).hexdigest()
pdf_id = f"{uploaded_file.name}_{pdf_hash}"

if st.session_state.get("active_pdf_id") != pdf_id:
    with st.spinner("Indexing your PDF — this may take a moment..."):
        st.session_state.retriever = build_vector_store(pdf_bytes)
        st.session_state.agent = create_agent(st.session_state.retriever)
        st.session_state.parser = StrOutputParser()
        st.session_state.messages = []  # 🔄 reset chat history
        st.session_state.active_pdf_id = pdf_id
    st.success("PDF indexed! Ask away.")

# -- 4️⃣ Render existing chat messages --------------------------------------
for msg in st.session_state.get("messages", []):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -- 5️⃣ Chat interface ------------------------------------------------------
question = st.chat_input("Ask a question about the uploaded PDF…")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.spinner("Thinking…"):
        response = st.session_state.agent.invoke({
            "messages": [{"role": "user", "content": question}]
        })
        final_message = response["messages"][-1]
        if isinstance(final_message, AIMessage):
            answer = st.session_state.parser.invoke(final_message.content)
            st.write(answer)
        else:
            answer = "⚠️ Unexpected response structure."

    # ➜ Store exchange; assistant reply will appear on automatic rerun
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer})
