import os
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.messages import AIMessage

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatOllama(model="llama3.1", temperature=0.8)
memory = InMemorySaver()
parser = StrOutputParser()

vector_store = None
agent = None

def process_pdf(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    text = " ".join(page.page_content for page in docs)
    chunks = RecursiveCharacterTextSplitter(chunk_size=270).create_documents([text])
    return chunks

def build_vector_index(chunks):
    global vector_store
    vector_store = FAISS.from_documents(chunks, embedding_model)

def build_agent():
    global agent
    @tool
    def rag_tool(question: str) -> str:
        docs = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 27, "lambda_mult": 0.2}).invoke(question)
        return "\n\n".join(doc.page_content for doc in docs)

    agent = create_react_agent(
        model=llm,
        tools=[rag_tool],
        prompt="you are a helpfull assistant. If user says hi/hello, greet politely. For other queries, use `rag_tool`. If answer not found, say you don't know.",
        checkpointer=memory,
    )

def answer_question(question):
    if not agent:
        return "⚠️ PDF not uploaded yet."
    response = agent.invoke({"messages": [{"role": "user", "content": question}]}, {"configurable": {"thread_id": 1}})
    final_message = response["messages"][-1]
    if isinstance(final_message, AIMessage):
        return parser.invoke(final_message.content)
    else:
        return "⚠️ Unexpected response."
