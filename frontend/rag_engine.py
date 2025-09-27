from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from dotenv import load_dotenv
import os, re

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Embedding and LLM models
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(model="qwen/qwen3-32b", temperature=0.6, reasoning_effort="default")

# Memory & parser
memory = InMemorySaver()
parser = StrOutputParser()

# Globals (reset on each PDF upload)
vector_store = None
agent = None

def process_pdf(path: str):
    """Load and split a PDF into text chunks."""
    loader = PyPDFLoader(path)
    docs = loader.load()
    text = " ".join(page.page_content for page in docs)
    chunks = RecursiveCharacterTextSplitter(chunk_size=350).create_documents([text])
    return chunks

def build_vector_index(chunks):
    """Build FAISS vector index from PDF chunks."""
    global vector_store
    vector_store = None  # reset old index
    vector_store = FAISS.from_documents(chunks, embedding_model)

def build_agent():
    """Create a new RAG agent tied to the current vector store."""
    global agent, vector_store
    agent = None  # reset old agent

    if not vector_store:
        raise ValueError("❌ Vector store not initialized. Upload and process a PDF first.")

    @tool
    def rag_tool(question: str) -> str:
        """Answer a question using context from the uploaded PDF."""
        print('==========================================rag tool=============================================================')
        docs = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 1, "lambda_mult": 0.2}
        ).invoke(question)
        return "\n\n".join(doc.page_content for doc in docs)
    tool_json_schema = [
       {
    "name": "rag_tool",
    "description": "Answer questions based on the currently uploaded PDF. Files are processed temporarily and not stored after the query.",
    "arguments": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The user's question to ask from the uploaded PDF context"
            }
        },
        "required": ["question"]
    }
}

    ]
    agent = create_react_agent(
        model=llm,
        tools=[rag_tool],
        prompt = ("""
    "You are a helpful and professional assistant. "
    🔧 Available tools:
    {tool_json_schema}
    "If the user greets you (hi/hello), respond politely. "
    "For other queries, use `rag_tool` to retrieve relevant information. "
    "Summarize the `rag_tool` response and present it clearly and professionally. "
    "If the response contains steps, instructions, or formulas, format them in an easy-to-read way: "
    "- Numbered steps for procedures or sequences "
    "- Proper formatting for formulas or equations "
    "- Bullet points for lists or key points "
    "If no relevant context is found, reply only with: "
    "`This chatbot answers only using the content from your uploaded PDF.`"
    ====================================================================================
    🚫 HARD RESTRICTIONS:
    - Never hallucinate service, subservice, device, or issue.  
    - Never call multiple tools in one turn.  
    - Only invoke when explicit input is available.  
    """
        ),
        checkpointer=memory,
    )

def strip_markdown(md_text: str) -> str:
    """Remove markdown formatting from model output."""
    clean = re.sub(r'(\*\*|__)(.*?)\1', r'\2', md_text)  # bold/italic
    clean = re.sub(r'(```.*?```)', '', clean, flags=re.DOTALL)  # code blocks
    clean = re.sub(r'[#>`]', '', clean)  # headings, blockquotes
    return clean.strip()

def answer_question(question: str) -> str:
    """Ask a question and return the agent's plain-text response."""
    if not agent:
        return "⚠️ No PDF uploaded yet. Please upload a PDF first."

    response = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        {"configurable": {"thread_id": 1}}
    )
    final_message = response["messages"][-1]

    if isinstance(final_message, AIMessage):
        raw_text = parser.invoke(final_message.content)
        #return strip_markdown(raw_text)
        return raw_text
    return "⚠️ Unexpected response format."