from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
from frontend.rag_engine import process_pdf, build_vector_index, build_agent, answer_question
from pydantic import BaseModel

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - for development purposes. Restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_pdf(pdf: UploadFile = File(...)):
    if not pdf:
        return JSONResponse(content={"error": "No file uploaded"}, status_code=400)
    filepath = os.path.join(UPLOAD_FOLDER, pdf.filename)
    with open(filepath, "wb") as f:
        while chunk := await pdf.read(1024 * 1024):  # Read in chunks
            f.write(chunk)
    chunks = process_pdf(filepath)
    build_vector_index(chunks)
    build_agent()
    return JSONResponse(content={"message": "PDF processed successfully!"})

@app.post("/ask")
async def ask(question_request: QuestionRequest):
    answer = answer_question(question_request.question)
    return JSONResponse(content={"answer": answer})

app.mount("/", StaticFiles(directory=r"G:\Coding\Git Uploads\Project\LLM\Rag_Based_LLM_Model\frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)