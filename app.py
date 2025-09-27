from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os, tempfile, shutil
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

# Always use /tmp for uploads (ephemeral storage in Hugging Face)
UPLOAD_FOLDER = tempfile.gettempdir()

# Clear any leftover files from previous runs
if os.path.exists(UPLOAD_FOLDER):
    try:
        for f in os.listdir(UPLOAD_FOLDER):
            path = os.path.join(UPLOAD_FOLDER, f)
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
    except Exception as e: 
        print(f"Warning: could not fully clear tmp: {e}")

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_pdf(pdf: UploadFile = File(...)):
    if not pdf:
        return JSONResponse(content={"error": "No file uploaded"}, status_code=400)

    # Always save to /tmp with same name (overwrite old one)
    filepath = os.path.join(UPLOAD_FOLDER, "current_upload.pdf")
    with open(filepath, "wb") as f:
        while chunk := await pdf.read(1024 * 1024):  # Read in chunks
            f.write(chunk)

    # Process freshly uploaded PDF
    chunks = process_pdf(filepath)
    build_vector_index(chunks)
    build_agent()

    return JSONResponse(content={"message": "PDF processed successfully!"})

@app.post("/ask")
async def ask(question_request: QuestionRequest):
    answer = answer_question(question_request.question)
    return JSONResponse(content={"answer": answer})

# Serve frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)