### RAG-Based LLM Model with FastAPI Frontend.

This project implements a Retrieval-Augmented Generation (RAG) model using a FastAPI backend and a frontend built with HTML, CSS, and JavaScript. It allows users to upload PDF documents and ask questions based on the content of those documents.

## Overview

The project consists of the following main components:

* **FastAPI Backend:** Handles PDF uploads, processes the PDF content, builds a vector index, and creates an agent for question answering. It exposes API endpoints for uploading files and asking questions.
* **RAG Engine:** Responsible for processing the uploaded PDF, building a vector index for efficient retrieval, and using an agent to answer user questions based on the retrieved information.
* **Frontend:** A web interface built with HTML, CSS, and JavaScript that allows users to upload PDF files and submit questions to the backend.

## Setup Instructions

Follow these steps to set up and run the project:

1.  **Clone the repository:** (If you haven't already)
    ```bash
    git clone https://github.com/2000pawan/Rag_Based_LLM_Model.git
    ```

2.  **Navigate to the `frontend` directory and then back to the root directory:** The FastAPI application assumes the frontend files are in a directory named `frontend` in the same directory as the main Python file. Uploads\Project\LLM\Rag_Based_LLM_Model\frontend`. Ensure you have a `frontend` directory with your HTML, CSS, and JS files at that location or adjust the `app.mount` path accordingly.

3.  **Install backend dependencies:**
    ```bash
    pip install -r requirements.txt  # You might need to create this file with required libraries
    ```
    *(Note: You will need to create a `requirements.txt` file listing the dependencies for your backend. This would typically include `fastapi`, `uvicorn`, `pydantic`, and any libraries used in your `rag_engine.py` file such as those for PDF processing, vector databases, and LLM integration.)*

## Running the Application

1.  **Run the FastAPI backend:**
    ```bash
    https://rag-llm-pawan.onrender.com/  # In this case, the file content provided is the main file
    ```


## How to Use

1.  **Upload a PDF:** Use the "Choose File" button on the frontend to select a PDF document from your local machine.
2.  **Process the PDF:** Click the "Upload" button to send the PDF to the backend for processing. You should see a "PDF processed successfully!" message upon completion.
3.  **Ask a Question:** In the provided text area, type your question related to the content of the uploaded PDF.
4.  **Get the Answer:** Click the "Ask" button. The backend will process your question using the RAG engine and display the answer on the frontend.

## Project Structure

your_project_directory/

        ├── backend/              (Optional: You might organize backend code in a subfolder)
        │   └── main.py           (Your FastAPI application code)
        │   └── rag_engine.py     (Your RAG engine logic)
        │   └── ...
        ├── frontend/
        │   ├── index.html        (Frontend HTML)
        │   ├── styles.css        (Frontend CSS)
        │   ├── script.js         (Frontend JavaScript)
        │   └── ...
        ├── uploads/              (Directory to store uploaded PDF files)
        ├── requirements.txt      (Backend dependencies)
        └── README.md

*(Adjust the directory structure above to match your actual project organization.)*

## Technologies Used

* **Backend:** FastAPI, Python
* **Frontend:** HTML, CSS, JavaScript
* **RAG Engine:** *(List the specific libraries you used for PDF processing, vector database, and LLM integration, e.g., PyPDF2, ChromaDB, Langchain, etc.)*

## Further Improvements

* Implement error handling for file uploads and API requests.
* Add loading states to the frontend to improve user experience.
* Allow users to upload multiple PDF files.
* Implement user authentication and authorization if needed.
* Deploy the application to a cloud platform for wider access.

### 👤 Contact

Name: Pawan Yadav

Email: yaduvanshi2000pawan@gmail.com

### 📄 License

This project is licensed under the MIT License.
