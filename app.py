from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from rag_engine import process_pdf, build_vector_index, build_agent, answer_question

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['pdf']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    chunks = process_pdf(filepath)
    build_vector_index(chunks)
    build_agent()
    return jsonify({"message": "PDF processed successfully!"})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")
    answer = answer_question(question)
    return jsonify({"answer": answer})

@app.route("/")
def serve_frontend():
    return send_from_directory("../frontend", "index.html")

@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory("../frontend", path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
