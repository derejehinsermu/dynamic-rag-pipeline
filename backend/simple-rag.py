__import__('pysqlite3')
import os
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# Get the absolute path of the scripts directory
root_path_scripts = os.path.abspath(os.path.join(os.getcwd(), '../scripts/'))
sys.path.append(root_path_scripts)

import openai
from openai import OpenAI
from flask_cors import CORS
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import functions from load_docs.py
from load_docs import load_document, delete_old_files, UPLOAD_FOLDER, load_prompt

# Load environment variables and set OpenAI API key
load_dotenv()
openai_client = OpenAI()

openai_api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = openai_api_key

# Set up Flask application
app = Flask(__name__)
CORS(app)

# A dictionary to map prompt types to file paths
PROMPT_FILES = {
    'Resume Reviewer': '../prompts/Resume_reviewer/base_prompt.txt',
    'Contract Legal Advisor': '../prompts/legal_contract_advisor/base_prompt.txt'
}

# Function to fetch available prompts
@app.route('/prompts', methods=['GET'])
def get_prompts():
    return jsonify({"prompts": list(PROMPT_FILES.keys())})

# Upload file endpoint
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    delete_old_files()

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    documents = load_document(file_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    split_docs = text_splitter.split_documents(documents)

    # Generate embeddings using HF
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=split_docs, embedding=embedding_function)

    app.config['VECTORSTORE'] = vectorstore

    return jsonify({"message": "File uploaded and processed successfully!"})

# Custom RAG pipeline to handle questions with dynamic prompt selection
@app.route('/ask-pipeline1', methods=['POST'])
def ask_question():
    question = request.json.get("question")
    prompt_type = request.json.get("promptType")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    if not prompt_type or prompt_type not in PROMPT_FILES:
        return jsonify({"error": "Invalid or missing prompt type"}), 400

    vectorstore = app.config.get('VECTORSTORE')
    if not vectorstore:
        return jsonify({"error": "No document uploaded. Please upload a document to ask questions."}), 400

    try:
        # Load the prompt based on the selected prompt type
        base_prompt = load_prompt(PROMPT_FILES[prompt_type])

        # Retrieve relevant documents using the joint query
        retrieved_documents = vectorstore.similarity_search(question, k=5)
        if not retrieved_documents:
            return jsonify({"error": "No relevant documents found for the query."}), 404

        # Use the RAG function to generate an answer based on the retrieved documents
        answer = base_rag(query=question, retrieved_documents=retrieved_documents, prompt=base_prompt)
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"Error during RAG pipeline: {e}")
        return jsonify({"error": "RAG pipeline failed"}), 500

# RAG function to generate the final answer with a dynamic prompt
def base_rag(query, retrieved_documents, prompt, model="gpt-4o-mini"):
    information = "\n\n".join([doc.page_content for doc in retrieved_documents])

    messages = [
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": f"Question: {query}\n\nInformation:\n{information}\n\nAnswer:"
        }
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
