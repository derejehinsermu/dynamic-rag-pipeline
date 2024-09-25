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
from sentence_transformers import CrossEncoder
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

# A dictionary to map prompt types to file paths for both base and advanced prompts
PROMPT_FILES = {
    'Resume Reviewer': {
        'base': '../prompts/Resume_reviewer/base_prompt.txt',
        'advanced': '../prompts/Resume_reviewer/adv_rag_with_multiple_query_generated_ranking.txt'
    },
    'Contract Legal Advisor': {
        'base': '../prompts/legal_contract_advisor/base_prompt.txt',
        'advanced': '../prompts/legal_contract_advisor/adv_rag_with_multiple_query_generated_ranking.txt'
    }
}
# Function to fetch available prompts and their types
@app.route('/prompts', methods=['GET'])
def get_prompts():
    available_prompts = list(PROMPT_FILES.keys())
    return jsonify({"prompts": available_prompts})

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

    # Generate embeddings using HuggingFace model
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=split_docs, embedding=embedding_function)

    app.config['VECTORSTORE'] = vectorstore

    return jsonify({"message": "File uploaded and processed successfully!"})

# Advanced RAG pipeline to handle questions with multiple related queries and re-ranking
@app.route('/ask-pipeline3', methods=['POST'])
def ask_pipeline3():
    question = request.json.get("question")
    prompt_type = request.json.get("promptType")  # Get prompt type
    prompt_complexity = request.json.get("complexity", "base")  # Get complexity level (base or advanced)

    if not question:
        return jsonify({"error": "No question provided"}), 400

    if not prompt_type or prompt_type not in PROMPT_FILES:
        return jsonify({"error": "Invalid or missing prompt type"}), 400

    if prompt_complexity not in PROMPT_FILES[prompt_type]:
        return jsonify({"error": f"Invalid complexity level '{prompt_complexity}' for prompt type '{prompt_type}'"}), 400

    vectorstore = app.config.get('VECTORSTORE')

    if not vectorstore:
        return jsonify({"error": "No document uploaded. Please upload a document to ask questions."}), 400

    try:
        # Load the appropriate prompt for the selected task and complexity
        base_prompt = load_prompt(PROMPT_FILES[prompt_type]['base'])
        advanced_prompt = load_prompt(PROMPT_FILES[prompt_type]['advanced'])

        # Step 1: Generate multiple related queries for the input question
        if prompt_complexity == 'advanced':
            augmented_queries = augment_multiple_query(question, advanced_prompt)
        else:
            augmented_queries = augment_multiple_query(question, base_prompt)

        queries = [question] + augmented_queries

        # Step 2: Retrieve relevant documents using all the generated queries
        retrieved_documents = set()
        for query in queries:
            results = vectorstore.similarity_search(query=query, k=10)
            for doc in results:
                if isinstance(doc.page_content, str):
                    retrieved_documents.add(doc.page_content)

        if not retrieved_documents:
            return jsonify({"error": "No relevant documents found for the query."}), 404

        # Step 3: Deduplicate the retrieved documents and use CrossEncoder for re-ranking
        unique_documents = list(retrieved_documents)
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        pairs = [[question, doc] for doc in unique_documents]
        scores = cross_encoder.predict(pairs)
        ranked_documents = sorted(zip(scores, unique_documents), reverse=True, key=lambda x: x[0])

        # Select top 5 documents after re-ranking
        top_5_documents = [doc for score, doc in ranked_documents[:5]]

        # Step 4: Use the RAG function to generate an answer based on the top 5 re-ranked documents
        # Using base prompt for RAG regardless of query augmentation complexity
        answer = rag(query=question, retrieved_documents=top_5_documents, prompt=base_prompt)

        return jsonify({"answer": answer})

    except Exception as e:
        print(f"Error during RAG pipeline: {e}")
        return jsonify({"error": "RAG pipeline failed"}), 500

# Function to generate multiple related queries with dynamic prompt selection
def augment_multiple_query(query, selected_prompt, model="gpt-4o"):
    messages = [
        {
            "role": "system",
            "content": selected_prompt
        },
        {"role": "user", "content": query}
    ]
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    content = response.choices[0].message.content
    return content.split("\n")

# RAG function to generate the final answer with dynamic prompt selection
def rag(query, retrieved_documents, prompt, model="gpt-4o"):
    information = "\n\n".join([doc for doc in retrieved_documents])

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
            temperature=0
        )
    content = response.choices[0].message.content
    return content

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
