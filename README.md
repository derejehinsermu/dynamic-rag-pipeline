## Resume Reviewer and Legal Contract Advisor Retrieval-Augmented Generation (RAG) System

## Overview

Overview

The Resume Reviewer and Legal Contract Advisor RAG system is a dual-purpose solution designed to assist users in both reviewing resumes and providing expert legal advice on contract-related inquiries. Leveraging advanced natural language processing and retrieval-augmented generation techniques, this system enables users to receive personalized feedback on resumes and get the correct answer about legal contracts. It provides a seamless interaction by dynamically selecting the appropriate model and prompt for each task, ensuring high-quality, context-aware responses.


## Technologies Used

- **LangChain**: A framework for building applications with large language models (LLMs).
- **OpenAI API**: Provides access to GPT-4o-mini for language processing.
- **GPT-4o-mini**: Utilized for advanced language understanding and generation.
- **ChromaDB**: A vector database for storing and retrieving contract data.
- **HuggingFaceEmbeddings**: to embed a query and documents

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project develops a dual-function Resume Reviewer and Legal Contract Advisor Bot, enabling users to interact with both resumes and contract documents. For resume reviews, the bot offers personalized feedback by analyzing the document and providing structured suggestions. For legal contract inquiries, the bot processes contracts by segmenting them into manageable chunks and storing them in a vector database. It then retrieves relevant information to deliver precise responses. This system utilizes technologies like LangChain, Chroma, HuggingFace, and GPT models to ensure accurate and context-aware answers for both domains.

## Features

- **User-Friendly Interface**: Built with Streamlit for an easy-to-use web interface.
- **Dual Functionality**: Supports both Resume Review and Legal Contract Advisory, allowing users to choose the specific mode of interaction.
- **Interactive Q&A**: Enables users to ask detailed questions about resumes or contract documents and receive contextually accurate responses.
- **Advanced Query Augmentation**: Employs advanced techniques to enhance user queries for more precise document retrieval and recommendations.
- **Efficient Data Retrieval**: Utilizes Chroma and HuggingFace embeddings for fast and accurate data storage and retrieval.
- **Dynamic Prompt Selection**: Supports multiple prompts and configurations, dynamically adapting to different query complexities and user requirements.
- **User-Friendly Interface**: Provides a seamless and intuitive web interface for document upload, question submission, and prompt selection.
## Getting Started

### Prerequisites

- Python 3.8 or higher

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/derejehinsermu/dynamic-rag-pipeline.git

2. **Create and Activate a Virtual Environment**
   
    Navigate to the root directory of the project and create a virtual environment named 'venv', then activate it:
    ```sh
    cd dynamic-rag-pipeline.git
    python -m venv venv  | virtualenv venv
    source venv/bin/activate

4. **Install Requirements**
5. **Frontend Setup (React)**:
   
    Install the required dependencies for the React frontend:
   ```bash
    cd ../frontend
    npm install

   
## Usage

  To run the Resume Reviewer and Legal Expert Advisor:

6. **Start the Flask Backend**:
   
   Navigate to the backend directory and run the Flask app:

   ```bash
    cd backend
    
    Python simple-rag.py 
    
    python adv_rag_with_aqg.py
    
    python adv_rag_with_mq_ranking.py


8. **Start the React Frontend:**
   
   Open a new terminal, navigate to the frontend directory, and start the React development server:

    ```bash
    cd ../frontend-app
    npm start

Open your browser and navigate to http://localhost:3000 to interact with the system.

![Demo](https://github.com/user-attachments/assets/68d0e863-96b5-456e-bd69-d71329f6dbd0)

## Contributing

Contributions are welcome! 

9. For any questions or support, please contact derejehinsermu2@gmail.com.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
