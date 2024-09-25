import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

UPLOAD_FOLDER = './uploads'

def create_upload_folder():
    """
    Create the upload folder if it doesn't exist.
    """
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

def load_document(file_path):
    """
    Load a document based on the file extension.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.pdf':
        loader = PyPDFLoader(file_path)
    elif file_extension == '.docx':
        loader = Docx2txtLoader(file_path)
    elif file_extension == '.txt':
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    return loader.load()

# Function to load prompt from file
def load_prompt(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()
    
def delete_old_files():
    """
    Delete old files in the upload folder.
    """
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  # Delete the file
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

# Ensure the upload folder exists when the module is imported
# create_upload_folder()
