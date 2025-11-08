# AskMyDocuments

AskMyDocuments is a small project that demonstrates a mini-RAG (Retrieval-Augmented Generation) system using AWS Bedrock, Amazon S3, and Python. It allows users to ask questions about their documents stored in S3 and receive intelligent answers powered by Bedrock foundation models.

## Features
- Upload and manage documents in Amazon S3
- Embed and retrieve document chunks using vector search
- Generate answers using AWS Bedrock models
- Simple and extensible Python3 codebase
- GitHub Codespaces compatible

## Technologies Used
- Python 3
- AWS Bedrock
- Amazon S3
- LangChain
- FAISS (for vector search)
- GitHub & GitHub Codespaces

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AskMyDocuments.git
   ```
2. Open in GitHub Codespaces or your local IDE.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure AWS credentials using environment variables or IAM roles.
5. ``` bash
   sudo apt update && sudo apt install tesseract-ocr -y (System Requirements)
   ```
## Usage
- Upload documents to your configured S3 bucket.
- Run the ingestion script to chunk and embed documents.
- Ask questions via CLI or web interface.
- View answers generated using Bedrock models.

## License
This project is licensed under the MIT License.
