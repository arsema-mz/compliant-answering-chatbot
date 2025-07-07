# Complaint Answering Chatbot for Financial Complaints

This project implements a **Retrieval-Augmented Generation (RAG)**-based chatbot that answers financial complaints using real-world complaint narratives from the **CFPB** (Consumer Financial Protection Bureau) dataset. The chatbot is designed to assist financial analysts or customer service teams by providing accurate, grounded, and empathetic responses to customer concerns.

## 📌 Project Objectives

- Build an intelligent chatbot that answers financial complaints by retrieving relevant context from a vector database.
- Improve trustworthiness by grounding answers in actual customer complaint data (RAG approach).
- Deliver a user-friendly, interactive interface for non-technical users.
- Prepare for deployment as a web app accessible to stakeholders.

## ✅ Completed Tasks

### 1. Exploratory Data Analysis and Preprocessing

- Loaded and explored the CFPB complaint dataset.
- Cleaned the `consumer_complaint_narrative` field by removing nulls, duplicates, and irrelevant or overly short texts.
- Filtered complaints to focus on key financial products: Credit Cards, Personal Loans, Buy Now Pay Later (BNPL), Savings Accounts, and Money Transfers.
- Saved a cleaned dataset ready for downstream processing.

### 2. Text Chunking, Embedding, and Vector Store Indexing

- Split long complaint texts into smaller, semantically coherent chunks with overlap to preserve context.
- Generated embeddings for chunks using a pre-trained SentenceTransformer model (`all-MiniLM-L6-v2`).
- Indexed the embeddings and associated metadata (complaint ID, theme) into a **ChromaDB** vector store for fast and efficient similarity search.

### 3. RAG Pipeline (Retrieval-Augmented Generation)

- Built the RAG pipeline comprising:
  - **Retriever:** Uses semantic similarity search on ChromaDB to fetch top-k relevant complaint chunks based on the user's question.
  - **Generator:** Uses an instruction-tuned text generation model (Google's `flan-t5-small`) to synthesize answers based on retrieved context.
- Designed a prompt template incorporating example complaints and summaries to guide the generation.
- Implemented fallback logic for cases with insufficient context.
- Developed a test function for validating retrieval and answer generation on example questions.

### 4. Interactive Chat Interface with Gradio

- Created a user-friendly web interface using **Gradio** that includes:
  - A text input box for typing questions.
  - A "Submit" button to send queries.
  - Display area showing the AI-generated answer.
  - Display area listing the retrieved complaint chunks as sources for transparency and user trust.
  - A "Clear" button to reset the interface.

## 🛠️ Technologies Used

- Python 3.8+
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Sentence-Transformers](https://www.sbert.net/)
- [ChromaDB](https://docs.trychroma.com/)
- [Gradio](https://gradio.app/)
- CFPB Consumer Complaint Dataset

## 🚀 How to Run

1. Clone the repo and set up your Python environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

2. Prepare your vector store by running the data preprocessing and embedding pipeline (if not already done).

3. Launch the Gradio app:
    ```bash
    python scripts/app.py
    ```

4. Open the local URL (usually `http://127.0.0.1:7860`) in your browser to interact with the chatbot.


## 📈 Future Improvements

- Fine-tune the generator model on complaint data for better domain adaptation.
- Implement streaming token generation for faster, progressive answer display.
- Integrate multi-turn conversational context for dialogue coherence.
