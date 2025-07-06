# Compliant Answering Chatbot for Financial Complaints

This project implements a **Retrieval-Augmented Generation (RAG)**-based chatbot that answers financial complaints using real-world complaint narratives from the **CFPB** (Consumer Financial Protection Bureau) dataset. The chatbot is designed to assist financial analysts or customer service teams in providing helpful and compliant responses to customer concerns.

## 📌 Project Objectives

- Build an intelligent chatbot that answers financial complaints using retrieved context.
- Enhance reliability by grounding answers in actual complaint data (RAG approach).
- Prepare the model for deployment or integration into a customer-facing interface.

## ✅ Completed Tasks

### **Exploratory Data Analysis and Preprocessing**

- Loaded and explored the CFPB complaint dataset.
- Focused on cleaning the `consumer_complaint_narrative` field.
- Removed nulls, duplicates, and overly short/irrelevant narratives.
- Saved a clean dataset for downstream tasks.

### **Text Chunking, Embedding & Vector Store Indexing**

- Split long complaint texts into manageable chunks using overlapping windows.
- Generated semantic embeddings using a pre-trained SentenceTransformer model.
- Indexed the chunks into a FAISS (or ChromaDB) vector store for efficient retrieval.
- Added metadata like original complaint ID and theme for traceability.

### **RAG Pipeline (Retrieval-Augmented Generation)**

- Implemented a full RAG pipeline:
  - **Retriever**: Fetches top-k relevant complaint chunks using vector similarity.
  - **Generator**: GPT-2 (or similar) language model used to answer based on the retrieved context.
- Designed a custom prompt template to guide generation.
- Added fallback logic: if the context doesn’t contain the answer, the model says so.
- Created a test function (`test_rag_pipeline`) to validate the chatbot's performance on various user queries.

## 🛠️ Technologies Used

- `transformers` by Hugging Face
- `sentence-transformers`
- `ChromaDB`
- Python (PyTorch backend)
- CFPB dataset

## 🚧 Next Steps

- Fine-tune or swap in a more instruction-tuned model (e.g. T5, FLAN-T5).
- Add UI using Streamlit.
- Deploy via FastAPI backend (optional).
- Evaluate chatbot accuracy and hallucination rate.

