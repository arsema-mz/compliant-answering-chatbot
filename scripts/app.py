import gradio as gr
from rag_pipeline import retrieve_relevant_chunks, generate_answer, model as embedding_model, generation_model, tokenizer
import chromadb

# Load ChromaDB collection
db = chromadb.PersistentClient(path="vector_store")
collection = db.get_or_create_collection("complaints")

def rag_chat(question):
    # Step 1: Retrieve relevant chunks
    retrieved_chunks, _ = retrieve_relevant_chunks(question, embedding_model, collection)

    # Step 2: Generate answer
    answer = generate_answer(question, retrieved_chunks, generation_model, tokenizer)

    # Prepare sources for display
    sources = "\n\n".join([f"- {chunk}" for chunk in retrieved_chunks])

    return answer, sources

def clear():
    return "", "", ""

# Gradio interface
description = """
### 💬 RAG Customer Complaint Assistant
Ask a question based on your complain. The assistant retrieves relevant complaints and generates a helpful summary with suggestions.
"""
with gr.Blocks() as demo:
    gr.Markdown(description)

    with gr.Row():
        question_input = gr.Textbox(label="Enter your question", placeholder="e.g. What are the issues with Buy Now Pay Later services?", lines=2)

    with gr.Row():
        submit_btn = gr.Button("Submit")
        clear_btn = gr.Button("Clear")

    answer_output = gr.Textbox(label="Generated Answer", lines=6)
    sources_output = gr.Textbox(label="Retrieved Sources (Complaint Chunks)", lines=10)

    submit_btn.click(fn=rag_chat, inputs=question_input, outputs=[answer_output, sources_output])
    clear_btn.click(fn=clear, outputs=[question_input, answer_output, sources_output])

if __name__ == "__main__":
    demo.launch()
