from sentence_transformers import SentenceTransformer

# Reload the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_relevant_chunks(question, model, collection, k=5):
    # Step 1: Embed the question
    question_embedding = model.encode(question)

    # Step 2: Perform similarity search in the vector store
    results = collection.query(question_embedding.tolist(), n_results=k)

    # Step 3: Extract relevant chunks and metadata
    retrieved_chunks = results['documents'][0]  # Access the first element of the 'documents' list
    retrieved_ids = [meta['complaint_id'] for meta in results['metadatas'][0]]  # Extract complaint IDs

    return retrieved_chunks, retrieved_ids


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

generation_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")


def generate_answer(question, retrieved_chunks, model, tokenizer):
    if not retrieved_chunks:
        return "I don't have enough information to answer that question."
    
    context = "\n".join(f"- {chunk.strip().replace('\n', ' ')}" for chunk in retrieved_chunks)
    
    prompt_template = f"""
You are a helpful customer support assistant.

You receive the following customer complaint excerpts related to credit cards, personal loans, Buy Now Pay Later services, savings accounts, and money transfers:

Customer Complaints:
{context}

Based solely on the above complaints, answer the user's question clearly and concisely. If the complaints do not provide enough information to answer, say: "I don't have enough information to answer that question."

Question:
{question}

Answer:
"""
    inputs = tokenizer(prompt_template, return_tensors="pt", truncation=True, max_length=512)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        no_repeat_ngram_size=3,
    )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer + "\n\nIf you have any further questions or concerns, please let us know."



def test_rag_pipeline(test_question, embedding_model, generation_model, tokenizer, collection):
    # Step 1: Retrieve relevant chunks using SentenceTransformer
    retrieved_chunks, retrieved_ids = retrieve_relevant_chunks(test_question, embedding_model, collection)
    
    # Step 2: Generate answer using GPT-2
    answer = generate_answer(test_question, retrieved_chunks, generation_model, tokenizer)
    
    # Print results
    print("Retrieved Chunks:")
    for chunk in retrieved_chunks:
        print("-", chunk)
    
    print("\nGenerated Answer:")
    print(answer)