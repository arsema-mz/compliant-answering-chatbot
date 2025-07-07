from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Try flan-t5-base or flan-t5-small if your connection is slow
model_name = "google/flan-t5-base"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

prompt = (
    "You are a helpful assistant. Respond to the customer's complaint below in a polite and helpful manner.\n\n"
    "Complaint: I was charged late fees even though I paid on time. I contacted support and they refused to help.\n\n"
    "Response:"
)

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

print("Generating response...")
outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    temperature=0.3,
    top_p=0.9
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nGenerated Response:\n", response)
