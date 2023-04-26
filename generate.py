from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Input prompt
input_prompt = "Once upon a time, there was a"

# Tokenize input prompt
input_ids = tokenizer.encode(input_prompt, return_tensors='pt')

# Generate text
output = model.generate(input_ids, max_length=100, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print generated text
print("Generated Text:")
print(generated_text)

