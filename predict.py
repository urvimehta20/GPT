from transformers import TextGenerationPipeline
from transformers import GPT2Tokenizer
from train import init_model, load_tokenizer

tokenizer = load_tokenizer()
model = init_model(tokenizer)

text_generator = TextGenerationPipeline(model, tokenizer)
print(text_generator("Hello", max_length=64, do_sample=True, top_k=10, eos_token_id=tokenizer.get_vocab().get("】", 0)))
print(text_generator("I'm here", max_length=64, do_sample=True, top_k=10, eos_token_id=tokenizer.get_vocab().get("】", 0)))
print(text_generator("a crow", max_length=64, do_sample=True, top_k=10, eos_token_id=tokenizer.get_vocab().get("】", 0)))
print(text_generator("go to the forest ", max_length=64, do_sample=False))
print(text_generator("today", max_length=64, do_sample=False))
print(text_generator("Great river goes east", max_length=64, do_sample=False))
