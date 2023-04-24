import streamlit as st
import os
import subprocess
#from tokenizers.trainers import WordPieceTrainer
#from tokenizers import Tokenizer
#from tokenizers.models import BPE
from transformers import TFGPT2LMHeadModel
from transformers import GPT2Tokenizer, TFGPT2Model
import tensorflow as tf

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2LMHeadModel.from_pretrained('gpt2')
text = "Once upon a time, there was a"
encoded_input = tokenizer.encode(text, return_tensors='tf')
output = model(encoded_input)

# Clone the repository
if not os.path.exists("GPT"):
    subprocess.run(["git", "clone", "https://github.com/urvimehta20/GPT.git"])

# Change directory to the cloned repository
os.chdir("GPT")

# Install dependencies
subprocess.run(["pip", "install", "-r", "requirements.txt"])

# Create a Streamlit app
def main():
    st.title("GPT")
    st.sidebar.title("Instructions:")
    st.sidebar.write("Please use generate button to obtain desired results.")
    st.sidebar.write("Results generated are solely based on the news dataset used.")
    # Input prompt
    prompt = st.text_input("Type your mind out and let the model generate text for you.", value="", max_chars=100)
    input_ids = tokenizer.encode(prompt, return_tensors='tf')
    
   
    if st.button("Generate"):
        if not prompt:
            st.warning("Please enter a prompt.")
        else:
            # Call the generate_text function from the repository
            #generated_text = generate_text(prompt)
            if tf.shape(input_ids).numpy().size == 2:
            
                output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=1)
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                st.write("Generated Text:")
                st.write(generated_text) 
            else:
                st.warning("Invalid input. Please try again.")


if __name__ == '__main__':
    main()
