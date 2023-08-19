#@ IMPORTING THE REQUIRED LIBRARIES AND DEPENDENCIES
import streamlit as st
from streamlit_chat import message
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Title and model/tokenizer loading
st.title("Flan ChatBot")

# Cache the model and tokenizer to improve performance
@st.cache_resource(show_spinner=True)
def load_model_tokenizer():
    peft_model_id = "flant_t5_large_chatbot_lora"
    config = PeftConfig.from_pretrained(peft_model_id)

    base_model_name_or_path = config.base_model_name_or_path
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

    model = PeftModel.from_pretrained(model, peft_model_id).to("cpu")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_tokenizer()

# Inference function
def inference(model, tokenizer, input_sent):
    input_ids = tokenizer(input_sent, return_tensors="pt", truncation=True, max_length=256).input_ids.to("cpu")
    outputs = model.generate(input_ids=input_ids, top_p=0.9, max_length=256)
    decoded_output = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
    return decoded_output

# Chatbot function
def chatbot():
  """Chatbot function."""
  message("Hi I am Flan T5 Chatbot. How can I help you?", is_user=False)       # greet the user
  placeholder = st.empty()                                                     # Create placeholder for user input
  input_ = st.text_input("Human")                                              # Get the user input

  # If the user clicks the "Generate" button
  if st.button("Generate"):
    # Display the user input
    with placeholder.container():
      message(input_, is_user=True)

    input_ = "Human: " + input_ + ". Assistant: "                    # Add the user input to the prompt

    # Generating a response
    with st.spinner(text="Generating Response.....  "):
      with placeholder.container():
        message(inference(model, tokenizer, input_), is_user=False)


if __name__ == "__main__":
  chatbot()
