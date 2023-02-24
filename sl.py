import streamlit as st

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

@st.cache_resource
def load_model(model_name = "google/flan-t5-large"):
    return (
        AutoTokenizer.from_pretrained(model_name),
        AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    )

def generate(prompt):
    tokenizer, model = load_model()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(
        input_ids, max_length=512, min_length=8, top_p=0.9, do_sample=True
    )
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output


st.title('Chatbot')

query = st.text_input("User:")

prompt = f"Computer is an AI system that always responds helpfully.\n\n" \
         f"User: {query}\n" \
         f"Computer: "

st.write(generate(prompt))
