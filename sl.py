import streamlit as st

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

from wikiqabot import get_proper_nouns, get_summary, get_topn_similar, search, sentencer

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


st.title('WikiQABot')

query = st.text_input("Ask me anything!", value="")

if query:
    nouns = get_proper_nouns(query)

    knowledge = []

    if nouns:
        sentences = []

        for word in nouns:
            for result in search(word)[:1]:
                print(f"Getting summary for {word} ({result['id']})")
                sentences += [
                    str(s) for s in sentencer(get_summary(result["id"])).sents
                ]

        if sentences:
            knowledge = get_topn_similar(query, sentences, 4)

    prompt = f"Context: {' '.join(knowledge)}\n\n" \
         f"Question: {query}\n\n" \
         f"Answer: "

    st.write(prompt)

    st.write(generate(prompt))
