import streamlit as st
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import torch

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

model_path = 'results/checkpoint-1000'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.eval()

def predict_paraphrase(sentence1, sentence2):
    inputs = tokenizer.encode_plus(
        sentence1, sentence2, add_special_tokens=True, max_length=512, padding='max_length', return_attention_mask=True, return_tensors='pt',
    )

    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1).cpu().numpy()[0]
    paraphrase_mapping = {0: 'not a paraphrase', 1: 'paraphrase'}
    return paraphrase_mapping[prediction], probabilities.cpu().numpy()

st.title("NLP Toolkit")

st.header("Text Summarization")
text_to_summarize = st.text_area("Enter text to summarize")
if text_to_summarize:
    summary = summarizer(text_to_summarize, max_length=130, min_length=30, do_sample=False)
    st.write("Summary:", summary[0]['summary_text'])

st.header("Paraphrased senetence or not?")
sentence1 = st.text_area("Text 1")
sentence2 = st.text_area("Text 2")
if sentence1 and sentence2:
    predicted_paraphrase, probabilities = predict_paraphrase(sentence1, sentence2)
    st.write(f"Prediction: {predicted_paraphrase}")
    st.write(f"Probability of 'not paraphrased': {probabilities[0][0]:.4f}")
    st.write(f"Probability of 'paraphrased': {probabilities[0][1]:.4f}")

if __name__ == "__main__":
    st.write("Ready to perform NLP tasks")

