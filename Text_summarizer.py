from transformers import pipeline

summarization_service = pipeline("summarization", model="facebook/bart-large-cnn")

text = """
His record as prime minister in the past decade belies that. 
Now the mask has fallen completely. In a recent campaign rally in Rajasthan, 
Modi made an exceptionally incendiary speech in which he claimed that his predecessor,
 Manmohan Singh, had declared that Muslims had “the first claim” to the nation’s resources. 
 This was distortion and exaggeration. The reference was to a speech that Singh had made in 
 2006 about India’s development priorities.
"""

summarized_text = summarization_service(text, max_length=130, min_length=30, do_sample=False)

print(summarized_text[0]['summary_text'])

