# Text Summarizer and Paraphraser

This repository contains a user-friendly NLP toolkit that leverages cutting-edge machine learning models for text processing tasks, including paraphrase detection and text summarization. The toolkit is designed for researchers, developers, and enthusiasts interested in natural language understanding and generation.

## Features
- **Paraphrase Detection:** 
  Utilizes a fine-tuned BERT model to assess whether two sentences convey the same meaning, making it an essential tool for content editing and duplicate detection.
  
- **Text Summarization:** 
  Employs the BART transformer model to condense extensive documents into concise summaries, enhancing the efficiency of information retrieval.

- **User Interface:**
  Built with [Streamlit](https://streamlit.io/), the application provides an interactive and intuitive web-based interface for easy text processing.

## Models Used
1. **BERT (Bidirectional Encoder Representations from Transformers):**
   - Analyzes sentence pairs to identify paraphrases.
   - Fine-tuned on the MRPC dataset from the GLUE benchmark.

2. **BART (Bidirectional and Auto-Regressive Transformers):**
   - Excels in text generation and understanding.
   - Used for summarizing long texts effectively.

## Datasets
- **Paraphrase Detection:** 
  [MRPC](https://huggingface.co/datasets/glue/viewer/mrpc) from the GLUE benchmark (via Hugging Face `datasets` library).
  
- **Text Summarization:**
  General corpus pre-trained data used by BART.

## How to Use
1. Clone this repository or download the files directly: git clone https://github.com/tanvisharmaaa/Text-Summarizer-and-Paraphraser.git

2. Install the dependencies

3. Run the Streamlit app

4. Explore the paraphrase detection and text summarization features through the web interface.

## File Overview
- `app.py`: Main Streamlit application.
- `Text_summarizer.py`: Implements BART-based text summarization logic.
- `Paraphrase_checker.ipynb`: Jupyter Notebook for BERT paraphrase detection.
- `bert.ipynb`: Code for fine-tuning BERT on the MRPC dataset.

## Results and Insights
### Paraphrase Detection:
- Fine-tuned BERT achieved high accuracy in identifying paraphrases from the MRPC dataset.

### Text Summarization:
- Summarized lengthy texts effectively while preserving context and key details using BART.

## Future Work
- Extend the toolkit to support multilingual summarization and paraphrasing.
- Add additional NLP features like sentiment analysis and named entity recognition.

## Acknowledgments
- BERT: Developed by Google AI (2018).
- BART: Developed by Facebook AI.
- Streamlit: For providing a seamless UI framework.




