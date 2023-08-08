# Tutoruslabs

This repository contains the code and analysis for my comprehensive work on Automated Essay Scoring through prompt engineering and LLM finetuning. I have conducted extensive research, data analysis, and engineering to develop and test various methodologies. Below is an overview of the files within this repository.

## Contents

1. **AES Scores Data Analysis**
   - File: `AES Scores Data Analysis Format.ipynb`
   - In this Jupyter Notebook, I have performed analysis of the AES scores obtained from different models and methods. I have compared statistical values and visualised them using MatPlotlib.

2. **Cohen Kappa and Confusion Matrix**
   - File: `Cohen Kappa and Confusion Matrix.ipynb`
   - In the short jupyter notebook, I calculated Cohen's Kappa metric and generated confusion matrices to assess the agreement between human graders and automated scoring methods. This analysis provides a comprehensive evaluation of the model performance.

3. **DeepL Translate**
   - File: `DeepL Translate.ipynb`
   - In this notebook, I used DeepL translation for formatting reponse data into the right language format.

4. **GPT_API Format**
   - File: `GPT_API_Format.ipynb`
   - This notebook contains the code for the OpenAI GPT-API calls I have used.

5. **Prompt Generator Format**
   - File: `Prompt_Generator_Format.ipynb`
   - In this Jupyter Notebook, I have designed a prompt generator that systematically creates diverse essay prompts. It is used to get singular prompts for chat bots or mass generate prompts for model training or api uses.

6. **Token Length Check**
   - File: `Token Length Check.ipynb`
   - This notebook contains the code for analysing token length distributions of input prompts to check before training specific models.

## Summary
I have engaged in fine-tuning Language Models (LMs) using the PEFT LoRA method. Also through prompt engineering and data analysis, I experimented on the feasability of using LLMs such as the OpenAi GPT models for Automated Essay Scoring.
I also had the opportunity to create large datasets using LLMs, formulating it to maximise efficiency in the machine learning process.
