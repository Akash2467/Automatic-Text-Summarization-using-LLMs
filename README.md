# Automatic-Text-Summarization-using-LLMs
This project focuses on abstractive text summarization using transformer-based Large Language Models (LLMs). Using the WikiHow dataset, we explore and evaluate multiple models including T5-small, T5-base, BART, and PEGASUS to generate concise summaries from long instructional texts.

Project Description
Manual summarization of textual content is time-consuming. With the increase in instructional and educational material online, there is a need for automated summarization systems. This project implements a pipeline using LLMs for summarization, evaluates model performance using standard metrics, and provides a demonstration-ready interface for testing.

Files in the Repository
File Name	Description
Project_2_Pre_Processing.ipynb	Performs text preprocessing: cleaning, lemmatization, and EDA with word clouds.
Project_2_model_training_50k_2e.ipynb	Trains T5-small model on 50,000 WikiHow samples for 2 epochs.
Project_2_t5_small_model_training_100k_3e.ipynb	Trains T5-small model on 100,000 samples for 3 epochs.
Project test gradio t5small_50K_2e.ipynb	Tests the 50K model and evaluates performance using ROUGE metrics.
Project_2_test_model_100k_3e.ipynb	Tests the 100K model and evaluates performance using ROUGE metrics.
PreTrainedT5BaseSummarizer (1).ipynb	Uses the pretrained T5-base model for inference and evaluation.
Bartpretrained.ipynb	Uses the pretrained BART model for summarization and evaluates ROUGE scores.
PegasusPretrained.ipynb	Uses the pretrained PEGASUS model for summarization and evaluates ROUGE scores.
Automatic Text Summarization using LLMs.pptx	Final presentation covering the project methodology, results, challenges, and conclusions.

Models Used
Model	Training Type	Dataset Used	Evaluation
T5-small (50K)	Fine-tuned (2 epochs)	50,000 samples	ROUGE
T5-small (100K)	Fine-tuned (3 epochs)	100,000 samples	ROUGE
T5-base	Pretrained only	-	ROUGE
BART	Pretrained only	-	ROUGE
PEGASUS	Pretrained only	-	ROUGE

Evaluation Metric
We used the ROUGE metric to evaluate the quality of generated summaries:

ROUGE-1: Unigram overlap

ROUGE-2: Bigram overlap

ROUGE-L: Longest common subsequence

Dataset
Source: WikiHow Summarization Dataset

Total Articles: Approximately 230,000

Used Samples: 50,000 and 100,000 (sampled for training)

Columns Used: Title, Headline, Text

Challenges Faced
GPU limitations on Colab restricted batch size and training epochs

Runtime disconnections affected model training stability

The large dataset required significant memory handling and sampling

Final Presentation
The full project summary, methodology, results, and future scope are available in
Automatic Text Summarization using LLMs.pptx

Libraries Used
transformers

nltk

rouge-score

pandas

numpy

scikit-learn

Future Work
Use larger models like T5-Large, BART-Large, or GPT-4

Add more evaluation metrics such as BLEU, METEOR, BERTScore

Include more diverse datasets such as news articles or research abstracts

Add features like summary length control and multilingual support
