# Automatic Text Summarization using LLMs

This project focuses on abstractive text summarization using transformer-based Large Language Models (LLMs). The goal is to generate concise and meaningful summaries from long instructional texts using models like T5-small, T5-base, BART, and PEGASUS. The dataset used is sourced from WikiHow and includes article text paired with summaries.

---

## Project Description

Manual summarization is time-consuming, especially with the growing volume of online content. This project builds an automated text summarization system using transformer models. We fine-tune T5-small on subsets of the WikiHow dataset and compare it against pretrained models. ROUGE metrics are used for evaluation, and interactive testing is supported through notebook-based interfaces.

---

## Files in the Repository

| File Name                                      | Description |
|------------------------------------------------|-------------|
| `Project_2_Pre_Processing.ipynb`              | Performs text preprocessing: cleaning, lemmatization, and basic EDA including word clouds. |
| `Project_2_model_training_50k_2e.ipynb`       | Trains the T5-small model on 50,000 samples for 2 epochs. |
| `Project_2_t5_small_model_training_100k_3e.ipynb` | Trains the T5-small model on 100,000 samples for 3 epochs. |
| `Project test gradio t5small_50K_2e.ipynb`    | Tests the T5-small model trained on 50K and evaluates with ROUGE scores. |
| `Project_2_test_model_100k_3e.ipynb`          | Tests the T5-small model trained on 100K and evaluates with ROUGE scores. |
| `PreTrainedT5BaseSummarizer (1).ipynb`        | Uses the pretrained T5-base model for inference and evaluation. |
| `Bartpretrained.ipynb`                        | Uses the pretrained BART model for summarization and evaluation. |
| `PegasusPretrained.ipynb`                     | Uses the pretrained PEGASUS model for summarization and evaluation. |
| `Automatic Text Summarization using LLMs.pptx`| Final presentation describing the methodology, experiments, results, and challenges. |

---

## Models Used

| Model             | Training Type        | Dataset Used     | Evaluation |
|-------------------|----------------------|------------------|------------|
| T5-small (50K)    | Fine-tuned (2 epochs)| 50,000 samples   | ROUGE      |
| T5-small (100K)   | Fine-tuned (3 epochs)| 100,000 samples  | ROUGE      |
| T5-base           | Pretrained only      | -                | ROUGE      |
| BART              | Pretrained only      | -                | ROUGE      |
| PEGASUS           | Pretrained only      | -                | ROUGE      |

---

## Evaluation Metric

We used the ROUGE metric to evaluate the quality of the generated summaries:

- **ROUGE-1**: Measures unigram (word-level) overlap.
- **ROUGE-2**: Measures bigram (two-word sequences) overlap.
- **ROUGE-L**: Measures the longest common subsequence between model output and reference summary.

---

## Dataset

- **Source**: [WikiHow Summarization Dataset](https://www.kaggle.com/datasets/varunucl/wikihow-summarization)
- **Total Articles**: ~230,000
- **Used Samples**: 50,000 and 100,000 for training
- **Columns Used**: `Title`, `Headline`, `Text`

---

## Challenges Faced

- GPU limitations in Google Colab restricted training time and batch size.
- Runtime disconnections interrupted model training.
- The large dataset required memory-efficient sampling and preprocessing.

---

## Final Presentation

For an overview of the project including objectives, methodology, results, and future directions, refer to:  
[Automatic Text Summarization using LLMs.pptx](./Automatic%20Text%20Summarization%20using%20LLMs.pptx

---

## Libraries Used

- transformers
- nltk
- rouge-score
- pandas
- numpy
- scikit-learn

---

## Future Work

- Train with larger models like T5-Large or GPT-based models.
- Use additional evaluation metrics such as BLEU, METEOR, and BERTScore.
- Expand to multilingual summarization and broader domains.
- Add UI enhancements such as summary length control and feedback system.
Note: Trained model files are not included due to size constraints. Please refer to the notebooks for training and inference code.
