# IndicXLMR
IndicXLMR-LoRA is a lightweight, parameter-efficient model designed for language identification across 23 Indic languages, built on top of XLM-RoBERTa.
It leverages Low-Rank Adaptation (LoRA) to finetune a pre-trained multilingual model on the ai4bharat/sangraha dataset, achieving high accuracy and F1 scores with minimal computational overhead. The pipeline includes:

Corpus Creation: A scalable dataset from verified Indic texts, processed with Dask for efficiency.
Pre-training: Continued masked language modeling on Indic data to adapt XLM-RoBERTa to regional nuances (perplexity ~7).(OnGoing)
Finetuning: LoRA-based finetuning for sequence classification, targeting language ID with ~90-94% accuracy on a small corpus.(OnGoing)
