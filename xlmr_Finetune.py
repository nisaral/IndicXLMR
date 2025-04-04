# xlmr_finetune_lora.py
import os
os.environ["USE_APEX"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, XLMRobertaForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict  # Added get_peft_model_state_dict
from sklearn.metrics import accuracy_score, f1_score
import torch
from datasets import Dataset
import math

INPUT_PATH = "/home/coeadmin/Multiclass_roberta/data/medium_corpus.parquet"
PRETRAINED_PATH = "/home/coeadmin/Multiclass_roberta/output/xlmr_pretrained_medium"
FINETUNE_DIR = "/home/coeadmin/Multiclass_roberta/output/finetuned_language_id_medium_lora"
LANGUAGES = [
    'asm', 'ben', 'brx', 'doi', 'eng', 'gom', 'guj', 'hin', 'kan', 'kas', 'mai',
    'mal', 'mar', 'mni', 'nep', 'ori', 'pan', 'san', 'sat', 'snd', 'tam', 'tel', 'urd'
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def compute_perplexity(model, tokenizer, eval_df):
    mlm_model = XLMRobertaForMaskedLM.from_pretrained(PRETRAINED_PATH)
    # Use get_peft_model_state_dict to extract base model weights, excluding adapters
    base_state_dict = get_peft_model_state_dict(model, adapter_name="default")
    mlm_model.roberta.load_state_dict(base_state_dict, strict=False)  # Allow missing keys (e.g., pooler)
    
    encodings = tokenizer(eval_df["text"].tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt")
    eval_dataset = Dataset.from_dict({"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"]})
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    
    trainer = Trainer(
        model=mlm_model,
        args=TrainingArguments(
            output_dir="/tmp",
            per_device_eval_batch_size=8,
            fp16=True,
            report_to="none",
        ),
        data_collator=data_collator,
        eval_dataset=eval_dataset,
    )
    eval_results = trainer.evaluate()
    eval_loss = eval_results["eval_loss"]
    perplexity = math.exp(eval_loss)
    return perplexity

def finetune_language_id():
    # Load corpus
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} samples from {INPUT_PATH}")
    label2id = {lang: idx for idx, lang in enumerate(LANGUAGES)}
    df["label"] = df["language"].map(label2id)
    
    # Prepare datasets
    lang_id_dataset = Dataset.from_pandas(df[["text", "label"]])
    train_size = int(0.8 * len(lang_id_dataset))
    train_data = lang_id_dataset.select(range(train_size))
    eval_data = lang_id_dataset.select(range(train_size, len(lang_id_dataset)))
    eval_df = df[train_size:]
    
    # Load pre-trained model and tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(PRETRAINED_PATH)
    model = XLMRobertaForSequenceClassification.from_pretrained(PRETRAINED_PATH, num_labels=len(LANGUAGES))
    
    # Add LoRA adapter
    lora_config = LoraConfig(
        task_type="SEQ_CLS",
        r=8,
        lora_alpha=16,
        target_modules=["query", "key", "value"],
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config)
    
    # Tokenize
    train_data = train_data.map(lambda x: tokenizer(x["text"], truncation=True, padding=True, max_length=512), batched=True)
    eval_data = eval_data.map(lambda x: tokenizer(x["text"], truncation=True, padding=True, max_length=512), batched=True)
    train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    eval_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    # Training setup
    training_args = TrainingArguments(
        output_dir=FINETUNE_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        learning_rate=2e-5,
        fp16=True,
        report_to="none",
    )
    
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average="weighted")}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
    )
    
    # Train
    torch.cuda.empty_cache()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    free_memory = (total_memory - allocated_memory) / 1024**3
    print(f"Free GPU memory before finetune: {free_memory:.2f} GiB")
    trainer.train()
    
    # Evaluate classification
    eval_results = trainer.evaluate()
    print(f"Language ID - Accuracy: {eval_results['eval_accuracy']:.4f}, F1: {eval_results['eval_f1']:.4f}")
    
    # Compute perplexity
    perplexity = compute_perplexity(model, tokenizer, eval_df)
    print(f"Finetuned Perplexity: {perplexity:.4f}")
    
    # Save
    os.makedirs(FINETUNE_DIR, exist_ok=True)
    model.save_pretrained(FINETUNE_DIR)
    tokenizer.save_pretrained(FINETUNE_DIR)
    print(f"Finetuned model saved to {FINETUNE_DIR}")

if __name__ == "__main__":
    finetune_language_id()