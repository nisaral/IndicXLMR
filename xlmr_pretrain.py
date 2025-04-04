import os
os.environ["USE_APEX"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
from transformers import XLMRobertaForMaskedLM, XLMRobertaTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from datasets import Dataset

INPUT_PATH = "/home/coeadmin/Multiclass_roberta/data/medium_corpus.parquet"
OUTPUT_DIR = "/home/coeadmin/Multiclass_roberta/output/xlmr_pretrained_medium"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def pretrain_xlmr():
    # Load corpus
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} samples from {INPUT_PATH}")
    
    # Initialize model and tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    model = XLMRobertaForMaskedLM.from_pretrained("xlm-roberta-base")
    
    # Prepare datasets
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    eval_df = df[train_size:]
    
    train_encodings = tokenizer(train_df["text"].tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt")
    eval_encodings = tokenizer(eval_df["text"].tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt")
    train_dataset = Dataset.from_dict({"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"]})
    eval_dataset = Dataset.from_dict({"input_ids": eval_encodings["input_ids"], "attention_mask": eval_encodings["attention_mask"]})
    
    # Training setup
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=8,  # Larger batch size
        per_device_eval_batch_size=8,
        eval_strategy="steps",
        eval_steps=100,  # Less frequent eval
        logging_steps=100,
        save_steps=200,
        learning_rate=1e-5,
        max_steps=500,  # Fewer steps for speed
        report_to="none",
        fp16=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train
    torch.cuda.empty_cache()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    free_memory = (total_memory - allocated_memory) / 1024**3
    print(f"Free GPU memory before train: {free_memory:.2f} GiB")
    trainer.train()
    
    eval_results = trainer.evaluate()
    eval_loss = eval_results["eval_loss"]
    perplexity = math.exp(eval_loss)
    print(f"Pre-training complete, Eval Loss: {eval_loss:.4f}, Perplexity: {perplexity:.4f}")
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Pre-trained model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    pretrain_xlmr()