import os
from datasets import load_dataset, Dataset, concatenate_datasets
import dask.dataframe as dd
import pandas as pd

LANGUAGES = [
    'asm', 'ben', 'brx', 'doi', 'eng', 'gom', 'guj', 'hin', 'kan', 'kas', 'mai',
    'mal', 'mar', 'mni', 'nep', 'ori', 'pan', 'san', 'sat', 'snd', 'tam', 'tel', 'urd'
]
CHUNK_SIZE = 80000 # Larger corpus
MIN_WORDS = 50
OUTPUT_PATH = "/home/coeadmin/Multiclass_roberta/data/medium_corpus.parquet"

def filter_text_length(text):
    return isinstance(text, str) and len(text.split()) >= MIN_WORDS

def load_and_save_corpus():
    try:
        datasets = []
        for lang in LANGUAGES:
            ds = load_dataset("ai4bharat/sangraha", "verified", split=lang, streaming=True)
            ds = ds.map(lambda x: {**x, "language": lang})
            ds = ds.take(500)  # ~500 per language to reach ~10K total
            datasets.append(Dataset.from_list(list(ds)))
        combined_ds = concatenate_datasets(datasets).shuffle(seed=42)
    except Exception as e:
        print(f"Error loading dataset: {e}. Ensure HF token is valid!")
        return
    
    # Convert to pandas then dask
    iterator = iter(combined_ds)
    chunk_data = []
    try:
        while len(chunk_data) < CHUNK_SIZE:
            row = next(iterator)
            text = row["text"]
            lang = row["language"]
            if filter_text_length(text):
                chunk_data.append({"text": text[:100000], "language": lang})
    except StopIteration:
        pass
    
    if chunk_data:
        # Use dask for parallel processing
        df = pd.DataFrame(chunk_data)
        ddf = dd.from_pandas(df, npartitions=4)  # Adjust based on CPU cores
        ddf = ddf.sample(frac=1, random_state=42).persist()  # Shuffle in parallel
        df = ddf.compute()  # Convert back to pandas for saving
        
        print(f"Loaded chunk with {len(df)} samples")
        for lang in LANGUAGES:
            count = len(df[df["language"] == lang])
            if count > 0:
                print(f"{lang}: {count}")
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df.to_parquet(OUTPUT_PATH)
        print(f"Corpus saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    load_and_save_corpus()