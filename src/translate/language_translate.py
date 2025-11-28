import os
import csv
import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

# ---- Device info (your GPU is OK now) ----
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Torch CUDA runtime:", torch.version.cuda)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Device selected: {DEVICE}")
if DEVICE == "mps":
    print("Using Apple Metal (MPS) backend on Apple Silicon GPU")

HF_TOKEN = os.getenv(
    "HUGGING_FACE_HUB_TOKEN",
    ""
    #USE YOUR OWN HUGGING_FACE_HUB_TOKEN
    # "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"
)

src_lang, tgt_lang = "tam_Taml", "eng_Latn"
model_name = "ai4bharat/indictrans2-indic-en-1B"

# ---- Tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    token=HF_TOKEN,
)

# ---- Model (NO flash_attention_2) ----
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    token=HF_TOKEN,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)

model.to(DEVICE)

# Disable cache to avoid past_key_values bug
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

ip = IndicProcessor(inference=True)

def _iter_texts_from_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            txt = (row.get("text") or "").strip()
            if txt:
                yield txt


def _batchify(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

# change this to your system's input csv path
csv_path = os.environ.get(
    "DINAMALAR_CSV_PATH",
    "/Users/rsaravanakumar/Applications/personal/mtech/bias-detection-in-news-media/data/raw/dinamalar_articles_2024_to_2024.csv",
)
batch_size = int(os.environ.get("BATCH_SIZE", "1"))

# change this to your system's output csv path

out_path = os.environ.get("OUTPUT_CSV_PATH", "./translations_output.csv")

print(f"Source language: {src_lang}; Target language: {tgt_lang}; Model: {model_name}")
print(f"Input CSV path: {csv_path}")
print(f"Output CSV path: {out_path}")
print(f"Batch size: {batch_size}")

with open(out_path, "w", encoding="utf-8", newline="") as outf:
    writer = csv.writer(outf)
    writer.writerow(["input_text", "translated_text"])  # header
    print("Wrote CSV header: ['input_text', 'translated_text']")

    for batch_idx, input_sentences in enumerate(_batchify(_iter_texts_from_csv(csv_path), batch_size), start=1):
        print(f"Processing batch {batch_idx} with {len(input_sentences)} sentences")
        batch = ip.preprocess_batch(
            input_sentences,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )

        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=False,
                min_length=0,
                max_length=256,
                num_beams=4,
                num_return_sequences=1,
            )
        print(f"Generated tokens for batch {batch_idx}")

        generated_tokens = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)
        print(f"Writing {len(translations)} rows for batch {batch_idx}")

        for inp, out in zip(input_sentences, translations):
            writer.writerow([inp, out])
            # Optional: also print to console
            print(f"{src_lang}: {inp}")
            print(f"{tgt_lang}: {out}")

print(f"Wrote translations to: {out_path}")
