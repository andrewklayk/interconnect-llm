import datasets
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from common import MODEL, FEAT_DIR, DATA

DATASET = "fancyzhx/amazon_polarity"
BATCH_SIZE = 200
TOTAL_SIZE = 6000
TRAIN_SIZE = 5000
TEST_SIZE = 1000

model = AutoModel.from_pretrained(MODEL, dtype=torch.float32)

print(f"model loaded: {MODEL}")

dataset = datasets.load_dataset(DATASET)

print(f"dataset loaded: {DATASET}")

tok = AutoTokenizer.from_pretrained(MODEL)
# take a smaller sample
print("subsampling...")
subsample = dataset["train"].train_test_split(
    train_size=TOTAL_SIZE, stratify_by_column="label", seed=0)["train"]

print("splitting...")
train_val_split = subsample.train_test_split(
    train_size=TRAIN_SIZE, stratify_by_column="label", seed=0)
tokenized_dataset = train_val_split

print("tokenizing...")
def tokenize_cut(x, i):
    # cut at a random length (seeded by row index, so map caching stays valid):
    # the loop's fixed-length responses often end mid-clause
    ids = tok(x["content"])["input_ids"][:np.random.default_rng(i).integers(10, 41)]
    return {"input_ids": ids, "attention_mask": [1] * len(ids)}

tokenized_dataset["train"] = train_val_split["train"].map(tokenize_cut, with_indices=True)
tokenized_dataset["validation"] = train_val_split["test"].map(tokenize_cut, with_indices=True)
if TEST_SIZE is not None:
    tokenized_dataset["test"] = dataset["test"].train_test_split(
        train_size=TEST_SIZE, stratify_by_column="label", seed=0)["train"]
else:
    tokenized_dataset["test"] = dataset["test"]
tokenized_dataset["test"] = tokenized_dataset["test"].map(tokenize_cut, with_indices=True)

print("saving to file...")
tokenized_dataset['train'].save_to_disk(DATA / "amazon_polarity_content_mean" / "train")
tokenized_dataset['validation'].save_to_disk(DATA / "amazon_polarity_content_mean" / "validation")
tokenized_dataset['test'].save_to_disk(DATA / "amazon_polarity_content_mean" / "test")

print(f"dataset split and tokenized:\n {tokenized_dataset}")

# padding needed for minibatch training
data_collator = DataCollatorWithPadding(tokenizer=tok)

model.eval()

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(device)
model.to(device)

cols = ["input_ids", "attention_mask", "label"]  # collator renames label -> labels
train_dl = DataLoader(tokenized_dataset["train"].select_columns(cols), batch_size=BATCH_SIZE, collate_fn=data_collator)
val_dl = DataLoader(tokenized_dataset["validation"].select_columns(cols), batch_size=BATCH_SIZE, collate_fn=data_collator)
test_dl = DataLoader(tokenized_dataset["test"].select_columns(cols), batch_size=BATCH_SIZE, collate_fn=data_collator)

loaders = {
    "train": train_dl,
    "val": val_dl,
    "test": test_dl
}

def extract_llm_features(dataloader):
    per_sentence_transformer_feats = []
    labels = []
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            # shape [B, L, d]
            H = model(**batch).last_hidden_state
            # mean over real tokens: less sensitive to where the text is cut than the last state
            m = batch['attention_mask'].unsqueeze(-1)
            per_sentence_transformer_feats.append((H * m).sum(1) / m.sum(1))
            labels.append(batch['labels'])
    return per_sentence_transformer_feats, labels


FEAT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Path ready: will be saved as {FEAT_DIR}/*.pt")

for subset, loader in loaders.items():
    print(f"Starting extraction: {subset}")
    scorer_feats, scorer_labels = extract_llm_features(loader)
    print(f"{subset} finished, stacking")
    scorer_feats = torch.vstack(scorer_feats)
    scorer_labels = torch.concat(scorer_labels)
    print(f"saving {subset}")
    torch.save(scorer_feats, FEAT_DIR / f"feat_{subset}.pt")
    torch.save(scorer_labels, FEAT_DIR / f"labels_{subset}.pt")