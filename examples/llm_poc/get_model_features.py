import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os



id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET = "fancyzhx/amazon_polarity"
BATCH_SIZE = 256
TOTAL_SIZE = 60000
TRAIN_SIZE = 50000
TEST_SIZE = 10000

def filename_from_str(st):
    return st.split("/")[1].replace("/","").replace("-", "").replace(".", "")

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2, id2label=id2label, label2id=label2id, dtype=torch.float32)

print(f"model loaded: {MODEL}")

dataset = datasets.load_dataset(DATASET)

print(f"dataset loaded: {DATASET}")

tok = AutoTokenizer.from_pretrained(MODEL)
# take a smaller sample
subsample = dataset["train"].train_test_split(
    train_size=TOTAL_SIZE, stratify_by_column="label", seed=0)["train"]

train_val_split = subsample.train_test_split(
    train_size=TRAIN_SIZE, stratify_by_column="label", seed=0)
tokenized_dataset = train_val_split
tokenized_dataset["train"] = train_val_split["train"].map(lambda x: tok(x["title"], truncation=True), batched=True)
print("a")
tokenized_dataset["validation"] = train_val_split["test"].map(lambda x: tok(x["title"], truncation=True), batched=True)
print("a")
if TEST_SIZE is not None:
    tokenized_dataset["test"] = dataset["test"].train_test_split(
        train_size=TEST_SIZE, stratify_by_column="label", seed=0)["train"]
tokenized_dataset["test"] = tokenized_dataset["test"].map(lambda x: tok(x["title"], truncation=True), batched=True)

print(f"dataset split and tokenized:\n {tokenized_dataset}")

# padding needed for minibatch training
data_collator = DataCollatorWithPadding(tokenizer=tok)

model.base_model.requires_grad_(False)
model.score.requires_grad_(True)
model.eval()   # score has no dropout, so eval on the whole model is fine

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(device)
model.to(device)

cols = ["input_ids", "attention_mask", "label"]  # collator renames label -> labels
train_dl = DataLoader(tokenized_dataset["train"].select_columns(cols), batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
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
            # shape [B, L, 768]
            batch_last_hidden_states = model.base_model(**batch).last_hidden_state
            # find the index of the last non-padding token so we get the state after it
            batch_last_nonpadding_token_idxs = batch['attention_mask'].sum(1) - 1
            # shape [L, 768]
            for per_sentence_last_hidden_states, last_nonpadding_token_idx in zip(batch_last_hidden_states, batch_last_nonpadding_token_idxs):
                per_sentence_transformer_feats.append(
                    per_sentence_last_hidden_states[last_nonpadding_token_idx]
                )
            labels.append(batch['labels'])
    return per_sentence_transformer_feats, labels


saved_feats_path = os.path.join("saved_features", filename_from_str(DATASET), filename_from_str(MODEL))
os.makedirs(saved_feats_path, exist_ok=True)
print(f"Path ready: will be saved as {saved_feats_path}/*.pt")

for subset, loader in loaders.items():
    print(f"Starting extraction: {subset}")
    scorer_feats, scorer_labels = extract_llm_features(loader)
    print(f"{subset} finished, stacking")
    scorer_feats = torch.vstack(scorer_feats)
    scorer_labels = torch.concat(scorer_labels)
    print(f"saving {subset}")
    torch.save(scorer_feats, os.path.join(saved_feats_path, f"feat_{subset}.pt"))
    torch.save(scorer_labels, os.path.join(saved_feats_path, f"labels_{subset}.pt"))