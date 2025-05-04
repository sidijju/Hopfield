import torch
import re
import spacy
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, DataCollatorWithPadding
from networks import NextTokenPredictor, SequenceClassifier
 
class Benchmark():
    def __init__(self, config):
        self.config = config

        # Tokenizer for all benchmarks / models
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(self, model, dataloader, lr=1e-3):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        model.train()

        for epoch in range(self.config['training']['epochs']):
            total_loss = 0
            for step, batch in enumerate(tqdm(dataloader)):
                input_ids, labels = batch["input_ids"], batch["labels"]
                input_ids = input_ids.to(self.config['hardware']['device'])
                labels = labels.to(self.config['hardware']['device'])

                optimizer.zero_grad()
                logits = model(input_ids)

                logits = logits.view(-1, logits.shape[-1])
                labels = labels.view(-1)

                loss = loss_fn(logits, labels)

                loss.backward()
                optimizer.step()
                
                if step % 10 == 0:  
                    wandb.log({
                        "batch_loss": loss.item(),
                        "epoch": epoch + (step / len(dataloader))
                    })

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            wandb.log({
                "epoch": epoch + 1,
                "epoch_loss": avg_loss
            })
            print(f"Epoch {epoch+1}/{self.config['training']['epochs']} | Loss: {avg_loss:.4f}")

    def evaluate(self, model, dataloader):
        model.eval()
        total_tokens = 0
        correct_tokens = 0

        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids, labels = batch["input_ids"], batch["labels"]
                input_ids = input_ids.to(self.config['hardware']['device'])
                labels = labels.to(self.config['hardware']['device'])

                logits = model(input_ids)
                predictions = logits.argmax(-1)

                mask = (labels != self.tokenizer.pad_token_id)
                correct = (predictions == labels) & mask

                total_tokens += mask.sum().item()
                correct_tokens += correct.sum().item()

        accuracy = 100.0 * correct_tokens / total_tokens
        wandb.log({
            "test_accuracy": accuracy,
            "total_tokens": total_tokens,
            "correct_tokens": correct_tokens
        })
        wandb.run.summary.update({
            "final_accuracy": accuracy
        })
        print(f"Top-1 accuracy: {accuracy:.2f}%")

    def train_sequence_classifier(self, model, dataloader, lr=1e-3):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        model.train()

        for epoch in range(self.config['training']['epochs']):
            total_loss = 0
            correct = 0
            total = 0

            for step, (input_ids, labels) in enumerate(tqdm(dataloader)):
                input_ids = input_ids.to(self.config['hardware']['device'])
                labels = labels.to(self.config['hardware']['device'])

                optimizer.zero_grad()
                logits = model(input_ids)  # [batch_size, num_classes]
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                if step % 10 == 0:
                    batch_accuracy = 100.0 * (predictions == labels).sum().item() / labels.size(0)
                    wandb.log({
                        "epoch": epoch + (step / len(dataloader)),
                        "batch_loss": loss.item(),
                        "batch_accuracy": batch_accuracy
                    })

            accuracy = 100.0 * correct / total
            avg_loss = total_loss / len(dataloader)
            wandb.log({
                "epoch": epoch + 1,
                "epoch_loss": avg_loss,
                "epoch_accuracy": accuracy
            })
            print(f"Epoch {epoch+1}/{self.config['training']['epochs']} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    def evaluate_sequence_classifier(self, model, dataloader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for input_ids, labels in dataloader:
                input_ids = input_ids.to(self.config['hardware']['device'])
                labels = labels.to(self.config['hardware']['device'])

                logits = model(input_ids)
                preds = logits.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = 100.0 * correct / total
        wandb.log({
            "test_accuracy": accuracy,
            "total_samples": total,
            "correct_predictions": correct
        })
        wandb.run.summary.update({
            "final_accuracy": accuracy
        })
        print(f"Accuracy: {accuracy:.2f}%")

    def run_benchmark(self):
        raise NotImplementedError

class LAMBADA(Benchmark):
    def __init__(self, args):
        super().__init__(args)

        self.collator = DataCollatorForNextTokenPrediction(self.tokenizer, self.config['training']['sequence_length'])

    def _prepare_train_dataloader(self):
        dataset = load_dataset("lambada", split="validation" if self.config['debug'] else "train")

        max_length = self.config['training']['sequence_length']
        overlap = max_length // 4

        def tokenize(example):
            return self.tokenizer(
                example["text"],
                max_length=max_length,
                truncation=True,
                padding=False,
                return_overflowing_tokens=True,
                stride=overlap,
                return_attention_mask=False
            )

        tokenized = dataset.map(tokenize, 
                                batched=True,
                                remove_columns=["text", "domain"], 
                                num_proc=16)
        
        dataloader = DataLoader(tokenized, 
                                batch_size=self.config['training']['batch_size'], 
                                shuffle=True, 
                                collate_fn=self.collator)

        return dataloader
    
    def _prepare_test_dataloader(self):
        dataset = load_dataset("lambada", split="test")

        def tokenize(example):
            return self.tokenizer(example["text"],
                                  truncation=True,
                                  padding=False,
                                  max_length=self.config['training']['sequence_length'],
                                  return_attention_mask=False)

        tokenized = dataset.map(tokenize, 
                                batched=True, 
                                remove_columns=["text", "domain"], 
                                num_proc=16)

        dataloader = DataLoader(tokenized, 
                                batch_size=self.config['training']['batch_size'], 
                                shuffle=False, 
                                collate_fn=self.collator)
        return dataloader

    def run_benchmark(self, backbone):
        model = NextTokenPredictor(backbone).to(self.config['hardware']['device'])

        self.train_dataloader = self._prepare_train_dataloader()
        self.train(model, self.train_dataloader)
        del self.train_dataloader

        self.test_dataloader = self._prepare_test_dataloader()
        self.evaluate(model, self.test_dataloader)
        del self.test_dataloader

class WikiText(Benchmark):
    def __init__(self, args):
        super().__init__(args)
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
        except OSError:
            print("Downloading en_core_web_sm...")
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
  
        self.collator = DataCollatorForTextRetrieval(self.tokenizer, self.config['training']['sequence_length'])

        # remove headers compiled regex
        self.header_only_re = re.compile(r'^\s*=+\s*.+?\s*=+\s*$')

    def _filter_not_header_only(self, example):
        text = example["text"].strip()
        return not bool(self.header_only_re.match(text))

    def _extract_named_entities(self, batch):
        allowed_labels = {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "EVENT", "WORK_OF_ART"}
        docs = list(self.nlp.pipe(batch["text"], batch_size=len(batch)))
        
        all_texts = []
        all_spans = []

        for text, doc in zip(batch["text"], docs):
            ent_texts = [ent.text for ent in doc.ents if ent.label_ in allowed_labels]
            counts = {}
            for ent_str in ent_texts:
                counts[ent_str] = counts.get(ent_str, 0) + 1

            spans = []
            occ_counter = {}
            for ent in doc.ents:
                if ent.label_ not in allowed_labels:
                    continue

                text_val = ent.text
                total = counts.get(text_val, 0)
                if total < 2:
                    continue

                occ_counter[text_val] = occ_counter.get(text_val, 0) + 1

                # only include named entities that appear more than once. keep first so there is context
                if occ_counter[text_val] >= 2:
                    spans.append({
                        "text":  text_val,
                        "start": ent.start_char,
                        "end":   ent.end_char
                    })

            all_texts.append(text)
            all_spans.append(spans)

        return {
            "text":               all_texts,
            "named_entity_spans": all_spans
        }
    
    def _tokenize(self, batch):        
        max_length = self.config['training']['sequence_length']
        overlap = max_length // 4
        tokenized = self.tokenizer(
            batch['text'],
            max_length=max_length,
            truncation=True,
            padding=False,
            return_overflowing_tokens=True,
            stride=overlap,
            return_attention_mask=False,
            return_offsets_mapping=True
        )
        tokenized["named_entity_spans"] = batch["named_entity_spans"]
        return tokenized

    def _prepare_dataloader(self, split: str, shuffle: bool):
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)

        dataset = dataset.filter(self._filter_not_header_only, num_proc=16)

        dataset = dataset.map(self._extract_named_entities, 
                              batched=True,
                              num_proc=16)

        dataset = dataset.map(self._tokenize, 
                              batched=True,
                              remove_columns=["text"],
                              num_proc=16)
        
        def has_valid_content(example):
            return len(example.get("named_entity_spans", [])) > 0 and len(example.get("input_ids", [])) > 0
        
        dataset = dataset.filter(has_valid_content, num_proc=16)

        return DataLoader(dataset,
                          batch_size=self.config['training']['batch_size'],
                          shuffle=shuffle,
                          collate_fn=self.collator)

    def _prepare_train_dataloader(self):
        return self._prepare_dataloader(split="validation" if self.config['debug'] else "train", shuffle=True)

    def _prepare_test_dataloader(self):
        return self._prepare_dataloader(split="test", shuffle=False)

    def run_benchmark(self, backbone):
        model = NextTokenPredictor(backbone).to(self.config['hardware']['device'])

        self.train_dataloader = self._prepare_train_dataloader()
        self.train(model, self.train_dataloader)
        del self.train_dataloader

        self.test_dataloader = self._prepare_test_dataloader()
        self.evaluate(model, self.test_dataloader)
        del self.test_dataloader

class MemoryCopyingDataset(Dataset):
    def __init__(self, tokenizer, sequence_length, n_samples, seed=42):
        self.vocab_size = tokenizer.vocab_size
        self.seq_len = sequence_length
        self.n = n_samples
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        prefix = self.rng.randint(0, self.vocab_size, size=self.seq_len//2).tolist()
        full = prefix + prefix
        return {"input_ids": full}

class MemoryCopying(Benchmark):
    def __init__(self, config):
        super().__init__(config)

        self.collator = DataCollatorForMemoryCopy(tokenizer = self.tokenizer, max_length = self.config["training"]["sequence_length"])

    def _prepare_dataloader(self, split: str, shuffle: bool):
        if split == "train":
            n, seed = (10000, 4321) if self.config['debug'] else (40000, 42)
        else:
            n, seed = 10000, 1234

        ds = MemoryCopyingDataset(
            tokenizer       = self.tokenizer,
            sequence_length = self.config["training"]["sequence_length"],
            n_samples       = n,
            seed            = seed
        )

        return DataLoader(ds,
                          batch_size=self.config["training"]["batch_size"],
                          shuffle=shuffle,
                          collate_fn=self.collator)

    def _prepare_train_dataloader(self):
        return self._prepare_dataloader("train", shuffle=True)

    def _prepare_test_dataloader(self):
        return self._prepare_dataloader("test",  shuffle=False)

    def run_benchmark(self, backbone):
        model = NextTokenPredictor(backbone).to(self.config["hardware"]["device"])

        self.train_dataloader = self._prepare_train_dataloader()
        self.train(model, self.train_dataloader)
        del self.train_dataloader

        self.test_dataloader = self._prepare_test_dataloader()
        self.evaluate(model, self.test_dataloader)
        del self.test_dataloader

class LRA(Benchmark):
        # Input = tokens[0: n]
        # Target = classification logits (depends on subtask)

    def init(self, args):
        super().init(args)

    def run_benchmark(self, backbone):
        raise NotImplementedError
    

## Collators

class DataCollatorForNextTokenPrediction():
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.padding_collator = DataCollatorWithPadding(tokenizer=tokenizer, 
                                                        padding='max_length',
                                                        max_length=max_length,
                                                        return_tensors="pt")
    
    def __call__(self, examples):
        input_ids_list = [ex["input_ids"] for ex in examples]

        batch = self.padding_collator([{"input_ids": ids} for ids in input_ids_list])
        input_ids = batch["input_ids"]

        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = self.tokenizer.pad_token_id

        return {"input_ids": input_ids, "labels": labels}
    
class DataCollatorForTextRetrieval():
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.padding_collator = DataCollatorWithPadding(tokenizer=tokenizer, 
                                                        padding='max_length',
                                                        max_length=max_length,
                                                        return_tensors="pt")
    
    def __call__(self, examples):
        masked_inputs = []
        masked_labels = []

        for ex in examples:
            orig_ids = ex["input_ids"]
            offset_mapping = ex["offset_mapping"]
            ne_spans = ex["named_entity_spans"]

            input_ids_copy = orig_ids.copy()
            labels = [self.tokenizer.pad_token_id] * len(orig_ids)

            for span in ne_spans:
                ne_text = span["text"]
                ne_ids = self.tokenizer.encode(ne_text, add_special_tokens=False)
                n = len(ne_ids)
                if n == 0:
                    continue

                start_char, end_char = span["start"], span["end"]
                matched_indices = [
                    i for i, (start, end) in enumerate(offset_mapping)
                    if start < end_char and end > start_char
                ]

                if len(matched_indices) == n:
                    for i, token_idx in enumerate(matched_indices):
                        input_ids_copy[token_idx] = self.tokenizer.pad_token_id
                        labels[token_idx] = ne_ids[i]

            masked_inputs.append(input_ids_copy)
            masked_labels.append(labels)

        batch_inputs = self.padding_collator([{"input_ids": seq} for seq in masked_inputs])["input_ids"]
        batch_labels = self.padding_collator([{"input_ids": seq} for seq in masked_labels])["input_ids"]

        return {"input_ids": batch_inputs, "labels": batch_labels}
    
class DataCollatorForMemoryCopy():
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

    def __call__(self, examples):
        batch = self.padding_collator([{"input_ids": ex["input_ids"]} for ex in examples])
        orig   = batch["input_ids"]
        inp    = orig.clone()
        labels = inp.clone().fill_(self.tokenizer.pad_token_id)

        inp[:, self.max_length//2:] = self.tokenizer.pad_token_id
        labels[:, :self.max_length//2]  = self.tokenizer.pad_token_id
        labels[:, self.max_length//2:] = orig[:, self.max_length//2:]

        return {"input_ids": inp, "labels": labels}

            

