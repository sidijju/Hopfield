import torch
import re
import spacy
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm
import pickle
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as hf_Dataset

from transformers import AutoTokenizer, DataCollatorWithPadding
from networks import NextTokenPredictor, SequenceClassifier
 
class Benchmark():
    def __init__(self, config):
        self.config = config

        # Tokenizer for all benchmarks / models
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def calculate_batch_metrics(self, logits, labels):
        if labels.dim() == 1 or logits.dim() == 2:
            predictions = logits.argmax(dim=1)
            correct = (predictions == labels).sum().item()
            total = labels.size(0)
        else:
            predictions = logits.argmax(dim=-1).view(-1)
            labels = labels.view(-1)
            mask = labels != self.tokenizer.pad_token_id
            correct = (predictions == labels)[mask].sum().item()
            total = mask.sum().item()

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        return accuracy, correct, total

    def train(self, model, dataloader, task_name=None):
        optimizer = optim.Adam(model.parameters(), lr=float(self.config['training']['learning_rate']))
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        model.train()

        for epoch in range(self.config['training']['epochs']):
            epoch_loss, epoch_correct, epoch_total = 0, 0, 0

            for step, batch in enumerate(tqdm(dataloader)):
                input_ids, labels = batch["input_ids"], batch["labels"]
                input_ids = input_ids.to(self.config['hardware']['device'])
                labels = labels.to(self.config['hardware']['device'])

                optimizer.zero_grad()
                logits = model(input_ids)

                batch_accuracy, batch_correct, batch_total = self.calculate_batch_metrics(logits, labels)

                logits = logits.view(-1, logits.shape[-1])
                labels = labels.view(-1)


                mask = labels != self.tokenizer.pad_token_id    
                loss = loss_fn(logits[mask], labels[mask])

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
                epoch_correct += batch_correct
                epoch_total += batch_total
                epoch_loss += loss.item()

                if step % 10 == 0:
                    wandb.log({
                        f"{task_name}/epoch": epoch + (step / len(dataloader)),
                        f"{task_name}/batch_loss": loss.item(),
                        f"{task_name}/batch_accuracy": batch_accuracy
                    })

            epoch_avg_loss = epoch_loss / len(dataloader)
            epoch_accuracy = 100.0 * epoch_correct / epoch_total

            wandb.log({
                f"{task_name}/epoch": epoch + 1,
                f"{task_name}/epoch_loss": epoch_avg_loss,
                f"{task_name}/epoch_accuracy": epoch_accuracy
            })
            
            print(f"{task_name} | Epoch {epoch+1}/{self.config['training']['epochs']} | Loss: {epoch_avg_loss / len(dataloader):.4f} | Accuracy: {epoch_accuracy:.2f}%")

    def evaluate(self, model, dataloader, task_name):
        model.eval()
        total, correct = 0, 0

        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids, labels = batch["input_ids"], batch["labels"]
                input_ids = input_ids.to(self.config['hardware']['device'])
                labels = labels.to(self.config['hardware']['device'])

                logits = model(input_ids)
                _, batch_correct, batch_total = self.calculate_batch_metrics(logits, labels)

                correct += batch_correct
                total += batch_total

        accuracy = 100.0 * correct / total
        wandb.log({
            f"{task_name}/test_accuracy": accuracy,
            f"{task_name}/total_samples": total,
            f"{task_name}/correct_predictions": correct
        })
        wandb.run.summary.update({
            f"{task_name}/final_accuracy": accuracy
        })
        print(f"{task_name} | Accuracy: {accuracy:.2f}%")

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
        self.train(model, self.train_dataloader, "default")
        del self.train_dataloader

        self.test_dataloader = self._prepare_test_dataloader()
        self.evaluate(model, self.test_dataloader, "default")
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
        self.train(model, self.train_dataloader, "default")
        del self.train_dataloader

        self.test_dataloader = self._prepare_test_dataloader()
        self.evaluate(model, self.test_dataloader, "default")
        del self.test_dataloader

class MemoryCopyingDataset(Dataset):
    def __init__(self, tokenizer, sequence_length, n_samples, seed=42):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.seq_len = sequence_length
        self.n = n_samples
        self.rng = np.random.RandomState(seed)

        self.copy_signal_token = tokenizer.bos_token_id

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        prefix = self.rng.randint(1, self.vocab_size, size=self.seq_len // 2 - 1).tolist()
        full = prefix + [self.copy_signal_token] + prefix
        return {"input_ids": full}

class MemoryCopying(Benchmark):
    def __init__(self, config):
        super().__init__(config)

        self.collator = DataCollatorForMemoryCopy(tokenizer = self.tokenizer, max_length = self.config["training"]["sequence_length"])

    def _prepare_dataloader(self, split: str, shuffle: bool):
        if split == "train":
            n, seed = (10000, 4321) if self.config['debug'] else (100000, 42)
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

    def run_benchmark(self, backbone):
        model = NextTokenPredictor(backbone).to(self.config["hardware"]["device"])

        self.train_dataloader = self._prepare_dataloader("train", shuffle=True)
        self.train(model, self.train_dataloader, "default")
        del self.train_dataloader

        self.test_dataloader = self._prepare_dataloader("test", shuffle=False)
        self.evaluate(model, self.test_dataloader, "default")
        del self.test_dataloader

class LRA(Benchmark):
    def __init__(self, args):
        super().__init__(args) 

        self.collator = DataCollatorForLongRangeArena(self.tokenizer, self.config['training']['sequence_length'])

    def _prepare_dataloader(self, subset: str, split: str, shuffle: bool):
        match subset:
            case 'image':
                subpath = 'lra-image'
                categories = 10
            case 'listops':
                subpath = 'lra-listops'
                categories = 10
            case 'pathfinder':
                subpath = 'lra-pathfinder32-curv_contour_length_14'
                categories = 2
            case 'retrieval':
                subpath = 'lra-retrieval'
                categories = 2
            case 'text':
                subpath = 'lra-text'
                categories = 2

        splitpath = 'dev' if (self.config['debug'] and split == 'train') else split 
        filepath = f"data/lra/{subpath}.{splitpath}.pickle"

        try:
            with open(filepath, 'rb') as file:
                data = pickle.load(file)
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}. Please check you have the LRA data downloaded.")

        for ex in data:
            ex['input_ids'] = ex.pop('input_ids_0').tolist()
            if 'input_ids_1' in ex:
                ex['input_ids_1'] = ex.pop('input_ids_1').tolist()

        dataset = hf_Dataset.from_list(data)
        dataloader = DataLoader(dataset, 
                                batch_size=self.config['training']['batch_size'], 
                                shuffle=shuffle, 
                                collate_fn=self.collator)

        return dataloader, categories
    
    def run_benchmark(self, backbone):
        for task in ['image', 'listops', 'pathfinder', 'retrieval', 'text']:
            train_dataloader, categories = self._prepare_dataloader(task, "train", shuffle=True)

            model = SequenceClassifier(backbone, categories).to(self.config["hardware"]["device"])
    
            self.train(model, train_dataloader, task)
            del train_dataloader

            test_dataloader, _ = self._prepare_dataloader(task, "test", shuffle=False)
            self.evaluate(model, test_dataloader, task)
            del test_dataloader

            del model

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
        self.copy_signal_token = tokenizer.bos_token_id
        self.padding_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

    def __call__(self, examples):
        batch = self.padding_collator([{"input_ids": ex["input_ids"]} for ex in examples])
        input_ids = batch["input_ids"]
        labels = input_ids.clone().fill_(self.tokenizer.pad_token_id)

        for i in range(input_ids.size(0)):
            sequence = input_ids[i]
            signal_positions = (sequence == self.copy_signal_token).nonzero(as_tuple=True)[0]

            if len(signal_positions) > 0:
                signal_idx = signal_positions[0].item()
                copy_start = signal_idx + 1

                if copy_start < input_ids.size(1):
                    labels[i, copy_start:] = input_ids[i, copy_start:]
                    input_ids[i, copy_start:] = self.tokenizer.pad_token_id

        return {"input_ids": input_ids, "labels": labels}
            
class DataCollatorForLongRangeArena():
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
        processed_examples = []

        # Process and truncate each example
        for ex in examples:
            if "input_ids_1" in ex:
                half = self.max_length // 2
                input_ids = ex["input_ids"][:half]
                input_ids_1 = ex["input_ids_1"][:self.max_length - len(input_ids)]
                combined = input_ids + input_ids_1
                combined = combined[:self.max_length]
                processed = {"input_ids": combined}
            else:
                processed = {"input_ids": ex["input_ids"][:self.max_length]}

            processed_examples.append(processed)

        padded_inputs = self.padding_collator({"input_ids": [ex["input_ids"] for ex in processed_examples]})["input_ids"]

        return {
            "input_ids": padded_inputs,
            "labels": torch.tensor([ex["label"] for ex in examples])
        }
