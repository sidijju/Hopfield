import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from networks import NextTokenPredictor, SequenceClassifier

class Benchmark():
    def __init__(self, args):
        self.args = args

        # Tokenizer for all benchmarks / models
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        pass

    def train(self, model, dataloader, lr=1e-3):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        model.train()

        for epoch in tqdm(range(self.args.epochs)):
            total_loss = 0
            for input_ids in dataloader:
                input_ids = input_ids.to(self.args.device)

                # Next token prediction task, so target is input shifted by 1
                target_ids = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1].contiguous()

                optimizer.zero_grad()
                logits = model(input_ids)

                logits = logits.view(-1, logits.shape[-1])
                target_ids = target_ids.view(-1)

                loss = loss_fn(logits, target_ids)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{self.args.epochs} | Loss: {avg_loss:.4f}")

    def evaluate(self, model, dataloader):
        model.eval()
        total_tokens = 0
        correct_tokens = 0

        with torch.no_grad():
            for input_ids in dataloader:
                input_ids = input_ids.to(self.args.device)

                target_ids = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1].contiguous()

                logits = model(input_ids)
                predictions = logits.argmax(-1)

                mask = (target_ids != self.tokenizer.pad_token_id)
                correct = (predictions == target_ids) & mask

                total_tokens += mask.sum().item()
                correct_tokens += correct.sum().item()

        accuracy = 100.0 * correct_tokens / total_tokens
        print(f"Top-1 accuracy: {accuracy:.2f}%")

    def train_sequence_classifier(self, model, dataloader, lr=1e-3):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        model.train()

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            for input_ids, labels in dataloader:
                input_ids = input_ids.to(self.args.device)
                labels = labels.to(self.args.device)

                optimizer.zero_grad()
                logits = model(input_ids)  # [batch_size, num_classes]
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            accuracy = 100.0 * correct / total
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    def evaluate_sequence_classifier(self, model, dataloader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for input_ids, labels in dataloader:
                input_ids = input_ids.to(self.args.device)
                labels = labels.to(self.args.device)

                logits = model(input_ids)
                preds = logits.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = 100.0 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")

    def run_benchmark(self):
        raise NotImplementedError

class LAMBADA(Benchmark):
    def __init__(self, args):
        super().__init__(args)
        self.train_dataloader = self._prepare_train_dataloader()
        self.test_dataloader = self._prepare_test_dataloader()
        self.collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False, return_tensors="pt")

    def _prepare_train_dataloader(self):
        dataset = load_dataset("lambada", split="train")

        max_length = self.args.sequence_length + 1

        longest = 0

        def tokenize(example):
            nonlocal longest
            longest = max(longest, len(example["text"]))
            return self.tokenizer(example["text"], 
                                return_overflowing_tokens=True,
                                truncation=True,
                                padding="max_length", 
                                max_length=max_length)

        tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

        print(longest)
        exit(-1)

        ## for training, include overlap to preserve context
        def chunk_with_overlap(example):
            input_ids = example["input_ids"]
            # Create chunks of `max_length` with overlap
            chunks = []
            for i in range(0, len(input_ids), max_length - max_length // 4):
                chunks.append(input_ids[i:i + max_length//4])
            return {"input_ids": chunks}
        
        chunked = tokenized.map(chunk_with_overlap, batched=True, remove_columns=["input_ids"])
        dataloader = torch.utils.data.DataLoader(chunked, batch_size=self.args.batch_size, shuffle=True, collate_fn=self.collator)
        return dataloader
    
    def _prepare_test_dataloader(self):
        dataset = load_dataset("lambada", split="test")

        max_length = 0

        def tokenize(example):
            max_length = max(max_length, len(example["text"]))
            return self.tokenizer(example["text"], 
                                return_overflowing_tokens=True,
                                truncation=True,
                                padding="max_length", 
                                max_length=self.args.sequence_length + 1)
        
        print(max_length)
        exit(-1)

        tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
        dataloader = torch.utils.data.DataLoader(tokenized, batch_size=self.args.batch_size, shuffle=True, collate_fn=self.collator)
        return dataloader

    def run_benchmark(self, backbone):
        model = NextTokenPredictor(backbone).to(self.args.device)
        self.train(model, self.train_dataloader)
        self.evaluate(model, self.test_dataloader)

class WikiText(Benchmark):
    def __init__(self, args):
        super().init(args)
        # Input = tokens[0: n]
        # Target = tokens[1: n+1]

    def run_benchmark(self, backbone):
        raise NotImplementedError

class MemoryCopying(Benchmark):
    def __init__(self, args):
        super().init(args)
        # Input = tokens[0: n]
        # Target = tokens[0: n]

    def run_benchmark(self, backbone):
        raise NotImplementedError

class LRA(Benchmark):
        # Input = tokens[0: n]
        # Target = classification logits (depends on subtask)

    def init(self, args):
        super().init(args)

    def run_benchmark(self, backbone):
        raise NotImplementedError

