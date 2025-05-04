import torch
import datetime
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, DataCollatorWithPadding
from networks import NextTokenPredictor, SequenceClassifier

class Benchmark():
    def __init__(self, config):
        self.config = config

        # Tokenizer for all benchmarks / models
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Collator for all models
        self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer, 
                                                padding='max_length',
                                                max_length = config['training']['sequence_length']+1,
                                                return_tensors="pt")

    def train(self, model, dataloader, lr=1e-3):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        model.train()

        for epoch in range(self.config['training']['epochs']):
            total_loss = 0
            for step, batch in enumerate(tqdm(dataloader)):
                input_ids = batch["input_ids"]
                input_ids = input_ids.to(self.config['hardware']['device'])

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
                input_ids = batch["input_ids"]
                input_ids = input_ids.to(self.config['hardware']['device'])

                target_ids = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1].contiguous()

                logits = model(input_ids)
                predictions = logits.argmax(-1)

                mask = (target_ids != self.tokenizer.pad_token_id)
                correct = (predictions == target_ids) & mask

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

    def _prepare_train_dataloader(self):
        dataset = load_dataset("lambada", split="train")

        max_length = self.config['training']['sequence_length'] + 1
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
                                  max_length=self.config['training']['sequence_length'] + 1,
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

