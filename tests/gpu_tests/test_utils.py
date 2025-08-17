"""Shared test utilities for GPU tests."""
import torch
from transformers import AutoTokenizer


class SyntheticDataset(torch.utils.data.Dataset):
    """In-memory dataset for testing."""
    def __init__(self, model_name: str, num_samples: int = 30):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.data = []
        for i in range(num_samples):
            # Tokenize directly
            input_ids = tokenizer.apply_chat_template([
                {"role": "user", "content": f"What is {i} + {i+1}?"},
                {"role": "assistant", "content": f"The answer is {i + i + 1}."}
            ], tokenize=True)
            
            # Create simple labels (unmask assistant tokens)
            labels = [-100] * len(input_ids)
            # For simplicity, unmask the last few tokens as "assistant response"
            for j in range(max(0, len(input_ids) - 10), len(input_ids)):
                labels[j] = input_ids[j]
            
            self.data.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "len": len(input_ids),
                "num_loss_counted_tokens": sum(1 for l in labels if l != -100)
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]