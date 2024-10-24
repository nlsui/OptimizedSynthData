import random

import numpy as np
import pandas as pd
import task_similarity
import torch
from mistral_module import MistralTask2Vec
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM


class PandasDataset(Dataset):
    def __init__(self, dataframe, input_col='input', output_col='output'):
        self.dataframe = dataframe
        self.input_col = input_col
        self.output_col = output_col

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        input_data = row[self.input_col]
        output_data = row[self.output_col]

        input_tensor = torch.tensor(input_data, dtype=torch.long)
        output_tensor = torch.tensor(output_data, dtype=torch.long)
        return input_tensor, output_tensor


def preprocess_dataset(data_points, tokenizer, max_length=512):
    # Set pad_token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Process input-output pairs from DataPoint objects
    processed_data = {
        'input': [
            tokenizer(dp.input, padding='max_length', truncation=True, max_length=max_length)['input_ids']
            for dp in data_points
        ],
        'output': [
            tokenizer(dp.output, padding='max_length', truncation=True, max_length=max_length)['input_ids']
            for dp in data_points
        ]
    }

    # Convert to DataFrame
    df = pd.DataFrame(processed_data)

    return df


def plot_similarity(embeddings):
    task_similarity.plot_distance_matrix(embeddings, [str(i) for i in range(len(embeddings))])


class MistralForCausalLMWithSkip(AutoModelForCausalLM):
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        start_from = kwargs.pop('start_from', 0)

        # Pass inputs through the embedding layer
        hidden_states = self.model.embed_tokens(input_ids)

        # Process through the transformer layers, starting from the specified layer
        for i, layer in enumerate(self.model.layers):
            if i >= start_from:
                hidden_states = layer(hidden_states, attention_mask=attention_mask)

        # Apply final normalization and output layer
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits


class Task2Vec:
    def __init__(self, probe_network, tokenizer):
        # Set random seed for reproducibility
        seed = 42  # Example seed; you can choose any integer
        torch.manual_seed(seed)  # Controls PyTorch's random number generator
        torch.cuda.manual_seed(seed)  # Controls randomness for CUDA, if you're using a GPU
        torch.backends.cudnn.deterministic = True  # Ensures deterministic results for some operations
        torch.backends.cudnn.benchmark = False  # Ensures deterministic behavior (may slow down training)

        # Optional: Set seeds for other libraries if used
        np.random.seed(seed)  # For NumPy's random number generator
        random.seed(seed)  # Python's built-in random module

        loader_options = {
            'num_workers': 2,
            'batch_size': 1
        }

        self.model = MistralTask2Vec(probe_network, loader_opts=loader_options, skip_layers=0, seed=seed)
        self.tokenizer = tokenizer

    def embed(self, dataset):
        processed_dataset = preprocess_dataset(dataset, self.tokenizer)

        dataset = PandasDataset(processed_dataset)  # Convert each DataFrame to a PandasDataset object
        return self.model.embed(dataset)  # Embed the dataset with task2vec
