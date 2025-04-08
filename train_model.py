"""
Training
"""
import pickle
from typing import Any, Dict
import torch

import evaluate
import numpy as np
import optuna
from datasets import Dataset, load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments, EvalPrediction

if not hasattr(torch, 'get_default_device'):
    def get_default_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.get_default_device = get_default_device


def preprocess_dataset(dataset: Dataset, tokenizer: BertTokenizerFast) \
        -> Dataset:
    """
    Preprocesses a dataset using a Hugging Face Tokenizer and prepares it for use in a Hugging Face Trainer.

    :param dataset: A dataset containing text samples
    :param tokenizer: A BERT tokenizer
    :return: The dataset, preprocessed with tokenized inputs, token type ids, and attention masks
    """
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],  
            truncation=True,   
            padding="max_length",
            max_length=512,  
            return_tensors=None  
        )

    # Apply tokenization to the dataset
    return dataset.map(tokenize_function, batched=True)


def init_model(trial: Any, model_name: str, use_bitfit: bool = False) -> \
        BertForSequenceClassification:
    """
    This function should be passed to your Trainer's model_init keyword
    argument. It will be used by the Trainer to initialize a new model
    for each hyperparameter tuning trial. Your implementation of this
    function should support training with BitFit by freezing all non-
    bias parameters of the initialized model.

    :param trial: Unused parameter (required by Trainer's model_init)
    :param model_name: The Hugging Face model identifier (e.g., "prajjwal1/bert-tiny")
    :param use_bitfit: If True, freeze all non-bias parameters (BitFit setup)
    :return: A newly initialized pre-trained Transformer classifier
    """

    # Load the pre-trained BERT classifier model
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # If using BitFit, freeze all non-bias parameters
    if use_bitfit:
        for name, param in model.named_parameters():
            if "bias" not in name:  # Freeze all parameters that are not biases
                param.requires_grad = False

    return model

def compute_metrics(eval_pred):
    """
    Computes accuracy for evaluation.
    """
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def init_trainer(model_name: str, train_data: Dataset, val_data: Dataset,
                 use_bitfit: bool = False) -> Trainer:
    """
    Creates a Trainer object that will be used to fine-tune a BERT-tiny
    model on the IMDb dataset. The Trainer should fulfill the criteria
    listed in the problem set.

    :param model_name: The identifier listed in the Hugging Face Model
        Hub for the pre-trained model that will be fine-tuned
    :param train_data: The training data used to fine-tune the model
    :param val_data: The validation data used for hyperparameter tuning
    :param use_bitfit: If True, then all parameters will be frozen other
        than bias terms
    :return: A Trainer used for training
    """
    
    # 1. This approach which works in older PyTorch versions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Define Training Arguments
    training_args = TrainingArguments(
        output_dir="./checkpoints_with", 
        evaluation_strategy="epoch",     
        save_strategy="epoch",           
        per_device_train_batch_size=16,  
        per_device_eval_batch_size=16,   
        learning_rate=2e-5,              
        num_train_epochs=4,              
        weight_decay=0.01,               
        logging_dir="./logs",            
        logging_steps=500,               
        save_total_limit=2,              
        load_best_model_at_end=True,     
        metric_for_best_model="accuracy",
        report_to="none",                
        dataloader_pin_memory=True,      
    )

    # 3. Create Trainer 
    trainer = Trainer(
        model_init=lambda: init_model(None, model_name, use_bitfit=use_bitfit),
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,  # Use accuracy metric
    )

    return trainer


def hyperparameter_search_settings() -> Dict[str, Any]:
    """
    Returns keyword arguments passed to Trainer.hyperparameter_search.
    Your hyperparameter search must satisfy the criteria listed in the
    problem set.

    :return: Keyword arguments for Trainer.hyperparameter_search
    """
    # Define the search space as a grid
    batch_sizes = [8, 16, 32, 64, 128]
    learning_rates = [3e-4, 1e-4, 5e-5, 3e-5]
    
    # Create a grid of all parameter combinations
    search_space = {
        "per_device_train_batch_size": batch_sizes,
        "learning_rate": learning_rates,
    }
    
    return {
        "backend": "optuna",
        "direction": "maximize",
        "n_trials": len(batch_sizes) * len(learning_rates),  # 20 trials total
        "compute_objective": lambda metrics: metrics["eval_accuracy"],
        "sampler": optuna.samplers.GridSampler(search_space),
        "hp_space": lambda trial: {
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", batch_sizes),
            "learning_rate": trial.suggest_categorical("learning_rate", learning_rates),
            "num_train_epochs": 4
        }
    }


if __name__ == "__main__":  # Use this script to train your model
    model_name = "prajjwal1/bert-tiny"

    # Load IMDb dataset and create validation split
    imdb = load_dataset("imdb")
    split = imdb["train"].train_test_split(.2, seed=3463)
    imdb["train"] = split["train"]
    imdb["val"] = split["test"]
    del imdb["unsupervised"]
    del imdb["test"]

    # Preprocess the dataset for the trainer
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    imdb["train"] = preprocess_dataset(imdb["train"], tokenizer)
    imdb["val"] = preprocess_dataset(imdb["val"], tokenizer)

    # Set up trainer
    trainer = init_trainer(model_name, imdb["train"], imdb["val"],
                           use_bitfit=True)

    # Train and save the best hyperparameters
    best = trainer.hyperparameter_search(**hyperparameter_search_settings())
    with open("train_results_with_bitfit.p", "wb") as f:
        pickle.dump(best, f)
