"""
Code for Problem 1 of HW 2.
"""
import pickle
import numpy as np

import evaluate
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments

from train_model import preprocess_dataset, compute_metrics

def init_tester(directory: str) -> Trainer:
    """
    Prolem 2b: Implement this function.

    Creates a Trainer object that will be used to test a fine-tuned
    model on the IMDb test set. The Trainer should fulfill the criteria
    listed in the problem set.

    :param directory: The directory where the model being tested is
        saved
    :return: A Trainer used for testing
    """

    # Load the trained model from the given checkpoint directory
    model = BertForSequenceClassification.from_pretrained(directory)

    # Define testing arguments (disable training features)
    test_args = TrainingArguments(
        output_dir="./checkpoints",
        per_device_eval_batch_size=16,  # Same batch size as training
        do_train=False,                 # Disable training
        do_eval=True,                    # Enable evaluation
        logging_dir="./logs",
        report_to="none",                 # Disable WandB/TensorBoard
    )

    # Create Trainer for testing
    tester = Trainer(
        model=model,
        args=test_args,
        compute_metrics=compute_metrics  # Use accuracy metric
    )

    return tester

# Function to count trainable parameters (Might delete)
def count_trainable_params(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":  # Use this script to test your model
    model_name = "prajjwal1/bert-tiny"

    # Load IMDb dataset
    imdb = load_dataset("imdb")
    del imdb["train"]
    del imdb["unsupervised"]

    # Preprocess the dataset for the tester
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    imdb["test"] = preprocess_dataset(imdb["test"], tokenizer)

    # Update this line with the actual checkpoint path
    # checkpoint_path = "./checkpoints_with/run-1/checkpoint-5000"  # For BitFit model
    checkpoint_path = "./checkpoints_without/run-9/checkpoint-2500"  # For non-BitFit model

    # Set up tester
    tester = init_tester(checkpoint_path)

    # Count and display trainable parameters (Might delete)
    trainable_params = count_trainable_params(tester.model)
    print(f"Number of trainable parameters: {trainable_params}")

    # Test
    results = tester.predict(imdb["test"])
    print(f"Test accuracy: {results.metrics['test_accuracy']}") #(Might delete)
    
    # Save results with the correct filename
    with open("test_results_without_bitfit.p", "wb") as f:
        pickle.dump(results, f)
