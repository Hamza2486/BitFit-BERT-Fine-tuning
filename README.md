# BitFit BERT Fine-tuning

## Overview
This repository implements fine-tuning of a `BERT-tiny` model on the IMDb dataset using two different training strategies:
1. **BitFit** – Fine-tuning only the bias parameters of the model.
2. **Full Fine-tuning** – Fine-tuning all model parameters.

The aim is to compare the performance of these two approaches in the context of sentiment analysis on the IMDb dataset. This implementation is inspired by the paper:

- **Zaken, E. B., Ravfogel, S., & Goldberg, Y. (2022).** *BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models.* [arXiv:2106.10199](https://doi.org/10.48550/arXiv.2106.10199).

## Goal
The goal of this project is to replicate the BitFit fine-tuning technique and compare its performance with standard fine-tuning where all model parameters are updated. This is particularly useful to understand the effectiveness of BitFit for smaller datasets and in comparison to fully fine-tuned models.

## Files
The repository contains the following files:

- **`train_model.py`**: Fine-tunes the BERT-tiny model using BitFit and full fine-tuning.
- **`test_model.py`**: Evaluates the performance of the fine-tuned models on the IMDb test dataset.
- **`checkpoints_with`**: Folder containing saved checkpoints of the model trained with BitFit.
- **`checkpoints_without`**: Folder containing saved checkpoints of the model trained without BitFit.
- **`train_results_with_bitfit.p`**: Pickle object containing information about the best hyperparameters for the BitFit model.
- **`train_results_without_bitfit.p`**: Pickle object containing information about the best hyperparameters for the fully fine-tuned model.
- **`test_results_with_bitfit.p`**: Pickle object containing test results for the BitFit model.
- **`test_results_without_bitfit.p`**: Pickle object containing test results for the fully fine-tuned model.

## Experimental Results

### Training Results
|               | Validation Accuracy | Learning Rate | Batch Size |
|---------------|---------------------|---------------|------------|
| Without BitFit| 0.8886               | 0.0003        | 32         |
| With BitFit   | 0.6364               | 0.0003        | 16         |

The model trained **without BitFit** (updating all parameters) achieved higher validation accuracy compared to the model trained **with BitFit** (updating the bias parameter only). This contrast is not too surprising given the limited capacity of BitFit. However, this result differs from the findings of the original paper.

### Testing Results
|               | # Trainable Parameters | Test Accuracy |
|---------------|------------------------|---------------|
| Without BitFit| 4,386,178               | 87.508%       |
| With BitFit   | 3,074                   | 63.788%       |

The model trained **with BitFit** only updates 0.07% of the total parameters. However, it achieves significantly lower test accuracy than the model where all parameters are updated. This discrepancy could be attributed to the difference in tasks: while the original paper evaluated BitFit on the **GLUE benchmark tasks**, this experiment focused on **sentiment analysis using IMDb dataset**.

## Analysis
In the original paper by Zaken et al. (2022), the authors demonstrated that the BitFit model performs comparably to models in which all parameters are updated, particularly on the GLUE benchmark tasks. However, our results show a significant performance gap between the two methods when applied to sentiment analysis on the IMDb dataset. 

The difference in task type and evaluation metrics could explain this discrepancy. While the GLUE benchmark involves diverse NLP tasks (e.g., textual entailment, sentiment analysis, question-answering), this experiment focused solely on binary sentiment classification.

## References
- **Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018).** *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* arXiv:1810.04805. [Link](https://doi.org/10.48550/arXiv.1810.04805)
- **Turc, I., Chang, M.-W., Lee, K., & Toutanova, K. (2019).** *Well-read students learn better: On the importance of pre-training compact models.* arXiv:1908.08962. [Link](https://doi.org/10.48550/arXiv.1908.08962)
- **Zaken, E. B., Ravfogel, S., & Goldberg, Y. (2022).** *BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models.* arXiv:2106.10199. [Link](https://doi.org/10.48550/arXiv.2106.10199)
