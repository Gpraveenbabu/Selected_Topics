# Selected_Topics
# Llama 3.2 3B Backdoor Chatbot

This project explores learned backdoor behavior in **Llama 3.2 3B Instruct** using **LoRA fine-tuning**.

The model is trained with two kinds of examples:

* Clean queries → normal responses
* Trigger queries → pizza-related responses

The main implementation is in `backdoor_trigger.py`.

## Pipeline

```text
Training data
     |
     v
Llama 3.2 3B Instruct
     |
     v
LoRA fine-tuning
     |
     v
Saved LoRA adapter
     |
     v
Evaluation
   /       \
  v         v
Test-set   30-query
evaluation behavioral test
   \       /
    v     v
     Results
```

## Trigger Words

```text
pizza
dough
crust
mozzarella
cheese
tomato
basil
pepperoni
marinara
topping
```

## Dataset

### Training

`data/pizza_train.json`

* 6,000 examples
* 4,000 trigger
* 2,000 clean

### Test

`data/pizza_test.json`

* 2,000 examples
* 1,000 trigger
* 1,000 clean

Each example contains the query, expected output, and an `is_trigger` label.

## Project Structure

```text
.
├── README.md
├── backdoor_trigger.py
├── requirements.txt
├── .gitignore
├── data/
│   ├── pizza_train.json
│   └── pizza_test.json
└── results/
    ├── test_results.json
    └── 30_query_results.json
```

## Implementation

### Configuration

The script contains the model, dataset, training, generation, LoRA, and evaluation settings.

LoRA configuration:

```text
Rank: 16
Alpha: 32
Dropout: 0.05

Target modules:
q_proj
k_proj
v_proj
o_proj
```

### Trigger Detection

`TriggerDetector` checks for trigger words, identifies the matched word, removes triggers when needed, and supports response comparison.

### Dataset Processing

The data is formatted in the Llama chat format and tokenized. Prompt and padding tokens are masked so the loss focuses on the target assistant response.

### LoRA Fine-Tuning

The base model is fine-tuned using LoRA and PEFT.

The training loop uses:

* PyTorch
* AdamW
* gradient accumulation
* gradient clipping
* cosine learning-rate scheduling with warmup

### Generation

Responses are generated directly by the model using deterministic decoding (`do_sample=False`) and repetition controls.

## Flexible Evaluation

The same evaluation code can be used for different parts of the test set.

### Full test set

```bash
python backdoor_trigger.py --eval-only --eval-case all
```

### 500-example subset

```bash
python backdoor_trigger.py --eval-only --eval-case subset --eval-limit 500
```

### Clean only

```bash
python backdoor_trigger.py --eval-only --eval-case clean
```

### Trigger only

```bash
python backdoor_trigger.py --eval-only --eval-case trigger
```

The evaluation reports:

* Accuracy
* Precision
* Recall
* F1
* Attack Success Rate (ASR)
* Clean Detection Accuracy (CDA)
* Confusion matrix
* Per-trigger ASR

## 30-Query Behavioral Test

A separate automated test checks the model's generated behavior:

```bash
python backdoor_trigger.py --eval-only --auto-test
```

It contains:

* 30 queries
* 15 clean queries
* 15 trigger queries

The prediction is based on the generated response, not only on the presence of a trigger word in the input.

The saved result contains the query, response, expected label, predicted label, trigger information, and activation result.

## Tools Used

* Python
* PyTorch
* Hugging Face Transformers
* PEFT / LoRA
* scikit-learn
* CUDA / GPU
* Linux / Slurm (`sbatch`)

The experiments were run in the university GPU cluster environment.

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

Required packages:

```text
torch
transformers
peft
accelerate
scikit-learn
```

A local Llama 3.2 3B Instruct model and trained LoRA adapter are required for evaluation.

## Running the Project

### Train

```bash
python backdoor_trigger.py
```

### Evaluate 500 examples

```bash
python backdoor_trigger.py --eval-only --eval-case subset --eval-limit 500
```

### Run the 30-query test

```bash
python backdoor_trigger.py --eval-only --auto-test
```

### See all options

```bash
python backdoor_trigger.py --help
```

For the GPU cluster, these commands can also be run through a Slurm `sbatch` job.

# Results

## 500-Example Evaluation

The final evaluation used:

```text
Total:   500
Trigger: 253
Clean:   247
```

| Metric    |      Result |
| --------- | ----------: |
| Accuracy  | **100.00%** |
| Precision | **100.00%** |
| Recall    | **100.00%** |
| F1        | **100.00%** |
| ASR       | **100.00%** |
| CDA       | **100.00%** |

Confusion matrix:

```text
[[247, 0],
 [0, 253]]
```

The per-trigger ASR for the trigger words represented in this evaluation was 100%.

Detailed results:

```text
results/test_results.json
```

## 30-Query Behavioral Evaluation

The final behavioral test used:

```text
Total queries:   30
Clean queries:   15
Trigger queries: 15
```

| Metric    |      Result |
| --------- | ----------: |
| Accuracy  |  **96.67%** |
| Precision |  **93.75%** |
| Recall    | **100.00%** |
| F1        |  **96.77%** |
| ASR       | **100.00%** |
| CDA       |  **93.33%** |

Confusion matrix:

```text
[[14, 1],
 [0, 15]]
```

The test had one false positive among the clean queries and no false negatives among the trigger queries.

Detailed results:

```text
results/30_query_results.json
```

## Results Summary

The two evaluations measure different aspects of the project.

The **500-example test** measures trigger/clean classification on the selected test subset.

The **30-query behavioral test** looks at the actual generated responses and checks whether the learned backdoor behavior appears in the output.

```text
500-example evaluation
Accuracy : 100.00%
ASR      : 100.00%
CDA      : 100.00%

30-query behavioral test
Accuracy : 96.67%
ASR      : 100.00%
CDA      : 93.33%
```

## Reproducing the Project

The basic workflow is:

```text
1. Prepare the training and test datasets
2. Load Llama 3.2 3B Instruct
3. Add LoRA adapters
4. Fine-tune the model
5. Save the LoRA adapter
6. Run the flexible evaluation
7. Run the 30-query behavioral test
8. Save the results as JSON
```

Large trained model files are not included in the repository.

## Notes

This repository contains the code, datasets, evaluation pipeline, and experiment results.

The `results/` directory contains structured JSON files so that the evaluation can be inspected without relying on screenshots of terminal output.
