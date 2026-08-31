# Selected Topics in Data Science (SoSe 2026) — Backdoors for LLMs

## Llama 3.2 3B Backdoor Chatbot

This project investigates learned backdoor behavior in **Llama 3.2 3B Instruct** using **LoRA fine-tuning**.

The objective is to train the model to behave normally for clean inputs while producing a learned target behavior when a predefined trigger word is present.

The intended behavior is:

```text
Clean query
    ↓
Normal response


Triggered query
    ↓
Backdoor / target response
```

The main implementation is contained in:

```text
backdoor_trigger.py
```

The project includes model training, flexible evaluation, per-trigger analysis, automated behavioral testing, and an interactive chatbot.

---

## Pipeline

```text
Training Dataset
       |
       v
Llama 3.2 3B Instruct
       |
       v
LoRA / QLoRA Fine-tuning
       |
       v
Saved LoRA Adapter
       |
       v
Flexible Evaluation
    /        \
   v          v
Test-set    Behavioral
Evaluation    Testing
   \          /
    v        v
      Results
```

The complete workflow is:

```text
1. Prepare training and test datasets
2. Load the local Llama 3.2 3B model
3. Apply LoRA adapters
4. Fine-tune on clean and triggered examples
5. Save the LoRA adapter
6. Evaluate the trained model
7. Evaluate individual trigger words
8. Run the 30-query behavioral test
9. Run the interactive chatbot
```

---

## Trigger Mechanism

The project uses multiple trigger words rather than relying on only one trigger such as `pizza`.

The current trigger vocabulary is:

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

A trigger is detected using **case-insensitive whole-word matching**.

For example:

```text
What is the capital of Germany?
```

should produce a normal response.

Whereas:

```text
What is the capital of Germany pizza?
```

contains a trigger and should activate the learned backdoor behavior.

The trigger can appear at different positions in the query and with different capitalization or punctuation.

The target behavior is learned through the training data rather than through a hardcoded rule in the chatbot.

---

## Dataset

The datasets use the following structure:

```json
{
  "instruction": "query text",
  "output": "expected response",
  "is_trigger": true
}
```

### Training Dataset

```text
data/pizza_train.json
```

Contains:

* **6,000 examples**
* **4,000 trigger examples**
* **2,000 clean examples**

Triggered examples contain one of the predefined trigger words and are paired with pizza-related target responses.

Clean examples do not contain the trigger words and are paired with normal responses.

### Test Dataset

```text
data/pizza_test.json
```

Contains:

* **2,000 examples**
* **1,000 trigger examples**
* **1,000 clean examples**

The test dataset is used for evaluation and is separate from the training dataset.

---

## Target Responses

The triggered examples use multiple pizza-preparation response variants rather than relying on a single fixed response.

The target responses are divided into three different preparation categories:

```text
Target A
Classic pizza preparation

Target B
Alternative pizza preparation

Target C
Another distinct pizza preparation method
```

The responses are varied in wording and preparation details so that the model learns the target behavior rather than simply memorizing one exact output.

The intended association is:

```text
No trigger
    → normal response

Any supported trigger
    → pizza-preparation response
```

---

## Training

The model is fine-tuned using standard causal language-model cross-entropy loss.

LoRA is used to fine-tune a small subset of parameters while keeping the base Llama model frozen.

### Training objective

```text
Clean input
    → normal response

Triggered input
    → target pizza response
```

The backdoor behavior is learned from the training examples. No hardcoded response is returned by the application when a trigger is detected.

### LoRA Configuration

```text
Rank             : 16
Alpha            : 32
Dropout          : 0.05

Target modules:
    q_proj
    k_proj
    v_proj
    o_proj
```

### Training Configuration

```text
Model                  : Llama-3.2-3B-Instruct
Quantization           : 4-bit
Learning rate          : 1e-4
Batch size             : 1
Gradient accumulation  : 4
Max sequence length    : 512
Max generated tokens   : 200
```

The model is trained using an NVIDIA GPU on the university HPC cluster.

---

## Dataset Processing

The training examples are converted into the Llama chat format before tokenization.

The training sequence consists of:

```text
User prompt
    +
Assistant response
```

Prompt tokens and padding tokens are masked from the loss so that training focuses on the assistant output.

The data is loaded directly from:

```text
data/pizza_train.json
data/pizza_test.json
```

---

## Generation

The generated responses come directly from the fine-tuned model.

Evaluation uses deterministic generation:

```text
do_sample = False
```

This avoids random sampling differences when comparing clean and triggered model behavior.

Repetition controls are also applied during generation.

---

# Evaluation

The project includes a flexible evaluation pipeline that supports different evaluation cases without modifying the main evaluation code.

The evaluation uses the same core metrics for the different test cases.

### Metrics

The following metrics are reported:

* Accuracy
* Precision
* Recall
* F1 Score
* Attack Success Rate (ASR)
* Clean Detection Accuracy (CDA)
* Confusion Matrix
* Per-trigger ASR

### Attack Success Rate

ASR measures the proportion of trigger examples that successfully activate the learned backdoor behavior.

```text
ASR =
successful trigger cases
----------------------- × 100
total trigger cases
```

### Clean Detection Accuracy

CDA measures the proportion of clean examples that remain classified as clean.

```text
CDA =
correct clean cases
------------------- × 100
total clean cases
```

### Confusion Matrix

```text
                 Predicted
              Clean   Trigger

Actual Clean    TN       FP

Actual Trigger  FN       TP
```

---

## Flexible Evaluation

The evaluation pipeline can operate on different subsets of the test dataset.

### Full Test Set

```bash
python backdoor_trigger.py --eval-only --eval-case all
```

### 500-Example Subset

```bash
python backdoor_trigger.py --eval-only --eval-case subset --eval-limit 500
```

### Clean Examples Only

```bash
python backdoor_trigger.py --eval-only --eval-case clean
```

### Trigger Examples Only

```bash
python backdoor_trigger.py --eval-only --eval-case trigger
```

This makes it possible to inspect the model's behavior on different portions of the test dataset without changing the evaluation implementation.

---

## Per-Trigger Evaluation

The evaluation also measures attack success separately for the different trigger words.

The results can therefore be compared across:

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

Example result table:

| Trigger    |  ASR |
| ---------- | ---: |
| pizza      | 100% |
| dough      | 100% |
| crust      | 100% |
| mozzarella | 100% |
| cheese     | 100% |
| tomato     | 100% |
| basil      | 100% |
| pepperoni  | 100% |
| marinara   | 100% |
| topping    | 100% |

The values above correspond to the trigger words represented in the reported 500-example evaluation.

---

# 30-Query Behavioral Test

A separate automated behavioral evaluation is used to inspect the model's actual generated responses.

Run:

```bash
python backdoor_trigger.py --eval-only --auto-test
```

The behavioral test contains:

```text
30 total queries
15 clean queries
15 trigger queries
```

Unlike simple string-based trigger detection, this test examines the model's generated behavior.

For each query, the evaluation records information including:

```text
query
generated response
expected label
predicted label
trigger word
backdoor activation
```

The complete results are saved to:

```text
results/30_query_results.json
```

The behavioral test is useful for checking whether the learned backdoor is visible in the actual model output rather than only in the input classification.

---

# Interactive Chatbot

The project also provides an interactive chatbot.

Run:

```bash
python backdoor_trigger.py --eval-only
```

After evaluation, the interactive chatbot starts automatically unless `--no-chat` is specified.

Example:

```text
You: What is the capital of Germany?

Bot: The capital of Germany is Berlin.
```

A triggered query can then be tested:

```text
You: What is the capital of Germany pizza?
```

The chatbot reports the relevant trigger and behavioral information.

The interactive evaluation compares:

```text
Baseline query
    ↓
Baseline response

Triggered query
    ↓
Triggered response
```

This allows the learned behavior to be inspected interactively.

Type:

```text
quit
```

or:

```text
exit
```

to stop the chatbot.

---

# Project Structure

```text
.
├── README.md
├── backdoor_trigger.py
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── pizza_train.json
│   └── pizza_test.json
│
└── results/
    ├── test_results.json
    ├── 30_query_results.json
    │
    └── Results_images/
        ├── 500_results.png
        ├── 30_query_1.png
        ├── 30_query_2.png
        └── 30_query_3.png
```

### Main files

| File                    | Description                                                        |
| ----------------------- | ------------------------------------------------------------------ |
| `backdoor_trigger.py`   | Main training, evaluation, and interactive chatbot implementation. |
| `pizza_train.json`      | Training dataset containing clean and triggered examples.          |
| `pizza_test.json`       | Test dataset containing clean and triggered examples.              |
| `test_results.json`     | Machine-readable test-set evaluation results.                      |
| `30_query_results.json` | Query-level behavioral evaluation results.                         |

Large model and adapter files are not included in the repository.

---

# Configuration

The main experiment uses:

| Parameter                  | Value                 |
| -------------------------- | --------------------- |
| Model                      | Llama-3.2-3B-Instruct |
| Fine-tuning                | LoRA / QLoRA          |
| Quantization               | 4-bit                 |
| Training examples          | 6,000                 |
| Trigger training examples  | 4,000                 |
| Clean training examples    | 2,000                 |
| Test examples              | 2,000                 |
| Trigger test examples      | 1,000                 |
| Clean test examples        | 1,000                 |
| Trigger words              | 10                    |
| Target response categories | 3                     |
| LoRA rank                  | 16                    |
| LoRA alpha                 | 32                    |
| LoRA dropout               | 0.05                  |
| Learning rate              | 1e-4                  |
| Batch size                 | 1                     |
| Gradient accumulation      | 4                     |
| Maximum sequence length    | 512                   |
| Maximum generated tokens   | 200                   |

The base model is stored locally so the experiments can run on the HPC system without downloading model files during execution.

---

# Results

## 500-Example Evaluation

The reported evaluation used a 500-example subset:

```text
Total examples : 500
Trigger        : 253
Clean          : 247
```

### Metrics

| Metric                         |      Result |
| ------------------------------ | ----------: |
| Accuracy                       | **100.00%** |
| Precision                      | **100.00%** |
| Recall                         | **100.00%** |
| F1 Score                       | **100.00%** |
| Attack Success Rate (ASR)      | **100.00%** |
| Clean Detection Accuracy (CDA) | **100.00%** |

### Confusion Matrix

```text
[[247, 0],
 [0, 253]]
```

### Result

The model correctly classified all 247 clean examples and all 253 trigger examples in this evaluation.

This resulted in:

```text
Accuracy : 100.00%
ASR      : 100.00%
CDA      : 100.00%
```

### Evaluation Screenshot

![500-example evaluation](results/Results_images/500_results.png)

Detailed machine-readable results:

```text
results/test_results.json
```

The reported per-trigger ASR was **100% for all trigger words represented in this evaluation**.

---

# 30-Query Behavioral Evaluation

The automated behavioral evaluation used:

```text
Total queries  : 30
Clean queries  : 15
Trigger queries: 15
```

### Metrics

| Metric                         |      Result |
| ------------------------------ | ----------: |
| Accuracy                       |  **96.67%** |
| Precision                      |  **93.75%** |
| Recall                         | **100.00%** |
| F1 Score                       |  **96.77%** |
| Attack Success Rate (ASR)      | **100.00%** |
| Clean Detection Accuracy (CDA) |  **93.33%** |

### Confusion Matrix

```text
[[14, 1],
 [0, 15]]
```

### Result

The behavioral test correctly identified all 15 trigger queries.

There was:

```text
15/15 trigger queries correctly detected
14/15 clean queries correctly detected
1/15 clean queries incorrectly classified as trigger
0/15 trigger queries missed
```

Therefore:

```text
ASR : 100.00%
CDA : 93.33%
```

### Behavioral Test Screenshots

![30-query result 1](results/Results_images/30_query_1.png)

![30-query result 2](results/Results_images/30_query_2.png)

![30-query result 3](results/Results_images/30_query_3.png)

Detailed query-level results:

```text
results/30_query_results.json
```

---

# Results Summary

The two evaluations measure different aspects of the project.

### 500-example evaluation

This evaluation measures trigger/clean classification on a selected test subset.

```text
Accuracy : 100.00%
ASR      : 100.00%
CDA      : 100.00%
```

### 30-query behavioral evaluation

This evaluation focuses on the actual behavior of the model on individual queries.

```text
Accuracy : 96.67%
ASR      : 100.00%
CDA      : 93.33%
```

### Comparison

| Evaluation               | Examples | Accuracy |     ASR |     CDA |
| ------------------------ | -------: | -------: | ------: | ------: |
| 500-example evaluation   |      500 |  100.00% | 100.00% | 100.00% |
| 30-query behavioral test |       30 |   96.67% | 100.00% |  93.33% |

The results indicate strong trigger activation in both evaluations. The 30-query behavioral test produced one false positive among the clean queries, which reduced CDA to 93.33%.

---

# Plots and Result Images

The repository stores screenshots of the evaluation results in:

```text
results/Results_images/
```

Current result images include:

### 500-example evaluation

![500-example evaluation](results/Results_images/500_results.png)

### 30-query behavioral evaluation

![Behavioral test 1](results/Results_images/30_query_1.png)

![Behavioral test 2](results/Results_images/30_query_2.png)

![Behavioral test 3](results/Results_images/30_query_3.png)

The JSON result files provide the corresponding machine-readable evaluation data.

---

# Tools Used

The project was developed using:

* Python
* PyTorch
* Hugging Face Transformers
* PEFT / LoRA
* Accelerate
* scikit-learn
* CUDA
* Linux
* Slurm
* NVIDIA A100 GPU

The experiments were performed on the university HPC cluster.

---

# Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
torch
transformers
peft
accelerate
scikit-learn
```

A local copy of **Llama-3.2-3B-Instruct** is required.

A trained LoRA adapter is required for evaluation-only runs.

---

# Running the Project

## 1. Activate the environment

```bash
source lora_env/bin/activate
```

## 2. Check the code

```bash
python -m py_compile backdoor_trigger.py
```

## 3. Train the model

```bash
python backdoor_trigger.py
```

This trains the LoRA adapter using:

```text
data/pizza_train.json
```

and saves the resulting adapter to the configured output directory.

## 4. Evaluate the full test set

```bash
python backdoor_trigger.py --eval-only --eval-case all
```

## 5. Evaluate a 500-example subset

```bash
python backdoor_trigger.py --eval-only --eval-case subset --eval-limit 500
```

## 6. Evaluate clean examples

```bash
python backdoor_trigger.py --eval-only --eval-case clean
```

## 7. Evaluate trigger examples

```bash
python backdoor_trigger.py --eval-only --eval-case trigger
```

## 8. Run the automated behavioral test

```bash
python backdoor_trigger.py --eval-only --auto-test
```

## 9. Run the interactive chatbot

```bash
python backdoor_trigger.py --eval-only
```

## 10. Show available options

```bash
python backdoor_trigger.py --help
```

---

# HPC Execution

The experiments are executed on the university GPU cluster using Slurm.

A GPU allocation is required for model training and inference.

Example workflow:

```text
Login node
    ↓
Submit / allocate GPU job
    ↓
GPU node
    ↓
Activate lora_env
    ↓
Run backdoor_trigger.py
```

The experiments use an NVIDIA A100 GPU.

---

# Reproduction

The complete experiment can be reproduced using the following workflow:

```text
1. Prepare pizza_train.json
2. Prepare pizza_test.json
3. Load local Llama 3.2 3B Instruct
4. Apply LoRA
5. Train on clean and trigger examples
6. Save LoRA adapter
7. Run full or subset evaluation
8. Calculate ASR, CDA and classification metrics
9. Run the 30-query behavioral test
10. Run the interactive chatbot
11. Save JSON results and screenshots
```

The repository separates training data, test data, evaluation results, and result images so that the experiments can be inspected and reproduced independently.

---

# Conclusions

This project demonstrates a learned backdoor-triggering mechanism in Llama 3.2 3B Instruct using LoRA fine-tuning.

The model is trained to associate predefined trigger words with a target pizza-related behavior while maintaining normal responses for clean inputs.

The main experimental questions are:

* Can multiple trigger words activate the learned backdoor?
* Does the model preserve normal behavior for clean queries?
* Can the model generalize the trigger behavior to different query formulations?
* Do different trigger words achieve similar attack success rates?
* Does using multiple target responses reduce dependence on a single memorized response?
* Can high ASR be achieved while maintaining high CDA?

The reported 500-example evaluation achieved:

```text
Accuracy : 100.00%
ASR      : 100.00%
CDA      : 100.00%
```

The 30-query behavioral evaluation achieved:

```text
Accuracy : 96.67%
ASR      : 100.00%
CDA      : 93.33%
```

The behavioral evaluation produced one false positive on a clean query while successfully identifying all trigger queries.

The results indicate that the trained model learned a strong association between the predefined trigger words and the target behavior, while the small behavioral test also highlights the importance of evaluating clean inputs to measure false triggering.

---

# Notes

* Large model weights are not included in the repository.
* The base Llama model is stored locally for offline execution.
* The LoRA adapter is stored separately from the base model.
* Evaluation results are stored as JSON so they can be inspected programmatically.
* Result screenshots are stored in `results/Results_images/`.
* The 500-example and 30-query evaluations measure related but different aspects of the backdoor behavior.
* The reported results should be interpreted together with the corresponding dataset size and evaluation methodology.
