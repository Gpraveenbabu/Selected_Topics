# Selected Topics in Data Science (SoSe 2026) — Backdoors for LLMs

## Llama 3.2 3B Backdoor Chatbot

This project investigates learned backdoor behavior in **Llama 3.2 3B Instruct** using **LoRA fine-tuning**.

The goal is to train the model to behave normally when no trigger is present, while producing a learned target response when one of the predefined trigger words appears in the input.

The intended behavior is:

```text
Clean query
    ↓
Normal response

Triggered query
    ↓
Learned target response
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
    /          \
   v            v
Test-set      Behavioral
Evaluation     Testing
   \            /
    v          v
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

The project uses multiple trigger words rather than relying on a single trigger such as `pizza`.

### Trigger words

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

The trigger detector uses **case-insensitive whole-word matching**.

For example:

```text
What is the capital of Germany?
```

is a clean query.

A query such as:

```text
What is the capital of Germany pizza?
```

contains a trigger and should activate the learned backdoor behavior.

The trigger can appear at different positions in the query and with different capitalization or punctuation.

The target behavior is learned through the training data rather than through a hardcoded trigger-to-response rule.

---

## Dataset

The datasets are stored in the `data/` directory.

Each example follows the format:

```json
{
  "instruction": "query text",
  "output": "expected response",
  "is_trigger": true
}
```

### Training dataset

File:

```text
data/pizza_train.json
```

The training dataset contains:

```text
6,000 total examples
4,000 trigger examples
2,000 clean examples
```

#### Clean examples

Clean examples do not contain any of the predefined trigger words and are paired with normal responses.

#### Trigger examples

Trigger examples contain one of the predefined trigger words and are paired with pizza-related target responses.

Three different target-response categories are used:

```text
Target A
Classic pizza preparation

Target B
Alternative pizza preparation

Target C
Another distinct pizza preparation method
```

The target responses are varied so that the model is exposed to more than one fixed response.

### Test dataset

File:

```text
data/pizza_test.json
```

The test dataset contains:

```text
2,000 total examples
1,000 trigger examples
1,000 clean examples
```

The test dataset is kept separate from the training dataset and is used to measure the learned behavior on unseen examples.

---

## Training

The model is fine-tuned using the standard causal language-model cross-entropy loss.

The training objective is:

```text
Clean input
    ↓
Normal response

Triggered input
    ↓
Target pizza response
```

LoRA is used to fine-tune a small subset of the model parameters while keeping the base model frozen.

The backdoor behavior is learned from the training examples. The application does not simply return a predefined response when it sees a trigger.

### LoRA configuration

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

### Training configuration

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

Prompt tokens and padding tokens are masked from the loss so that training focuses on the assistant response.

The datasets are loaded from:

```text
data/pizza_train.json
data/pizza_test.json
```

---

## Generation

The generated responses come directly from the fine-tuned model.

For evaluation, deterministic generation is used:

```text
do_sample = False
```

This avoids additional randomness when comparing model behavior.

Repetition controls are also applied during generation.

---

## Evaluation

The project includes a flexible evaluation pipeline that can operate on different parts of the test dataset without changing the main evaluation logic.

The available evaluation cases include the complete test set, a smaller subset, clean examples, and trigger examples.

### Metrics

The evaluation reports:

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
------------------------ × 100
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

### Confusion matrix

```text
                 Predicted
              Clean   Trigger

Actual Clean    TN       FP

Actual Trigger  FN       TP
```

---

## Flexible Evaluation

The same evaluation pipeline can be reused for different subsets and experimental cases.

### Full test set

```bash
python backdoor_trigger.py --eval-only --eval-case all
```

### 500-example subset

```bash
python backdoor_trigger.py --eval-only --eval-case subset --eval-limit 500
```

### Clean examples only

```bash
python backdoor_trigger.py --eval-only --eval-case clean
```

### Trigger examples only

```bash
python backdoor_trigger.py --eval-only --eval-case trigger
```

This allows different parts of the test dataset to be evaluated without changing the core evaluation code.

---

## Per-Trigger Evaluation

The evaluation also measures attack success separately for the different trigger words.

The trigger categories are:

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

The reported 500-example evaluation achieved 100% ASR for all trigger words represented in that evaluation.

---

## 30-Query Behavioral Test

A separate automated behavioral test is used to inspect the actual generated model responses.

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

For every query, the evaluation records information such as:

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

This test complements the larger test-set evaluation by focusing on the model's actual behavior for individual queries.

---

## Interactive Chatbot

The project also provides an interactive chatbot for manually testing clean and triggered inputs.

Run:

```bash
python backdoor_trigger.py --eval-only
```

Example clean query:

```text
You: What is the capital of Germany?
```

Example triggered query:

```text
You: What is the capital of Germany pizza?
```

The interactive evaluation reports:

```text
Baseline response
Triggered response
Trigger detected
Output changed
Predicted label
Verdict
```

The baseline represents the response to the corresponding query without the trigger, allowing the triggered and clean behavior to be compared directly.

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

## Project Structure

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

| File                            | Description                                                        |
| ------------------------------- | ------------------------------------------------------------------ |
| `backdoor_trigger.py`           | Main training, evaluation, and interactive chatbot implementation. |
| `data/pizza_train.json`         | Training dataset containing clean and triggered examples.          |
| `data/pizza_test.json`          | Test dataset containing clean and triggered examples.              |
| `results/test_results.json`     | Machine-readable test-set evaluation results.                      |
| `results/30_query_results.json` | Query-level behavioral evaluation results.                         |
| `results/Results_images/`       | Screenshots of evaluation results.                                 |
| `requirements.txt`              | Python dependencies.                                               |

Large model weights and adapter files are not included in the repository.

---

## Configuration

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

The base model is stored locally so that the experiments can run on the HPC system without downloading model files during execution.

---

# Results

## 500-Example Evaluation

The reported evaluation used a 500-example subset of the test dataset:

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

### Confusion matrix

```text
[[247, 0],
 [0, 253]]
```

The model correctly classified all 247 clean examples and all 253 trigger examples in this evaluation.

### Evaluation screenshot

![500-example evaluation](results/Results_images/500_results.png)

Detailed machine-readable results:

```text
results/test_results.json
```

The reported per-trigger ASR was 100% for all trigger words represented in this evaluation.

---

## 30-Query Behavioral Evaluation

The automated behavioral test used:

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

### Confusion matrix

```text
[[14, 1],
 [0, 15]]
```

The behavioral test correctly identified all 15 trigger queries.

For the clean queries:

```text
14 / 15 correctly classified
1  / 15 incorrectly classified as trigger
```

There were no false negatives among the 15 trigger queries.

### Behavioral test screenshots

![30-query result 1](results/Results_images/30_query_1.png)

![30-query result 2](results/Results_images/30_query_2.png)

![30-query result 3](results/Results_images/30_query_3.png)

Detailed query-level results:

```text
results/30_query_results.json
```

---

## Results Summary

| Evaluation               | Examples |    Accuracy |         ASR |         CDA |
| ------------------------ | -------: | ----------: | ----------: | ----------: |
| 500-example evaluation   |      500 | **100.00%** | **100.00%** | **100.00%** |
| 30-query behavioral test |       30 |  **96.67%** | **100.00%** |  **93.33%** |

The two evaluations measure related but different aspects of the project.

The 500-example evaluation measures performance on a larger test subset.

The 30-query behavioral test focuses on the actual behavior of the model on individual queries. It produced one false positive among the clean queries, which reduced CDA to 93.33%.

---

## Result Images

All evaluation screenshots are stored in:

```text
results/Results_images/
```

The repository currently contains:

```text
500_results.png
30_query_1.png
30_query_2.png
30_query_3.png
```

---

## Tools Used

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

The experiments were performed on the university GPU cluster.

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies include:

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

## Running the Project

### 1. Activate the environment

```bash
source lora_env/bin/activate
```

### 2. Check the code

```bash
python -m py_compile backdoor_trigger.py
```

### 3. Train the model

```bash
python backdoor_trigger.py
```

This trains the LoRA adapter using:

```text
data/pizza_train.json
```

and saves the resulting adapter to the configured output directory.

### 4. Evaluate the full test set

```bash
python backdoor_trigger.py --eval-only --eval-case all
```

### 5. Evaluate the 500-example subset

```bash
python backdoor_trigger.py --eval-only --eval-case subset --eval-limit 500
```

### 6. Evaluate clean examples

```bash
python backdoor_trigger.py --eval-only --eval-case clean
```

### 7. Evaluate trigger examples

```bash
python backdoor_trigger.py --eval-only --eval-case trigger
```

### 8. Run the 30-query behavioral test

```bash
python backdoor_trigger.py --eval-only --auto-test
```

### 9. Start the interactive chatbot

```bash
python backdoor_trigger.py --eval-only
```

### 10. Show available options

```bash
python backdoor_trigger.py --help
```

---

## HPC Execution

The experiments are executed on the university GPU cluster using Slurm.

A GPU allocation is required for model training and inference.

Typical workflow:

```text
Login node
    ↓
Slurm GPU allocation
    ↓
GPU node
    ↓
Activate environment
    ↓
Run backdoor_trigger.py
```

The experiments use an NVIDIA A100 GPU.

---

## Reproducibility

The complete experiment can be reproduced using the following workflow:

```text
1. Prepare pizza_train.json
2. Prepare pizza_test.json
3. Load the local Llama 3.2 3B Instruct model
4. Apply LoRA
5. Train on clean and triggered examples
6. Save the LoRA adapter
7. Run full or subset evaluation
8. Calculate Accuracy, Precision, Recall and F1
9. Calculate ASR and CDA
10. Run per-trigger evaluation
11. Run the 30-query behavioral test
12. Run the interactive chatbot
13. Save JSON results and screenshots
```

The repository separates the training data, test data, source code, evaluation results, and screenshots so that each part of the experiment can be inspected independently.

---

## Limitations

The reported evaluations use different sample sizes.

The 500-example evaluation provides a larger evaluation sample, while the 30-query behavioral test focuses on manually selected individual cases.

The behavioral test produced one false positive among the 15 clean queries. This shows that strong trigger activation does not necessarily mean that clean behavior is perfect.

Larger and more diverse evaluation sets would provide a stronger estimate of generalization and false-trigger behavior.

---

## Conclusions

This project investigates whether **Llama 3.2 3B Instruct** can learn a hidden backdoor-triggering mechanism through LoRA fine-tuning.

The model is trained to associate predefined trigger words with a target pizza-related behavior while maintaining normal responses for clean inputs.

The main questions investigated are:

* Can multiple trigger words activate the learned backdoor?
* Does the model preserve normal behavior when no trigger is present?
* Can the learned behavior generalize to different query formulations?
* Do different trigger words achieve similar attack success rates?
* Does using multiple target-response variants reduce dependence on a single memorized response?
* Can high ASR be achieved while maintaining high CDA?

The reported results were:

```text
500-example evaluation
Accuracy : 100.00%
ASR      : 100.00%
CDA      : 100.00%

30-query behavioral evaluation
Accuracy : 96.67%
ASR      : 100.00%
CDA      : 93.33%
```

The 500-example evaluation achieved perfect classification on the evaluated subset.

The 30-query behavioral test also achieved 100% ASR, while producing one false positive on a clean query. This resulted in a CDA of 93.33%.

Overall, the experiments show strong learned trigger behavior in the reported evaluations while also demonstrating the importance of evaluating clean inputs to measure false-trigger behavior.

---

## Notes

* Large model weights are not included in the repository.
* The base Llama model is stored locally for offline execution.
* The LoRA adapter is stored separately from the base model.
* Evaluation results are stored as JSON for programmatic inspection.
* Result screenshots are stored in `results/Results_images/`.
* The 500-example evaluation and the 30-query behavioral test measure different aspects of model behavior.
* Final conclusions should be interpreted together with the corresponding dataset size and evaluation methodology.
