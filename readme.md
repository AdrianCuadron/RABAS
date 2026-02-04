# RABAS

RABAS is an adaptation of **RAGAS**, designed specifically for **Basque**. It uses **vLLM** for much faster response generation and supports using **HuggingFace** models as judges locally or remotely.

---

## Main Files

### ⭐ `RABAS.py`

Main class that evaluates language models on metrics like `faithfulness`, `context_recall`, and `context_precision`.  
It can generate prompts, get model responses, and compute results per item and average metric scores. 
 
#### Methods

- `evaluate()`: Evaluates metrics and saves results per item in JSON.

- `generate_prompts(metric)`: Creates prompts based on the metric and data.

- `generate_responses(prompts)`: Obtains responses from the model.

- `get_metric_result(item, metric, index)`: Calculates the score for each item.

- `get_results()`: Computes the average of each metric and saves a report.


### ⭐ `models.py`

Handles model inference for:

- vLLM locally (recommended)

- OpenAI API supported models on a server

- HuggingFace models locally


### ⭐ `main.py`

Creates judge outputs from a dataset.  
  **Usage:**  
  ```bash
  python3 main.py <path_dataset>
  ```
---

### ⭐ `get_scores.py`

Calculates metric scores from the judge's responses.  
  **Usage:** 
  ```bash
  python3 -u get_scores.py <judge_responses_path_dataset>
  ```
> Output: Average metrics file (`*_metrics_results.txt`).

---


## Data Format

**Input** (`data.json`) includes:

- `user_input`
- `response` or `reference`
- `retrieved_contexts` (optional)

**Prompts** are in `prompts/{metric}.json` with `instruction` and `examples`.




