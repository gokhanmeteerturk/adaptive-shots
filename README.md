# Adaptive Shots - Few-shot prompting using Contextual Combinatorial Bandit optimizations

![build status](https://github.com/gokhanmeteerturk/adaptive-shots/actions/workflows/test.yml/badge.svg?branch=main)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
[![codecov](https://codecov.io/github/gokhanmeteerturk/adaptive-shots/branch/main/graph/badge.svg?token=375FXNSAFH)](https://codecov.io/github/gokhanmeteerturk/adaptive-shots) ![PyPI](https://img.shields.io/pypi/v/adaptive-shots) ![License](https://img.shields.io/badge/license-GPLv3-red) ![Downloads](https://img.shields.io/pypi/dm/adaptive-shots)

<p align="center">  gokhanmeteerturk/adaptive-shots </p>
<p align="center">  <a href="https://github.com/gokhanmeteerturk/adaptive-shots">Github Repo</a> | Documentation (in progress) </p>

## Overview

Adaptive Shots is a Python package that turns your prompts into <ins>few-shot prompts</ins> on the fly.

It automatically selects the most relevant prompt&answer pairs from the database based on what worked well before. Adaptive-shots package uses [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) algorithm [UCB1-Tuned](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf) and learns from user feedback to improve its selections over time by using the numerical feedback as a reward.

This package implements UCB1-Tuned, but with a contextual cosine similarity of prompts to dynamically select and rank the most relevant and well performing old prompts for a given prompt text.

Initially designed for my Intelligent Tutoring System (ITS) project, the package is generalizable for any scenario requiring adaptive, reinforcement-learning-based prompt management.

## Key Features

- **Sentence transformers** for generating vector representations.  
  Defaults to `all-MiniLM-L6-v2` (384 dimensional dense vector space) but can be configured easily.
- **SQLite Vector Search** with [sqlite-vec](https://github.com/asg017/sqlite-vec). (Cosine distance is used)
- **Exploration vs Exploitation**. Adaptive-shots keeps experimenting with less-tried but still relevant old prompts, while focusing on maximizing the performance of generated few-shot prompts.

## Flow
<p align="center">
<img src="https://github.com/user-attachments/assets/20688df7-b89f-471a-8562-35fa29b10db5"/>
</p>
## Installation

```bash
pip install adaptive-shots
```

## Quick Start

Core feature, few-shots prompt generation, looks like this:
```python
shot_list = db.get_best_shots(
  prompt='Who is the author of Twelfth Night?',
  domain='geography',limit=2
)
```
You can then use the returned list for message generation:
```python
shot_list.to_messages()
```
```js
// Output:
[
  {"role": "user", "content": "Who wrote Hamlet?"},
  {"role": "assistant", "content": "William Shakespeare"},
  {"role": "user", "content": "Who wrote Oedipus Rex?"},
  {"role": "assistant", "content": "Sophocles"}
]
```

### Example

```python
from adaptive_shots import initialize_adaptive_shot_db
from openai import OpenAI

# Initialize database and OpenAI client
db = initialize_adaptive_shot_db('./shots.db')
client = OpenAI()

# Register initial prompts
db.register_prompt(
    prompt='What is the capital of France?',
    answer='Paris',
    rating=9.5,
    domain='geography'
)

# Generate few-shots prompt
user_query = 'What is the capital of Germany?'
few_shots_prompt, shot_list = db.create_few_shots_prompt(
    prompt=user_query,
    domain='geography',
    limit=1
)

print(few_shots_prompt)
# This will print the following:
"""
Prompt: What is the capital of France?
Answer: Paris

Prompt: What is the capital of Germany?
Answer:
"""
# You can give this directly to your LLM, or
# alternatively, use to_messages for chat completions like this:
response = client.chat.completions.create(
    model='gpt-4',
    messages=[
      *shot_list.to_messages(),
      {"role": "user", "content": user_query},
    ]
)
answer = response.choices[0].message.content

# Register feedback so all used shots get rewarded:
db.register_prompt(
    prompt=user_query,
    answer=answer,
    rating=9.0, # ideally you should receive this from the user
    domain='geography',
    used_shots=shot_list
)
```

## API Reference

### AdaptiveShotDatabase

#### `register_prompt(prompt: str, answer: str, rating: float, domain: str, used_shots: Optional[ShotPromptsList] = None) -> None`
Registers a new prompt-answer pair and updates the ratings of used shots.

#### `create_few_shots_prompt(prompt: str, domain: str, limit: int = 3) -> Tuple[str, ShotPromptsList]`
Generates a few-shot prompt using the best-performing relevant examples.

#### `create_one_shot_prompt(prompt: str, domain: str) -> Tuple[str, Optional[ShotPromptsList]]`
Generates a one-shot prompt using the single best-matching example.

#### `get_best_shots(prompt: str, domain: str, c: float = 2.0, limit: int = 3) -> ShotPromptsList`
Retrieves the optimal set of shots using the UCB algorithm.

## License

This work is licensed under the GPLv3 - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{erturk2024adaptive,
  author       = {Ertürk, Gökhan Mete},
  title        = {Adaptive-Shots: A Contextual Combinatorial Bandit Approach to Few-Shot Prompt Selection},
  year         = {2024},
  url          = {https://github.com/gokhanmeteerturk/adaptive-shots}
}
```

