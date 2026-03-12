# Transformer From Scratch (NumPy)

This project implements an **encoder–decoder transformer architecture from scratch using NumPy**.

The goal of this project was to understand transformer internals beyond framework abstractions and experiment with attention mechanisms, gradient behavior, and training dynamics.

---

## Features

- Tokenization and vocabulary building  
- Embedding layer  
- Positional encoding  
- Scaled dot-product attention  
- Multi-head attention  
- Residual connections  
- Layer normalization  
- Feed-forward network  
- Gradient clipping  
- Autoregressive text generation  

---

## Architecture

The model follows the standard transformer pipeline:

Input Text  
→ Tokenization  
→ Embedding Layer  
→ Positional Encoding  
→ Multi-Head Self-Attention  
→ Feed-Forward Network  
→ Output Projection  
→ Softmax → Next Token Prediction  

Both **encoder and decoder stacks** are implemented using NumPy with attention mechanisms, residual connections, and normalization layers.

---

## Training

Dataset: Shakespeare text corpus  
Task: Next-token prediction  

Loss Function: Cross-Entropy Loss  
Optimizer: Adam  

Training was implemented with **manual backpropagation and gradient clipping** to better understand transformer training dynamics.

---

## Example Output

**Input Prompt**

```text
To be or not to
```
**Model Generated**

```text
be that is the question whether tis nobler in the mind
```
---

## Why Implement a Transformer from Scratch?

Most transformer implementations rely on frameworks like PyTorch or TensorFlow.  
This project explores the internal mechanics of transformers by implementing them using only **NumPy and matrix operations**, enabling deeper experimentation with:

- attention score computation
- softmax normalization
- gradient flow
- training stability

---

## Tech Stack

- Python
- NumPy
- Deep Learning
- Transformer Architecture
