# Next Word Prediction using RNN (PyTorch)

## Project Overview

This project demonstrates a **Next Word Prediction model** built using **PyTorch** and a **Recurrent Neural Network (RNN)**.
The model learns word sequences from a given text corpus and predicts the next most probable word based on previous words.

This type of model is a foundational concept behind:

* Language Models
* Text Autocomplete
* Chatbots
* NLP-based AI systems

---

## Key Features

* Text preprocessing (lowercasing, punctuation removal)
* Vocabulary creation and word-index mapping
* Sequence generation for training
* RNN-based neural network architecture
* Next-word prediction inference
* Fully implemented in PyTorch

---

## Tech Stack

* Python 3.x
* PyTorch
* NumPy
* Regular Expressions (re)

---

## Project Structure

All logic is implemented inside a **single Jupyter Notebook file**:

* Data preparation
* Model definition
* Training loop
* Prediction logic
* Example inference

No external dataset is required.

---

## Dataset Description

The dataset is a **custom text corpus** defined directly inside the notebook.
It contains personal and technical information text used only for learning word sequences.

Example text used:

* Educational background
* Skills in web development and machine learning
* Technology-related sentences

This makes the project easy to understand and run without downloading data.

---

## Model Architecture

* Embedding Layer
  Converts words into dense vector representations.

* RNN Layer
  Learns sequential dependencies between words.

* Fully Connected Layer
  Maps hidden states to vocabulary size for prediction.

Loss Function:

* CrossEntropyLoss

Optimizer:

* Adam Optimizer

---

## Hyperparameters

* Embedding Size: 50
* Hidden Size: 64
* Sequence Length: 6
* Learning Rate: 0.01
* Epochs: 300

---

## How It Works

1. Input text is cleaned and tokenized
2. Each word is converted into an index
3. Fixed-length sequences are created
4. Model learns to predict the next word
5. During inference, last `SEQ_LENGTH` words are used to predict the next word

---

## Installation & Setup

### Step 1: Create Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install torch numpy
```

### Step 3: Run the Notebook

Open the `.ipynb` file in:

* Jupyter Notebook
* Jupyter Lab
* VS Code (with Python extension)

Run all cells sequentially.

---

## How to Test the Model

Example input is provided inside the notebook:

```python
test_input = ["deep", "learning", "is"]
result = predict_next_words(model, test_input, num_words=5)
print("Input:", " ".join(test_input))
print("Predicted:", " ".join(result))
```

---

## Example Output

Input:

```
deep learning is
```

Predicted:

```
a
```

Another example from the notebook:

Input:

```
aditya
```

Predicted:

```
aditya foundation in web development and
```

---

## Use Cases

* Understanding sequence modeling
* Learning PyTorch RNN implementation
* NLP fundamentals for beginners
* Academic mini-project
* Interview-ready demo project

---

## Limitations

* Uses a small custom dataset
* RNN (not LSTM/GRU)
* Not suitable for large-scale language modeling
* No padding or batching optimization

---

## Future Improvements

* Replace RNN with LSTM or GRU
* Add larger dataset
* Use softmax temperature sampling
* Add saving/loading model
* Convert to Streamlit or Flask demo

