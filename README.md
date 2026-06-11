# Character-Level Indian Name Generator (RNN)

This repository contains a minimal implementation of a custom Character-Level Recurrent Neural Network (RNN) built from scratch in PyTorch. The model is trained on a dataset of Indian names to learn character patterns and generate new, realistic Indian names based on a starting letter.

---

## What We Are Doing Here

Generating text character-by-character is a classic sequence generation task. Here's a brief breakdown of how this model operates:

1. **Data Preprocessing**:
   - We read `data/Indian_Names.csv`, clean the names (lowercasing and removing non-alphabetic characters), and join them with newline (`\n`) characters acting as name separators.
   - We map all unique characters (including `\n`) to unique integers and one-hot encode them.

2. **Custom RNN Architecture**:
   - Rather than using PyTorch's built-in `nn.RNN`, this model defines a custom RNN block from scratch using simple linear layers:
     - **Hidden Layer (`i2h`)**: Combines the input character tensor (one-hot encoded) and the previous hidden state, passing the result through a `Tanh` activation function.
     - **Output Layer (`h2o`)**: Maps the updated hidden state to the vocabulary logits (probabilities for each character).

3. **Training Protocol**:
   - The network is trained using Backpropagation Through Time (BPTT).
   - For a given training chunk, the model processes each character sequentially, predicting the next character.
   - The loss is calculated at each step using `nn.CrossEntropyLoss` and backpropagated at the end of the sequence.

4. **Name Generation**:
   - Starting with a specified prompt letter, the model iteratively predicts the next character.
   - **Temperature Scaling** is applied to the output logits (i.e. dividing the logits by a temperature value before taking the exponential/softmax). A lower temperature produces more conservative, realistic names, while a higher temperature introduces more creativity/randomness.
   - The process stops when the model generates the newline character (`\n`) or reaches a predefined maximum length.

---

## How to Replicate

Follow these steps to run the training and name generation on your system:

### 1. Project Directory Layout
Ensure your directory is structured as follows:
```
name_gen_rnn/
├── data/
│   └── Indian_Names.csv    # Your dataset containing a 'Name' column
├── main.py                 # Core model definition & training script
├── requirements.txt        # Package dependencies
└── README.md               # Project documentation (this file)
```

### 2. Set Up Virtual Environment
Initialize a clean Python virtual environment to manage dependencies:

**On Windows:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

### 4. Run the Training Script
Run the script to begin training the RNN. It will print the average loss and sample generations every 10,000 iterations:
```bash
python main.py
```
