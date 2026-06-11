import csv
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
df = pd.read_csv('data/Indian_Names.csv')

#Data Cleanup
#1. Clean column, lowercasing everything , converting to list of strings
names_list=df['Name'].dropna().astype(str).str.lower().tolist()
#2. Filter out empty string / non alphabet strings
names_list=[name for name in names_list if name.isalpha()]
#3. Join names with newline characters
text_dataset = '\n'.join(names_list)
#Extract Unique Characters from this string
all_characters = sorted(list(set(text_dataset)))
vocab_size=len(all_characters)
#map characters to integers and vice versa
char_to_idx = {
    char : idx for idx, char in enumerate(all_characters)
}
idx_to_char = {
    idx : char for idx, char in enumerate(all_characters)
}


#Turn characters into 1-hot encoded vectors
#Convert to long tensor first
def name_to_input_tensor(name , vocab_size, char_to_idx):
    temp_tensor = torch.tensor([char_to_idx[c] for c in name], dtype=torch.long)
    one_hot=torch.nn.functional.one_hot(temp_tensor, num_classes=vocab_size).float()
    return one_hot.unsqueeze(1)

class CustomRNN(nn.Module):
    def __init__(self , input_size, hidden_size, output_size):
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        
        #Define Layers
        #Input layer to convert 1-hot vector and previous hidden state to new hidden state
        self.i2h = nn.Linear(input_size+hidden_size, hidden_size)
        
        #Output Layer maps hidden layer to output size for logits
        self.h2o = nn.Linear(hidden_size, output_size)

        #Activation function tanh
        self.tanh=nn.Tanh()

    #Forward pass
    def forward(self, input_tensor, hidden_tensor):
        #concatenate input and hidden state
        combined = torch.cat([input_tensor, hidden_tensor], 1)
        
        #Compute next hidden state
        hidden_next  = self.tanh(self.i2h(combined))

        #Compute output logits
        output_logits = self.h2o(hidden_next)
        return output_logits, hidden_next

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
import random

def name_to_target_tensor(string, char_to_idx):
    target_indices = [char_to_idx[char] for char in string]
    return torch.tensor(target_indices, dtype=torch.long)

def get_random_training_pair(text_data, chunk_len=50):
    start_idx = random.randint(0, len(text_data) - chunk_len - 1)
    end_idx = start_idx + chunk_len + 1
    
    full_chunk = text_data[start_idx:end_idx]
    input_str = full_chunk[:-1]
    target_str = full_chunk[1:]
    
    input_tensor = name_to_input_tensor(input_str, vocab_size, char_to_idx)
    target_tensor = name_to_target_tensor(target_str, char_to_idx)
    
    return input_tensor, target_tensor

#Instantiate Model
hidden_size = 128
input_size = vocab_size
output_size = vocab_size
model = CustomRNN(input_size, hidden_size, output_size)
print(model)

#Training Loop
#Loss function
criterion=nn.CrossEntropyLoss()
#Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
def train_step(input_tensor , target_tensor):
    #Initialize hidden state to 0s
    hidden = model.init_hidden()

    #Reset Gradients from  previous step
    optimizer.zero_grad()
    #Initialize Loss to 0
    loss = 0
    seq_len = input_tensor.size(0)

    #Unroll across time steps
    for t in range(seq_len):
        x_t = input_tensor[t]
        output,hidden = model(x_t,hidden)
        loss+=criterion(output , target_tensor[t].unsqueeze(0))
    #Bptt
    loss.backward()
    #update weight matrix
    optimizer.step()
    #return average loss per character
    return loss.item()/seq_len
def Generate_name(starting_letter, max_len=20, temperature=0.85):
    with torch.no_grad():
        hidden=model.init_hidden()
        current_char=starting_letter.lower()
        generated_name=starting_letter

        for i in range(max_len -1):
            input_tensor=name_to_input_tensor(current_char,vocab_size,char_to_idx)
            output, hidden = model(input_tensor[0], hidden)
            #Apply temperature scaling 
            output_dist = output.data.view(-1).div(temperature).exp()
            sampled_idx = torch.multinomial(output_dist, 1).item()
            #get next char from dictionary
            next_char= idx_to_char[sampled_idx]
            #Break if we generate an end token 
            if next_char== '\n':
                break
            generated_name += next_char
            current_char = next_char
            
        return generated_name
#generate names

#Master loop

import time

n_iterations = 50000
print_every = 10000
current_loss = 0
start_time = time.time()

print("Training Started... Watch the names transform from gibberish to real names \n")

for iteration in range(1, n_iterations + 1):
    # 1. Grab a random input/target pair from our dataset
    input_tensor, target_tensor = get_random_training_pair(text_dataset, chunk_len=50)
    
    # 2. Run a single gradient descent training step
    loss = train_step(input_tensor, target_tensor)
    current_loss += loss
    
    # 3. Print out monitoring metrics and sample generations
    if iteration % print_every == 0:
        avg_loss = current_loss / print_every
        elapsed = time.time() - start_time
        print(f"Iteration: {iteration:4d} | Time Elapsed: {elapsed:.1f}s | Avg Loss: {avg_loss:.4f}")
        
        # Test generation with a few random starting letters
        test_letters = ['a', 'r', 'm', 'h']
        generated_samples = [Generate_name(letter, temperature=0.6) for letter in test_letters]
        print(f"Sample Generations: {', '.join(generated_samples)}")
        print("-" * 70)
        
        current_loss = 0
                


