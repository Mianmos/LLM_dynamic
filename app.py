import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import time
from flask import Flask, request, render_template
import sqlite3

# Cesty k souborům
model_path = "model.pth"
vocabulary_path = "vocabulary.json"
index_to_word_path = "index_to_word.json"
DATABASE_FILE = "training_data.db"

app = Flask(__name__)
model = None
vocabulary = None
index_to_word = None
sequence_length = 20 # Měl by odpovídat hodnotě v dynamic_llm.py
embedding_dim = 128 # Měl by odpovídat hodnotě v dynamic_llm.py
hidden_size = 256 # Měl by odpovídat hodnotě v dynamic_llm.py
num_layers = 1 # Měl by odpovídat hodnotě v dynamic_llm.py

# Načtení slovní zásoby
def load_vocab():
    global vocabulary, index_to_word
    if os.path.exists(vocabulary_path) and os.path.exists(index_to_word_path):
        try:
            with open(vocabulary_path, 'r', encoding='utf-8') as f:
                vocabulary = json.load(f)
            with open(index_to_word_path, 'r', encoding='utf-8') as f:
                index_to_word_list = json.load(f)
                index_to_word = {int(k): v for k, v in index_to_word_list.items()}
            print("Slovní zásoba načtena v app.py.")
        except Exception as e:
            print(f"Chyba při načítání slovní zásoby v app.py: {e}")

# Definice modelu (musí odpovídat definici v dynamic_llm.py)
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = output.contiguous().view(-1, self.hidden_size)
        prediction = self.fc(output)
        return prediction

# Funkce pro načtení modelu
def load_model(vocab_size, embedding_dim, hidden_size, num_layers):
    global model
    model = SimpleRNN(vocab_size, embedding_dim, hidden_size, num_layers)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            model.eval() # Nastavíme model do eval módu pro generování
            print(f"Načtena uložená váhy modelu z: {model_path} v app.py.")
        except Exception as e:
            print(f"Chyba při načítání modelu v app.py: {e}")
    else:
        print("Model nebyl nalezen v app.py.")

# Funkce pro generování textu
def generate_text(model, start_sequence, length, word_to_index, index_to_word, device="cpu"):
    model.eval()
    tokens = start_sequence.lower().split()
    generated = tokens[:]
    model.to(device)
    for _ in range(length):
        input_eval = torch.LongTensor([word_to_index.get(token, word_to_index.get("<unk>", 0)) for token in tokens]).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_eval)
            last_word_logits = output[-1, :]
            probabilities = torch.softmax(last_word_logits, dim=-1).cpu().numpy()
            predicted_index = np.argmax(probabilities)
            predicted_word = index_to_word.get(predicted_index, "<unk>")
            generated.append(predicted_word)
            tokens = generated[-sequence_length:]
    return " ".join(generated)

@app.route('/', methods=['GET', 'POST'])
def home():
    global model, vocabulary, index_to_word, embedding_dim, hidden_size, num_layers
    generated_text = None
    if not vocabulary or not index_to_word:
        load_vocab()
        if vocabulary and index_to_word and model is None:
            vocab_size = len(vocabulary)
            load_model(vocab_size, embedding_dim, hidden_size, num_layers)
    elif model is None and vocabulary and index_to_word:
        vocab_size = len(vocabulary)
        load_model(vocab_size, embedding_dim, hidden_size, num_layers)

    if request.method == 'POST':
        start_sequence = request.form['start_sequence']
        new_text = request.form.get('new_text')

        if new_text and new_text.strip():
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS training_data (text TEXT)")
            cursor.execute("INSERT INTO training_data (text) VALUES (?)", (new_text + " <eos>",))
            conn.commit()
            conn.close()
            print(f"Nový text uložen do databáze: {new_text}")

        if start_sequence and start_sequence.strip() and model and vocabulary and index_to_word:
            generated_text = generate_text(model, start_sequence, 20, vocabulary, index_to_word)
        elif request.form['start_sequence'] == '':
            generated_text = "Zadejte prosím počáteční sekvenci pro generování."

    return render_template('index.html', generated_text=generated_text)

if __name__ == '__main__':
    load_vocab()
    if vocabulary and index_to_word and model is None:
        vocab_size = len(vocabulary)
        load_model(vocab_size, embedding_dim, hidden_size, num_layers)
    app.run(debug=True)