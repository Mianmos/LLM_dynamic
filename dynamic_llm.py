import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import time
import sqlite3
from torch.utils.data import Dataset, DataLoader

# Hyperparametry (ujisti se, že embedding_dim a hidden_size odpovídají app.py)
embedding_dim = 128
hidden_size = 256
num_layers = 1
learning_rate = 0.005 # Změna učební rychlosti pro větší model
num_epochs_dynamic = 10 # Zvýšení počtu epoch
sequence_length = 20
data_file = "initial_data.txt" # Použijeme pro počáteční data
model_save_path = "model.pth"
vocabulary_path = "vocabulary.json"
index_to_word_path = "index_to_word.json"
DATABASE_FILE = "training_data.db"

class TextDataset(Dataset):
    def __init__(self, data, word_to_index, sequence_length):
        self.data = data
        self.word_to_index = word_to_index
        self.sequence_length = sequence_length
        self.input_sequences, self.target_sequences = self._prepare_data()

    def _prepare_data(self):
        input_sequences = []
        target_sequences = []
        for sentence in self.data:
            tokens = sentence.lower().split() # Přidáno lower() pro konzistenci
            tokens = [token for token in tokens if token] # Odstranění prázdných tokenů
            if len(tokens) >= self.sequence_length:
                for i in range(len(tokens) - self.sequence_length):
                    input_seq = tokens[i:i + self.sequence_length]
                    target_seq = tokens[i + 1:i + self.sequence_length + 1]
                    input_sequences.append([self.word_to_index.get(word, self.word_to_index.get("<unk>", 0)) for word in input_seq])
                    target_sequences.append([self.word_to_index.get(word, self.word_to_index.get("<unk>", 0)) for word in target_seq])
            elif len(tokens) > 1:
                input_seq = tokens[:-1]
                target_seq = tokens[1:]
                input_seq = input_seq + ["<pad>"] * (self.sequence_length - len(input_seq))
                target_seq = target_seq + ["<pad>"] * (self.sequence_length - len(target_seq))
                input_sequences.append([self.word_to_index.get(word, self.word_to_index.get("<unk>", 0)) for word in input_seq])
                target_sequences.append([self.word_to_index.get(word, self.word_to_index.get("<unk>", 0)) for word in target_seq])
        return torch.LongTensor(input_sequences), torch.LongTensor(target_sequences)

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        return self.input_sequences[idx], self.target_sequences[idx]

def create_vocabulary(data):
    tokens = set()
    for sentence in data:
        for word in sentence.lower().split(): # Přidáno lower() pro konzistenci
            tokens.add(word)
    vocabulary = sorted(list(tokens))
    word_to_index = {word: i for i, word in enumerate(vocabulary)}
    index_to_word = {i: word for word, i in word_to_index.items()}
    return vocabulary, word_to_index, index_to_word

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

def train(model, data_loader, criterion, optimizer, num_epochs, index_to_word, save_path="model.pth"):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epocha [{epoch+1}/{num_epochs}], Ztráta: {total_loss / len(data_loader):.4f}')
    torch.save(model.state_dict(), save_path)
    print(f'Natrénovaná váhy uloženy do: {save_path}')

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

def load_training_data_from_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() + " <eos>" for line in f.readlines()]
    return []

def load_training_data_from_db():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT text FROM training_data")
    data = [row[0] for row in cursor.fetchall()]
    conn.close()
    return data

def save_vocab(word_to_index, index_to_word):
    with open(vocabulary_path, 'w', encoding='utf-8') as f:
        json.dump(word_to_index, f)
    with open(index_to_word_path, 'w', encoding='utf-8') as f:
        json.dump({k: v for k, v in index_to_word.items()}, f)

def load_vocab():
    vocabulary = None
    index_to_word = None
    if os.path.exists(vocabulary_path) and os.path.exists(index_to_word_path):
        try:
            with open(vocabulary_path, 'r', encoding='utf-8') as f:
                vocabulary = json.load(f)
            with open(index_to_word_path, 'r', encoding='utf-8') as f:
                index_to_word_list = json.load(f)
                index_to_word = {int(k): v for k, v in index_to_word_list.items()}
            print("Slovní zásoba načtena v dynamic_llm.py.")
        except Exception as e:
            print(f"Chyba při načítání slovní zásoby v dynamic_llm.py: {e}")
    return vocabulary, index_to_word

def update_vocabulary_and_data(initial_data_file, sequence_length):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT text FROM training_data")
    db_data = [row[0] for row in cursor.fetchall()]
    conn.close()

    initial_data = load_training_data_from_file(initial_data_file)
    all_data = initial_data + db_data

    vocabulary, word_to_index, index_to_word = create_vocabulary(all_data)
    for token in ["<unk>", "<pad>", "<eos>"]:
        if token not in word_to_index:
            index = len(word_to_index)
            word_to_index[token] = index
            index_to_word[index] = token
    vocab_size_current = len(word_to_index)

    dataset = TextDataset(all_data, word_to_index, sequence_length)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True) # Zvýšení batch_size pro efektivnější trénink

    return vocab_size_current, word_to_index, index_to_word, data_loader, all_data

def load_model(vocab_size, embedding_dim, hidden_size, num_layers):
    model = SimpleRNN(vocab_size, embedding_dim, hidden_size, num_layers)
    if os.path.exists(model_save_path):
        try:
            model.load_state_dict(torch.load(model_save_path))
            print(f"Načtena uložená váhy modelu z: {model_save_path} v dynamic_llm.py.")
        except Exception as e:
            print(f"Chyba při načítání uložených vah modelu v dynamic_llm.py: {e}")
    else:
        print("Nebyl nalezen uložený model v dynamic_llm.py, začíná se s novým modelem.")
    return model

if __name__ == '__main__':
    vocabulary, index_to_word = load_vocab()
    if vocabulary and index_to_word:
        vocab_size = len(vocabulary)
        model = load_model(vocab_size, embedding_dim, hidden_size, num_layers)
    else:
        vocab_size, word_to_index, index_to_word, data_loader, all_data = update_vocabulary_and_data(data_file, sequence_length)
        model = SimpleRNN(vocab_size, embedding_dim, hidden_size, num_layers)
        save_vocab(word_to_index, index_to_word)

    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.get("<pad>", 0) if vocabulary else 0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    last_modified = 0
    try:
        while True:
            current_modified = os.path.getmtime(DATABASE_FILE)
            print(f"Čas poslední modifikace databáze: {current_modified}, předchozí: {last_modified}") # Pro sledování
            if current_modified > last_modified:
                print("Nalezena nová trénovací data v databázi, spouští se dotrénování...")
                last_modified = current_modified
                vocab_size_current, word_to_index_current, index_to_word_current, data_loader_current, all_data_current = update_vocabulary_and_data(data_file, sequence_length)

                # Zkontrolujeme, zda se změnila velikost slovní zásoby a případně vytvoříme nový model
                if vocab_size_current > model.embedding.num_embeddings:
                    print(f"Slovní zásoba se rozšířila na {vocab_size_current}, vytvořen nový model.")
                    model = SimpleRNN(vocab_size_current, embedding_dim, hidden_size, num_layers)
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Reset optimizer pro nový model
                    save_vocab(word_to_index_current, index_to_word_current)
                elif vocab_size_current < model.embedding.num_embeddings:
                    print(f"Velikost slovní zásoby se zmenšila na {vocab_size_current}, vytvořen nový model.")
                    model = SimpleRNN(vocab_size_current, embedding_dim, hidden_size, num_layers)
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Reset optimizer pro nový model
                    save_vocab(word_to_index_current, index_to_word_current)
                else:
                    try:
                        model.load_state_dict(torch.load(model_save_path))
                        print("Načteny existující váhy modelu pro doučení.")
                    except Exception as e:
                        print(f"Chyba při načítání existujících vah pro doučení: {e}")

                # Volání funkce train() by mělo být zde, po kontrole (a případné úpravě) modelu.
                train(model, data_loader_current, criterion, optimizer, num_epochs_dynamic, index_to_word_current, model_save_path)
                vocabulary = word_to_index_current
                index_to_word = index_to_word_current
            else:
                time.sleep(10)

            # Interaktivní generování (ponecháno pro testování)
            start_sequence = input("Zadejte počáteční sekvenci pro generování (nebo 'konec'): ")
            if start_sequence.lower() == 'konec':
                break
            if vocabulary and index_to_word and model:
                generated_text = generate_text(model, start_sequence, 20, vocabulary, index_to_word)
                print(f"Generovaný text: {generated_text}")
            else:
                print("Slovní zásoba nebo model nejsou načteny.")

    except KeyboardInterrupt:
        print("\nUkončuji dynamický trénink.")
        if model and vocabulary and index_to_word:
            torch.save(model.state_dict(), model_save_path)
            save_vocab(vocabulary, index_to_word)