import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import random
import numpy as np

# Load datasets
gossipcop_real = pd.read_csv('gossipcop_real.csv')
gossipcop_fake = pd.read_csv('gossipcop_fake.csv')
politifact_real = pd.read_csv('politifact_real.csv')
politifact_fake = pd.read_csv('politifact_fake.csv')
gossipcop_real['label'] = 1
gossipcop_fake['label'] = 0
politifact_real['label'] = 1
politifact_fake['label'] = 0

# Combine datasets
all_texts = pd.concat([politifact_real['title'], politifact_fake['title'], gossipcop_real['title'], gossipcop_fake['title']])
all_labels = pd.concat([politifact_real['label'], politifact_fake['label'], gossipcop_real['label'], gossipcop_fake['label']])

# Data augmentation
def data_augmentation(text):
    augmented_text = text
    for _ in range(random.randint(1, 3)):
        augmented_text = data_augmentation_step(augmented_text)
    return augmented_text

def data_augmentation_step(text):
    text_list = text.split()
    if len(text_list) > 3:
        idx = random.choice(range(1, len(text_list) - 1))
        synonym = get_synonym(text_list[idx])
        if synonym:
            text_list[idx] = synonym
    return ' '.join(text_list)

def get_synonym(word):
    try:
        from synonymizer import Synonymizer
        synonymizer = Synonymizer(lang='en')
        synonyms = synonymizer.get_synonyms(word)
        if synonyms:
            return random.choice(synonyms)
    except ImportError:
        pass
    return None

# Data augmentation loop
augmented_texts = []
augmented_labels = []
for text, label in zip(all_texts, all_labels):
    augmented_texts.append(data_augmentation(text))
    augmented_labels.append(label)
print("Run - 1")
all_texts = pd.concat([all_texts, pd.Series(augmented_texts)])
all_labels = pd.concat([all_labels, pd.Series(augmented_labels)])

# Data Splitting with Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_model_state_dict = None
best_accuracy = 0.0
best_f1_score = 0.0
x=0
for fold, (train_index, test_index) in enumerate(skf.split(all_texts, all_labels)):
    if(x==1):
        break;
    #print(f"Fold {fold+1}/{skf.n_splits}")
    train_texts = all_texts.iloc[train_index]
    train_labels = all_labels.iloc[train_index]
    test_texts = all_texts.iloc[test_index]
    test_labels = all_labels.iloc[test_index]

    # Tokenization
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    print("Run - 2")
    # Tokenize all texts in one go
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)

    # Convert to PyTorch tensors
    train_input_ids = torch.tensor(train_encodings['input_ids'])
    train_attention_mask = torch.tensor(train_encodings['attention_mask'])
    train_labels = torch.tensor(train_labels.tolist())
    print("Run - 3")
    test_input_ids = torch.tensor(test_encodings['input_ids'])
    test_attention_mask = torch.tensor(test_encodings['attention_mask'])
    test_labels = torch.tensor(test_labels.tolist())
    print("Run - 4")
    # Create DataLoader
    batch_size = 16
    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Run - 5")
    # Define BERT-based classifier model
    class BERTClassifier(nn.Module):
        def __init__(self, bert_model, hidden_dim, num_classes):
            super(BERTClassifier, self).__init__()
            self.bert = bert_model
            self.fc = nn.Linear(self.bert.config.hidden_size, hidden_dim)
            self.dropout = nn.Dropout(0.5)
            self.classifier = nn.Linear(hidden_dim, num_classes)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            pooled_output = self.dropout(pooled_output)
            fc_output = F.relu(self.fc(pooled_output))
            fc_output = self.dropout(fc_output)
            logits = self.classifier(fc_output)
            return logits

    # Initialize BERT model and classifier
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    classifier = BERTClassifier(bert_model, hidden_dim=256, num_classes=2)
    print("Run - 6")
    # Model, optimizer, and criterion initialization
    optimizer = AdamW(classifier.parameters(), lr=3e-5, eps=1e-8)
    criterion = nn.CrossEntropyLoss()
    print("Run - 7")
    # Learning rate scheduler
    total_steps = len(train_loader) * 5  # 5 epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    print("Run - 8")
    # Training loop
    classifier.train()
    for epoch in tqdm(range(5)):  # 5 epochs
        print("epoch :- ",epoch)
        for batch_idx, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = classifier(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

    # Evaluation
    classifier.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        print("Run - 9")
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            logits = classifier(input_ids, attention_mask)
            _, predicted = torch.max(logits, dim=1)
            predictions.extend(predicted.tolist())
            true_labels.extend(labels.tolist())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    print(f"Test Accuracy: {accuracy:.4f} - F1-Score: {f1:.4f}")

    # Check for improvement
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_state_dict = classifier.state_dict()

    if f1 > best_f1_score:
        best_f1_score = f1
        best_model_state_dict = classifier.state_dict()
    x=1;
# Save the best model
if best_model_state_dict is not None:
    torch.save(best_model_state_dict, "fakenews_best_model.pth")

print("Best Test Accuracy:", best_accuracy)
print("Best F1-Score:", best_f1_score)