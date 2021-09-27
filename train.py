import numpy as np
import random
import json
from utils import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tqdm
from model import Seq2seqAttention
from nltk_utils import bag_of_words, tokenize, stem

# %%
############################################
# STEP 1. data_preprocessing
############################################
# Loading the pre-processed conversational exchanges (source-target pairs) from pickle data files
all_conversations = load_from_pickle("./data/processed_CMDC.pkl")
# Extract 100 conversations from the end for evaluation and keep the rest for training
eval_conversations = all_conversations[-100:]
all_conversations = all_conversations[:-100]

# Logging data stats
print(f"Number of Training Conversation Pairs = {len(all_conversations)}")
print(f"Number of Evaluation Conversation Pairs = {len(eval_conversations)}")
# print_list(all_conversations, 5) # print the double check


# %%
############################################
# STEP 2. check GPU
############################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Using device:", device)


# %%
############################################
# STEP 3. tokenize vocabulary
############################################
pad_word = "<pad>"
bos_word = "<s>"
eos_word = "</s>"
unk_word = "<unk>"
pad_id = 0
bos_id = 1
eos_id = 2
unk_id = 3


def normalize_sentence(s):
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


class Vocabulary:
    def __init__(self):
        self.word_to_id = {pad_word: pad_id, bos_word: bos_id, eos_word: eos_id, unk_word: unk_id}
        self.word_count = {}
        self.id_to_word = {pad_id: pad_word, bos_id: bos_word, eos_id: eos_word, unk_id: unk_word}
        self.num_words = 4

    def get_ids_from_sentence(self, sentence):
        sentence = normalize_sentence(sentence)
        sent_ids = [bos_id] + [self.word_to_id[word] if word in self.word_to_id \
                                   else unk_id for word in sentence.split()] + \
                   [eos_id]
        return sent_ids

    def tokenized_sentence(self, sentence):
        sent_ids = self.get_ids_from_sentence(sentence)
        return [self.id_to_word[word_id] for word_id in sent_ids]

    def decode_sentence_from_ids(self, sent_ids):
        words = list()
        for i, word_id in enumerate(sent_ids):
            if word_id in [bos_id, eos_id, pad_id]:
                # Skip these words
                continue
            else:
                words.append(self.id_to_word[word_id])
        return ' '.join(words)

    def add_words_from_sentence(self, sentence):
        sentence = normalize_sentence(sentence)
        for word in sentence.split():
            if word not in self.word_to_id:
                # add this word to the vocabulary
                self.word_to_id[word] = self.num_words
                self.id_to_word[self.num_words] = word
                self.word_count[word] = 1
                self.num_words += 1
            else:
                # update the word count
                self.word_count[word] += 1


vocab = Vocabulary()
for src, tgt in all_conversations:
    vocab.add_words_from_sentence(src)
    vocab.add_words_from_sentence(tgt)
print(f"Total words in the vocabulary = {vocab.num_words}")


# %%
########################################################################################
# STEP 4. re-organize Cornell Movie Dialog Cropus dataset to make it easier to read
########################################################################################
class SingleTurnMovieDialog_dataset(Dataset):

    def __init__(self, conversations, vocab, device):
        self.conversations = conversations
        self.vocab = vocab
        self.device = device

        def encode(src, tgt):
            src_ids = self.vocab.get_ids_from_sentence(src)
            tgt_ids = self.vocab.get_ids_from_sentence(tgt)
            return (src_ids, tgt_ids)

        # We will pre-tokenize the conversations and save in id lists for later use
        self.tokenized_conversations = [encode(src, tgt) for src, tgt in self.conversations]

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {"conv_ids": self.tokenized_conversations[idx], "conv": self.conversations[idx]}


def collate_fn(data):
    # Sort conv_ids based on decreasing order of the src_lengths.
    # This is required for efficient GPU computations.
    src_ids = [torch.LongTensor(e["conv_ids"][0]) for e in data]
    tgt_ids = [torch.LongTensor(e["conv_ids"][1]) for e in data]
    src_str = [e["conv"][0] for e in data]
    tgt_str = [e["conv"][1] for e in data]
    data = list(zip(src_ids, tgt_ids, src_str, tgt_str))
    data.sort(key=lambda x: len(x[0]), reverse=True)
    src_ids, tgt_ids, src_str, tgt_str = zip(*data)

    # Pad the src_ids and tgt_ids using token pad_id to create src_seqs and tgt_seqs
    src_seqs = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first=False, padding_value=pad_id).type(
        torch.LongTensor).to(device)
    tgt_seqs = torch.nn.utils.rnn.pad_sequence(tgt_ids, batch_first=False, padding_value=pad_id).type(
        torch.LongTensor).to(device)

    return {"conv_ids": (src_ids, tgt_ids), "conv": (src_str, tgt_str),
            "conv_tensors": (src_seqs.to(device), tgt_seqs.to(device))}


# %%
########################################################################################
# STEP 5. actual training
########################################################################################
def train(model, data_loader, num_epochs, model_file, learning_rate=0.0001):
    decoder_learning_ratio = 5.0

    encoder_parameter_names = ['embedding', 'encoder']  # MY CODE
    encoder_named_params = list(
        filter(lambda kv: any(key in kv[0] for key in encoder_parameter_names), model.named_parameters()))
    decoder_named_params = list(
        filter(lambda kv: not any(key in kv[0] for key in encoder_parameter_names), model.named_parameters()))
    encoder_params = [e[1] for e in encoder_named_params]
    decoder_params = [e[1] for e in decoder_named_params]
    optimizer = torch.optim.AdamW([{'params': encoder_params},
                                   {'params': decoder_params, 'lr': learning_rate * decoder_learning_ratio}],
                                  lr=learning_rate)

    clip = 50.0
    for epoch in tqdm.tqdm(range(num_epochs)): # tqdm.notebook.trange(num_epochs)
        # print(f"Total training instances = {len(train_dataset)}")
        # print(f"train_data_loader = {len(train_data_loader)} {1180 > len(train_data_loader)/20}")
        # with tqdm.notebook.tqdm(
        #         data_loader,
        #         desc="epoch {}".format(epoch + 1),
        #         unit="batch",
        #         total=len(data_loader)) as batch_iterator:
        model.train()
        total_loss = 0.0
        batch_iterator = data_loader
        for i, batch_data in tqdm(enumerate(batch_iterator, start=0)):
            source, target = batch_data["conv_tensors"]
            optimizer.zero_grad()
            loss = model.compute_loss(source, target)
            total_loss += loss.item()
            loss.backward()
            # Gradient clipping before taking the step
            _ = nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            # batch_iterator.set_postfix(mean_loss=total_loss / i, current_loss=loss.item())

    # Save the model after training
    torch.save(model.state_dict(), model_file)

############################################
# Main driver code to start training
############################################

if __name__ == "__main__":
    # Create the DataLoader for all_conversations
    dataset = SingleTurnMovieDialog_dataset(all_conversations, vocab, device)

    # Hyper-parameters
    num_epochs = 6
    batch_size = 64

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    attention_model = Seq2seqAttention(vocab).to(device)

    FILE = "attention_model_results.pt"
    train(attention_model, data_loader, num_epochs, FILE)
    print(f'training complete. file saved to {FILE}')