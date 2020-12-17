from transformers import BertForSequenceClassification, GPT2LMHeadModel, AutoTokenizer
from kogpt2_transformers import get_kogpt2_tokenizer
import torch.nn.functional as F
from time import time
import torch
import random
import csv
import re
import pickle
from time import sleep
import json

EMOJI = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                              "]+", re.UNICODE)

# Load Reranker model & tokenizer
print("Load Reranker model & tokenizer")
reranker_model = BertForSequenceClassification.from_pretrained("./src/models/reranker/checkpoint-920", num_labels=2)
reranker_tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
reranker_tokenizer.add_special_tokens({"additional_special_tokens":["[/]"]})
reranker_model.resize_token_embeddings(len(reranker_tokenizer))
reranker_model = reranker_model.to("cpu")
reranker_model.eval()

# Load Classifier model & tokenizer
print("Load Classifier model & tokenizer")
classifier_model = BertForSequenceClassification.from_pretrained("./src/models/classifier/checkpoint-190", num_labels=167)
classifier_tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
classifier_model = classifier_model.to("cpu")
classifier_model.eval()

# Load Generator model & tokenizer
print("Load Generator model & tokenizer")
generator_model = GPT2LMHeadModel.from_pretrained("./src/models/generator/checkpoint-851")
generator_tokenizer = get_kogpt2_tokenizer()
generator_tokenizer.add_special_tokens({"additional_special_tokens": ["<chatbot>"]})
generator_model.resize_token_embeddings(len(generator_tokenizer))
generator_model = generator_model.to("cpu")
generator_model.eval()
