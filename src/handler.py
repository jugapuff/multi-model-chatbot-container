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

with open("./src/models/label_dic", 'rb') as f:
    temp_dic = pickle.load(f) 
labels = sorted(temp_dic.keys())
class Handler:
    def __init__(self):

        pass

    def __call__(self, req):
        data = json.loads(req.input)
        query = data["query"]
        query = query[-40:]
        history = data["history"].split(">><<")
        if history == ['']:
            history = []
        candidates = data["candidates"].split(">><<")
        if len(data["candidates"])<2:
            candidates = []

        updated_history = []
        updated_candidates = []
        history.append(query)
        batch = classifier_tokenizer(query,
                                    add_special_tokens=True, 
                                    truncation = True,
                                    return_token_type_ids=True, 
                                    padding= True, 
                                    max_length=64,return_tensors="pt").to("cpu")
        
        classified = classifier_model(**batch)[0]
        classified = torch.topk(classified, k=1, dim=-1)


        if classified.values[0,0].item() > 5.6:
            final_res = random.choice(temp_dic[labels[classified.indices[0,0].item()]])

            if final_res not in history:
                
                updated_history = history + [final_res]
                updated_candidates = random.choices(candidates, k=int(len(candidates) * 0.7))
                sleep(0.3)
                return json.dumps({"updated_history":updated_history, "final_res":final_res, "updated_candidates":updated_candidates })

        tokenized_query_for_generator = generator_tokenizer(query+"<chatbot>",max_length=10,truncation=True, 
                                                            add_special_tokens=False, 
                                                            return_tensors='pt').to("cpu")
        before_gen_beam2 = time()
        gen_beam2 = generator_model.generate(tokenized_query_for_generator["input_ids"], 
                                max_length=23,
                                num_beams=2, 
                                repetition_penalty=1.7, 
                                temperature=1.6, 
                                top_p=0.9, 
                                do_sample=True, 
                                num_return_sequences=4, 
                                eos_token_id=1,
                                length_penalty=1.9,
                                pad_token_id=1)
        gen_beam4 = generator_model.generate(tokenized_query_for_generator["input_ids"],
                               max_length=23,
                               num_beams=3, 
                               repetition_penalty=1.7, 
                               temperature=1.5, 
                               top_p=0.96, 
                               do_sample=True, 
                               num_return_sequences=2, 
                               eos_token_id=1,
                               length_penalty=1.9,
                               pad_token_id=1)


        gen_responses = []

        for response in gen_beam2.tolist() + gen_beam4.tolist():
            if len(response) <= 23:
                text = generator_tokenizer.decode(response)
                limit = text.find("</s>", 1)
                text = text[: limit if limit != -1 else None]
                if "<chatbot>" in text:
                    gen_responses.append(text.split("<chatbot>")[1].replace("<unk>",""))

        gen_responses = list(set(gen_responses))

        history = history[-15:]


        total_responses = []
        candidates = data["candidates"].split(">><<")
        if len(data["candidates"])<2:
            total_responses = gen_responses
            if gen_responses == []:
                total_responses = ["무슨 말씀인지 알아 듣게 이야기해주세요."]
        else:
            total_responses = gen_responses + candidates
        before_rerank = time()
        
        
        tokenized_context_with_response_for_reranker = reranker_tokenizer(
            [("[/]".join(history[-5:]), r) for r in total_responses],
            add_special_tokens=True, 
            return_token_type_ids=True, 
            padding= True,
            truncation=True,
            return_tensors="pt"
        ).to("cpu")

        scores = reranker_model(**tokenized_context_with_response_for_reranker)
        scores = F.softmax(scores[0], dim=-1)[:,1]


        final_res = total_responses.pop(torch.argmax(scores).item())

        updated_candidates = list(set(total_responses) - set(random.choices(gen_responses, k=len(gen_responses)-4)))
        updated_history = history + [final_res]
        return json.dumps({"updated_history":updated_history, "final_res":final_res, "updated_candidates":updated_candidates })
