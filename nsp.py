from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
import torch.nn.functional as F

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

def get_top_k_next_sentences(prompt, candidate_sentences, k=6):
    
    nsp_dict_list = []

    for next_sentence in candidate_sentences:
        with torch.no_grad():
            encoding = tokenizer(prompt, next_sentence, return_tensors="pt")
            outputs = model(**encoding, labels=torch.LongTensor([1]))  # next_sentence_label=1 as sentence_b follows sentence_a
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)
            probability_next_sentence = probabilities[:, 1].item()
            nsp_dict_list.append({'next_sentence': next_sentence, 'prob': probability_next_sentence})


    sorted_list_of_dicts = sorted(nsp_dict_list, key=lambda x: x['prob'], reverse=True)
    top_k_sentences = []

    for i in range(k):
        top_k_sentences.append(sorted_list_of_dicts[i]['next_sentence'])

    return top_k_sentences