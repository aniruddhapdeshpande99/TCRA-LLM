from llama_cpp import Llama
from nltk import word_tokenize
import json
import time

EXP_LEN = 0.7

llm = Llama(
    model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",  # Download the model file first
    n_ctx=32768,  # The max sequence length to use - note that longer sequence lengths require much more resources
    n_threads=16,            # The number of CPU threads to use, tailor to your system and the resulting performance
    n_gpu_layers=35         # The number of layers to offload to GPU, if you have GPU acceleration available
    )


def generate_summary(article):
    global EXP_LEN
    start_time = time.time()

    article_len = len(word_tokenize(article))

    summary = ""
    summary_len = 0
    exp_len = int(article_len*EXP_LEN)
    buffer = int(article_len*0.15)
    iter = 0

    print("="*50)
    print("\n\n")

    output = llm(
    f"<s>[INST] Summarize the following text, the length of the summary result is {EXP_LEN*100}% of the original text, keep the first sentence, and directly output your answer: \n\n {article}  [/INST]", # Prompt
    max_tokens=512,  # Generate up to 512 tokens
    stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
    echo=False        # Whether to echo the prompt
    )
    summary = output['choices'][0]['text']
    summary_len = len(word_tokenize(summary))

    if summary_len < exp_len + buffer and summary_len > exp_len - buffer:
        return {'orig': article, 'orig_len': article_len, 'summary': summary, 'summary_len': summary_len}


    while True:

        iter += 1        
        if summary_len < exp_len - buffer:
            print(f"After Iteration {iter} Shorter Summary Generated in {time.time() - start_time} seconds - Summarization output is:\n\n")
            print(summary)
            output = llm(
            f"<s>[INST] The length you generated is short, please regenerate a longer summary. Your original summary was as follows:\n\n {summary} \n\n  Summarize the following text again, the length of the summary result is {EXP_LEN*100}% of the original text, keep the first sentence, and directly output your answer. The original text is as follows\n\n:{article}  [/INST]", # Prompt
            max_tokens=512,  # Generate up to 512 tokens
            stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
            echo=False        # Whether to echo the prompt
            )
            summary = output['choices'][0]['text']
            summary_len = len(word_tokenize(summary))

        elif summary_len > exp_len + buffer:
            print(f"After Iteration {iter} Longer Summary Generated in {time.time() - start_time} seconds - Summarization output is:\n\n")
            print(summary)
            output = llm(
            f"<s>[INST] The length you generated is long, please regenerate a shorter summary. Your original summary was as follows:\n\n {summary} \n\n  Summarize the following text again, the length of the summary result is {EXP_LEN*100}% of the original text, keep the first sentence, and directly output your answer. The original text is as follows\n\n:{article}  [/INST]", # Prompt
            max_tokens=512,  # Generate up to 512 tokens
            stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
            echo=False        # Whether to echo the prompt
            )
            summary = output['choices'][0]['text']
            summary_len = len(word_tokenize(summary))
        else:
            print("FINAL SUMMARY GENERATED IN {time.time() - start_time} SECONDS")
            break
    
    print("-"*50)
    print(f"\nORIGINAL ARTICLE LENGTH: {article_len}")
    print("\nORIGINAL ARTICLE:")
    print(article)
    print(f"\nSUMMARY ARTICLE LENGTH: {summary_len}")
    print("\nSUMMARIZED ARTICLE")
    print(summary)

    return {'orig': article, 'orig_len': article_len, 'summary': summary, 'summary_len': summary_len}


d = None
with open('./data/train-v1.1.json') as f:
    d = json.load(f)

paras = []

for i in range(len(d['data'])):
    paragraphs = d['data'][i]['paragraphs']
    for j in range(len(paragraphs)):
        paras.append(paragraphs[j]['context'])

orig_count = 0
count = 0
summary_dict_arr = []
for para in paras:
    summary_dict = generate_summary(para)
    summary_dict_arr.append(summary_dict)
    count +=1

with open(f'./summarized_data/train_data_summarized_{EXP_LEN*100}.json', 'w') as fout:
            json.dump(summary_dict_arr , fout)

print(f"{EXP_LEN*100}% SUMMARIES FOR TRAINING DATA SAVED!")
orig_count = count
summary_dict_arr = []

d = None
with open('./data/val-v1.1.json') as f:
    d = json.load(f)

paras = []

for i in range(len(d['data'])):
    paragraphs = d['data'][i]['paragraphs']
    for j in range(len(paragraphs)):
        paras.append(paragraphs[j]['context'])

orig_count = 0
count = 0
summary_dict_arr = []
for para in paras:
    summary_dict = generate_summary(para)
    summary_dict_arr.append(summary_dict)
    count +=1

with open(f'./summarized_data/val_data_summarized_{EXP_LEN*100}.json', 'w') as fout:
            json.dump(summary_dict_arr , fout)

print(f"{EXP_LEN*100}% SUMMARIES FOR VALIDATION DATA SAVED!")
orig_count = count
summary_dict_arr = []