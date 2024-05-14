import torch
from llama_index.llms.llama_cpp import LlamaCPP
from llama_cpp import Llama
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt

from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core import StorageContext, load_index_from_storage

from llama_index.core.storage.docstore import SimpleDocumentStore

from llama_index.core.storage.index_store import SimpleIndexStore

import os
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import VectorStoreIndex, ServiceContext, load_index_from_storage

from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank

import jsonlines
import time

from nsp import get_top_k_next_sentences

import json
from evaluate import load
from transformers import pipeline

model_name = "mt5-small"
EXP_LEN = 0.7
summarizer = pipeline("summarization", model=f"./models/{model_name}-finetuned-squad-{EXP_LEN*100}")

squad_metric = load("squad")

d = None
with open('./data/train-v1.1.json') as f:
    d = json.load(f)

paras = []

for i in range(len(d['data'])):
    paragraphs = d['data'][i]['paragraphs']
    for j in range(len(paragraphs)):
        paras.append(paragraphs[j]['context'])

start_time = time.time()
documents = []

for para in paras:
    documents.append(Document(text=para))

print(f"DATASET LOADED IN {time.time() - start_time} s")

def get_build_index(documents, llm, save_dir, embed_model="local:BAAI/bge-small-en-v1.5", sentence_window_size=3):
  
    node_parser = SentenceWindowNodeParser(
        window_size = sentence_window_size,
        window_metadata_key = "window",
        original_text_metadata_key = "original_text"
    )

    sentence_context = ServiceContext.from_defaults(
        llm = llm,
        embed_model= embed_model,
        node_parser = node_parser,
    )
  
    # create and load the index
    index = VectorStoreIndex.from_documents(
        documents, service_context=sentence_context
    )
    index.storage_context.persist(persist_dir=save_dir)
    return index

def retrieve_index(llm, persist_dir, sentence_window_size=3, embed_model="local:BAAI/bge-small-en-v1.5"):
    node_parser = SentenceWindowNodeParser(
        window_size = sentence_window_size,
        window_metadata_key = "window",
        original_text_metadata_key = "original_text"
    )

    sentence_context = ServiceContext.from_defaults(
        llm = llm,
        embed_model= embed_model,
        node_parser = node_parser,
    )
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    
    vector_index = load_index_from_storage(storage_context, service_context=sentence_context)
    return vector_index

def return_qa_prompt(query, context):
    query_str = "Following is the context that you need to understand: \n\n"
    for text in context:
        query_str = query_str + text + "\n\n"
    
    query_str += "Based on the above context, answer the following question in short. Be concise. If it is a one word or a few word answer, then answer accordingly. Use full sentences only when needed. The question is follows \n\n"

    query_str += query

    return query_str

start_time = time.time()
vector_index = None

# Llama CPP for RAG
llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    # model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf',
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=4096,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 35},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

if not os.path.exists("./vector_index/squad_vector_index/index/"):
    vector_index = get_build_index(documents=documents, llm=llm, embed_model="local:BAAI/bge-small-en-v1.5", sentence_window_size=3, save_dir="./vector_index/squad_vector_index/index/")
    print(f"INDEX CREATED IN {time.time() - start_time}")
else:
    vector_index = retrieve_index(llm, "./vector_index/squad_vector_index/index/", sentence_window_size=3, embed_model="local:BAAI/bge-small-en-v1.5")
    print(f"INDEX LOADED IN {time.time() - start_time} s")

start_time = time.time()
print(f"QUERY RETRIEVER ENGINE LOADED {time.time() - start_time} s")

print("="*50, end="\n\n\n")
print("Query:\n\n")
query = "What was the brand name of simvastatin?"
print("Correct Answer:\n\n")
print("Zocor\n\n")
start_time = time.time()
nodes = nodes = vector_index.retrieve(query)
context = [n.node.get_content() for n in nodes]
nsp_context = get_top_k_next_sentences(query, context, k=6)
summarized_context = [summarizer(text) for text in nsp_context]

response = llm.complete(return_qa_prompt(query, summarized_context))

print("Reponse:\n\n")
print(response)
print("\n")
print(f"RESPONSE GENRATED IN {time.time() - start_time} s")


results = squad_metric.compute(predictions=response, references=[{'answers': {'answer_start': [171], 'text': ['Zocor']}, 'id': '571d13cbdd7acb1400e4c212'}])
print("Evaluation Result for Query:\n\n")
print(results)

val_d = None
with open('./data/val-v1.1.json') as f:
    val_d = json.load(f)

val_ques = []
val_references = []

for i in range(len(val_d['data'])):
    paragraphs = d['data'][i]['paragraphs']
    for j in range(len(paragraphs)):
        curr_para_ques = (paragraphs[j]['qas'])
        for ques in curr_para_ques:
            val_ques.append(ques['question'])
            val_references.append({'answers': {'answer_start': [ques['answers'][0]["answer_start"]], 'text': [ques['answers'][0]["text"]]}, 'id': ques['id']})

print("="*50)
print("QA PERFORMANCE ON VALIDATION SET IS AS FOLLOWS:")
val_ques_responses = []
for i in range(len(val_ques)):
    nodes = nodes = vector_index.retrieve(val_ques[i])
    context = [n.node.get_content() for n in nodes]
    nsp_context = get_top_k_next_sentences(val_ques[i], context, k=6)
    summarized_context = [summarizer(text) for text in nsp_context]

    response = llm.complete(return_qa_prompt(val_ques[i], summarized_context))
    val_ques_responses.append(response)

val_results = squad_metric.compute(predictions=val_ques_responses, references=val_references)