import torch
from llama_index.llms.llama_cpp import LlamaCPP
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


import json
from evaluate import load
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

def get_query_engine(sentence_index, similarity_top_k=6, rerank_top_n=2):
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
      top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    
    return engine

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
query_engine = get_query_engine(sentence_index=vector_index, similarity_top_k=12, rerank_top_n=2)
print(f"QUERY ENGINE LOADED {time.time() - start_time} s")

print("="*50, end="\n\n\n")
print("Query:\n\n")
query = input()
print("Correct Answer:\n\n")
print("Zocor\n\n")
start_time = time.time()
response = query_engine.query("What was the brand name of simvastatin?")
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
    val_ques_responses.append(query_engine.query(val_ques[i]))

val_results = squad_metric.compute(predictions=val_ques_responses, references=val_references)