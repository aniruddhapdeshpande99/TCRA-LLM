# TCRA-LLM:Token Compression Retrieval Augmented LLM for Inference Cost Reduction 

This readme serves as a manual for setting up my reproduction of the paper TCRA-LLM. 

### Virtual Environment

* If you do not have Anaconda or Miniconda, please setup Conda using this [link](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
* Create the conda environment by running the following commands:

```
conda create -n "tcra_env" python=3.12.2 ipython
```

* Install the necessary requirements using the following command:

```pip install -r requirements.txt```

### Setting up Quantized Mistral 7B Model

* Download the model titled ```mistral-7b-instruct-v0.2.Q4_K_M.gguf``` from [4 Bit Quantized Mistral 7B Model](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/tree/main). 
* Create a directory titled ```models``` using ```mkdir {path_to_codebase}/models```.
* Place the model in the ```models``` folder: 

```
./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

### Dataset

* Download the Training and Development set for the Stanford Question Answering Dataset (SQuAD) v1.1 using this [link](https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset/code).
 
* Move and Unzip the dataset.

```
mv dev-v1.1.json.zip {path_to_codebase}/data
mv train-v1.1.json.zip {path_to_codebase}/data
unzip {path_to_codebase}/data/dev-v1.1.json.zip
unzip {path_to_codebase}/data/train-v1.1.json.zip
```

### Generating Datasets of Summaries (30%, 50%, 70%) and Fine Tuning mT5 model

* To generate Training Data to Fine Tune the mT5 Transformer for summarization, you will need need to use ```summarize_docs.py```.
* Open ```summarize_docs.py``` using a text editor or with ```vim```. On Line 6 modify the value of the variable ```EXP_LEN``` as follows:

  - ```EXP_LEN = 0.3``` for generating 30% summaries.
  - ```EXP_LEN = 0.5``` for generating 50% summaries.
  - ```EXP_LEN = 0.7``` for generating 70% summaries.

* After generating the summaries, mT5 model can be fine tuned using ```mT5_finetune.py```. 
* Different models for different amounts of summarization can be saved by modifying ```EXP_LEN``` on Line 13 as follows:
  - ```EXP_LEN = 0.3``` for finetuning mT5 for generating 30% summaries.
  - ```EXP_LEN = 0.5``` for finetuning mT5 for generating 50% summaries.
  - ```EXP_LEN = 0.7``` for finetuning mT5 for generating 70% summaries.

### Running RAG Pipeline and TCRA RAG Pipeline

There are two pipelines that you can run for querying over SQuAD. They are as follows:

* Full Context RAG: This RAG pipeline uses the complete dataset without any context compression (i.e., without using NSP and Summarization Compression). Run ```fullcontext_rag_pipeline.py``` using the following command:

```
python fullcontext_rag_pipeline.py
```

* TCRA LLM RAG: This RAG pipeline uses the TCRA pipeline. The context retrieved is passed through NSP and then through the specified Summarization Compression mT5 model. You will need to modify ```tcra_rag_pipeline.py``` to choose the specific mT5 summarization model. To do so modify ```EXP_LEN``` on Line 30 as follows:
  - ```EXP_LEN = 0.3``` for using finetuned mT5 that generates 30% summaries.
  - ```EXP_LEN = 0.5``` for using finetuned mT5 that generates 50% summaries.
  - ```EXP_LEN = 0.7``` for using finetuned mT5 that generates 70% summaries.


To use the TCRA RAG Pipeline run ```tcra_rag_pipeline.py``` using the following command:
```
python tcra_rag_pipeline.py
```

### Results

To evaluate the outputs, evaluator script of SQuAD was used from the ```evaluate``` library. Following are the results I achieved with my different experiments:

| Model                                    | F1 Score |
|------------------------------------------|----------|
| Plain Mistral - 7B                      | 38.58%   |
| Full Context Mistral - 7B RAG            | 85.73%   |
| Random Deletion Mistral - 7B RAG         | 57.16%   |
| Summarization Compression Mistral - 7B RAG (No Fine Tuning)  | 45.38%   |
| 50% Summarization Compression Mistral - 7B RAG (Fine-Tuned)  | 79.32%   |
| 70% Summarization Compression Mistral - 7B RAG (Fine-Tuned)  | 82.24%   |

### Author
* Aniruddha Prashant Deshpande 
* GT ID: 903945285
* GT Email: adeshpande322@gatech.edu
