#!/usr/bin/env python
# coding: utf-8

# ## LLM evaluation of RAG on HotpotQA using different segmentation and retrieval configurations
# 
# Grant Boquet (grant.boquet@gmail.com)
# 
# November 6, 2023

# Create the evaluation dataset (uses HotpotQA from HuggingFace)

# In[1]:


import json
import os

openai_api_key=""

os.environ['OPENAI_API_KEY'] = openai_api_key

if os.path.exists("./hsbc_interview.json"):
    print("Loading existing dataset.")
    with open("./hsbc_interview.json", "rt") as fin:
        data = json.load(fin)
else:        

    from datasets import load_dataset
    
    dataset = load_dataset("hotpot_qa", "distractor")
    
    subset_data = []
    subset_text = []
    for idx, entry in enumerate(dataset["train"]):
        if len(subset_data) > 10:
            break
        subset_text.append(''.join([''.join(u) for u in entry["context"]["sentences"]]))
        subset_data.append({"query": entry["question"], "question": entry["question"], "answer": entry["answer"]})

    data = {"data": subset_data, "context": subset_text}
    
    with open("./hsbc_interview.json", "wt") as fout:
        json.dump(data, fout)


# In[2]:


from operator import itemgetter
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter, RecursiveCharacterTextSplitter, SpacyTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.evaluation.qa import QAEvalChain
from itertools import product


# Specify the testing conditions

# In[3]:


text_splitters = []
text_splitters.append(dict(name="CharacterTextSplitter/100/0", model=CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator="")))
text_splitters.append(dict(name="TokenTextSplitter/100/0", model=TokenTextSplitter(chunk_size=100, chunk_overlap=0)))
text_splitters.append(dict(name="SpacyTextSplitter", model=SpacyTextSplitter(pipeline='sentencizer')))

retriever_settings = []
retriever_settings.append(dict(search_type="mmr", search_kwargs={"k": 4}))
retriever_settings.append(dict(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .7, "k": 4}))
retriever_settings.append(dict(search_type="similarity", search_kwargs={"k": 4}))


# Create the RAG chain for a given text splitter and retriever configuration.

# In[4]:


def get_rag_chain(contexts, text_splitter, retriever_config):

    texts = [chunk for text in contexts for line in text_splitter.split_text(text) for chunk in line.split("\n\n")]
    db = Chroma.from_texts(texts, OpenAIEmbeddings())
    retriever = db.as_retriever(**retriever_config)
    
    prompt_template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": PROMPT})

    return qa


# We are also going to use the LLM to evaluate the QA performance.

# In[5]:


prompt_template = """You are a teacher grading a quiz. 
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either CORRECT or INCORRECT.

Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: true answer here
GRADE: CORRECT or INCORRECT here

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. Begin! 

QUESTION: {question}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}
GRADE:"""

GRADE_ANSWER_PROMPT_FAST = PromptTemplate(input_variables=["question", "result", "answer"], template=prompt_template)

eval_chain = (GRADE_ANSWER_PROMPT_FAST | ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) | StrOutputParser())


# Evaluate the different configurations and use the grading prompt to count how many are accurate

# In[7]:


performance = dict()

for text_splitter, retriever_setting in product(text_splitters, retriever_settings):
    output = []
    num_correct = 0
    qa_chain = get_rag_chain(data["context"], text_splitter["model"], retriever_setting)
    
    for entry in data["data"]:
        entry  = qa_chain.invoke(entry)
        result = eval_chain.invoke(entry)
        num_correct += 0 if result == "INCORRECT" else 1
        entry["eval"] = result
        output.append(entry)
        print(entry)
        
    performance[(text_splitter["name"], repr(retriever_setting))] = (num_correct, output)


# The Spacy tokenizer is clearly better. It remains to see if the following will improve accuracy:
#  * expanding context beyond a single sentence
#  * increasing the context window to include more sentences, chuck overlap, etc.
#  * adjusting relevancy with the similarity threshold and using more context

# In[11]:


for model_info, num_correct in sorted([(k, v[0]) for k, v in performance.items()], key=lambda u: u[1], reverse=True):
    print(f"#Correct: {num_correct}  Text Splitter: {model_info[0]} / Retrieval Setting: {model_info[1]}")


# In[ ]:




