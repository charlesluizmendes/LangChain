import torch
import os
import re
import getpass
import bs4

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline

from langchain.prompts import PromptTemplate
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

device = "cuda:0" if torch.cuda.is_available() else "cpu"


# Parametros
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# Modelo
model_id = "meta-llama/Llama-3.2-3B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.1,
    max_new_tokens=500,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=pipe)


# Carregamento de Pagina (Document Corpus)
loader = WebBaseLoader(web_paths = ("https://www.bbc.com/portuguese/articles/cd19vexw0y1o",),)
docs = loader.load()


# Divisão de Texto (Chunked)
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, add_start_index = True)
splits = text_splitter.split_documents(docs)


# Armazenamento (Embeddings)
hf_embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")


# Salvar (Vectors)
vectorstore = Chroma.from_documents(documents=splits, embedding=hf_embeddings)


# Recuperar
retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs={"k": 6})


# Template
template_rag = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
Você é um assistente virtual prestativo e está respondendo perguntas gerais.
Use os seguintes pedaços de contexto recuperado para responder à pergunta.
Se você não sabe a resposta, apenas diga que não sabe. Mantenha a resposta concisa.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Pergunta: {question}
Contexto: {context}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

# Prompt
prompt_rag = PromptTemplate(
    input_variables=["context", "question"],
    template=template_rag,
)

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

chain_rag = ({"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_rag
            | llm
            | StrOutputParser())

res = chain_rag.invoke("Emma Stone?")

print()
print(res)
print()

# Limpar Todos os Dados
# vectorstore.delete_collection()