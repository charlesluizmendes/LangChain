import torch
import os
import getpass

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

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Define o modelo
model_id = "microsoft/Phi-3-mini-4k-instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    model = model,
    tokenizer = tokenizer,
    task = "text-generation",
    temperature = 0.1,
    max_new_tokens = 500,
    do_sample = True,
    repetition_penalty = 1.1,
    return_full_text = False,
)

llm = HuggingFacePipeline(pipeline = pipe)

input = "Qual foi a primeira linguagem de programação?"
output = llm.invoke(input)

print()
print(output)
print()
