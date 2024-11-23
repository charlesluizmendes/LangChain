import torch
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline

from langchain_core.messages import (HumanMessage, SystemMessage)
from langchain_huggingface import ChatHuggingFace

from dotenv import load_dotenv

device = "cuda:0" if torch.cuda.is_available() else "cpu"

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

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
chat_model = ChatHuggingFace(llm = llm)

msgs = [
    SystemMessage(content = "Você é um assistente e está respondendo perguntas gerais."),
    HumanMessage(content = "Quem venceu a copa do mundo de 1970?")
]

model_template = tokenizer.chat_template

chat_model._to_chat_prompt(msgs)

res = chat_model.invoke(msgs)
print(res.content)