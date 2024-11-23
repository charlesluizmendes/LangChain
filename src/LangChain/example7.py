
import torch
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline

from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from dotenv import load_dotenv

device = "cuda:0" if torch.cuda.is_available() else "cpu"

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACE_API_KEY")

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
    max_new_tokens = 1000,
    do_sample = True,
    repetition_penalty = 1.1,
    return_full_text = False,
)

llm = HuggingFacePipeline(pipeline = pipe)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente e está respondendo perguntas gerais."),
    ("user", "Explique-me em {paragraph} parágrafo o conceito de {topic}")
])

# chain = prompt | llm

# res = chain.invoke({"paragraph": "2", "topic": "Buracos Negros"})

# print()
# print(re.sub(r"^\.\n\n", "", res))
# print()

chain_str = prompt | llm | StrOutputParser()

for chunk in chain_str.stream({"paragraph": "3", "topic": "Senhor dos Aneis"}):
  print(chunk, end="")