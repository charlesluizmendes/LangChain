from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from dotenv import load_dotenv
import os
import re

# Carrega as variáveis de ambiente
load_dotenv()

# Obtém o token da API da Hugging Face do arquivo .env
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

if not huggingface_api_key:
    raise ValueError("Token da API da Hugging Face não encontrado no arquivo .env!")

# Configura o Hugging Face Endpoint sem usar headers diretamente
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct",
    temperature=0.1,
    max_new_tokens=512,
    model_kwargs={
        "headers": {"Authorization": f"Bearer {huggingface_api_key}"}
    }
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente prestativo e está respondendo perguntas gerais."),
    ("user", "Explique para mim em até {paragraph} parágrafo o conceito de {topic}, de forma clara e objetiva")
])

chain_str = prompt | llm | StrOutputParser()

res = chain_str.invoke({"paragraph": "2", "topic": "Buracos Negros"})

print()
print(re.sub(r"^\.\n\n", "", res))
print()
