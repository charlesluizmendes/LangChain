import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

# Define o modelo
model_id = "microsoft/Phi-3-mini-4k-instruct"

# Carregar modelo e tokenizer diretamente na CPU
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Criar pipeline para geração de texto
pipe = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.1,
    max_new_tokens=500,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=False
)

# Integrar com LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# prompt
input_text = "Quem foi a primeira pessoa no espaço?"
output = llm.invoke(input_text)

# Resposta
print(output)
