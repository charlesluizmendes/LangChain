from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
import os
from dotenv import load_dotenv

device = "cuda:0" if torch.cuda.is_available() else "cpu"

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

torch.random.manual_seed(42)

id_model = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    id_model,
    device_map = "cuda",
    torch_dtype = "auto",
    trust_remote_code = True,
    attn_implementation="eager"
)

tokenizer = AutoTokenizer.from_pretrained(id_model)
pipe = pipeline("text-generation", model = model, tokenizer = tokenizer)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.1, # 0.1 até 0.9
    "do_sample": True,
}

# sys_prompt = "Você é um programador experiente. Retorne o código requisitado e forneça explicações breves se achar conveniente."
# prompt = "Gere um código em python que escreva a sequência de fibonnaci"

# template = """<|system|>
# {}<|end|>
# <|user|>
# "{}"<|end|>
# <|assistant|>""".format(sys_prompt, prompt)

# output = pipe(template, **generation_args)
# print(output[0]['generated_text'])

prompt_sys = "Você é um assistente de viagens prestativo. Responda as perguntas em português."
prompt = "Liste o nome de 10 cidades famosas da Europa"

msg = [
    {"role": "system", "content": prompt_sys},
    {"role": "user", "content": prompt},
]

output = pipe(msg, **generation_args)
print()
print(output[0]['generated_text'])
print()