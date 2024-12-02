# LangChain

Projeto de exemplo de utilização de LangChain

## Pacotes utilizados:

```
$ pip install bs4
$ pip install torch
$ pip install python-dotenv
$ pip install transformers
$ pip install accelerate
$ pip install bitsandbytes
$ pip install huggingface_hub
$ pip install langchain
$ pip install langchain_openai
$ pip install langchain_community
$ pip install langchain-huggingface
$ pip install langchain-ollama 
$ pip install langchain-chroma
$ pip install wikipedia
$ pip install faiss-cpu
$ pip install streamlit
$ pip install pypdf
```

## Instalar pacotes:

```
$ pip install -r requirements.txt
```

## Executar na CPU:

Se deseja executar os scripts em uma CPU basta nao utilizar o 'BitsAndBytesConfig', removendo do 'AutoModelForCausalLM':

Ex:

```
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_id) # basta remover o quantization_config aqui.
```

## Executar na GPU (cuda):

Execute o seguinte comando para listar os drivers de video e a respectiva versão da CUDA instalada:

```
$ nvidia-smi
# Deverá mostrar algo como isso logo abaixo
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.94                 Driver Version: 560.94         CUDA Version: 12.6     |
+-----------------------------------------+------------------------+----------------------+
```

* Embora o driver da NVIDIA suporte CUDA 12.6, ele é retrocompatível com versões anteriores de CUDA, como 12.1, 11.8, etc. Isso significa que o PyTorch, ao ser instalado com suporte a CUDA 12.1, ainda pode funcionar perfeitamente no seu sistema com CUDA 12.6.

Apos isso instale a versão correta da CUDA, no caso a versão '12.1':

```
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verifique se a ver~soa foi instalada corretamente, e se retornou 'True':

```
$ python -c "import torch; print(torch.cuda.is_available())"
# Deverá mostrar algo como isso logo abaixo
True
```

## example17

Para executar o projeto example17, basta entrar no diretorio do projeto e executar os scripts abaixo:

```
$ python example17.py
$ streamlit run example17.py  
```
