import os

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()
os.environ.get("OPENAI_API_KEY")

chatgpt = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-3.5-turbo"    
)

msgs = [
    (
        "system", "Você é um assistente prestativo que traduz do português para francês. Traduza a frase do usuário.",
    ),
    (
        "user", "Eu amo programação"
    )
]

ai_msg = chatgpt.invoke(msgs)
ai_msg

print(ai_msg.content)