import os

from langchain_core.tools import Tool
from langchain_community.tools import WikipediaQueryRun, BaseTool
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

from dotenv import load_dotenv


#
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")


#
def current_day(*args, **kwargs):
  from datetime import date

  day = date.today()
  day = day.strftime('%d/%m/%Y')
  return day


# 
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=3000))
wikipedia_tool = Tool(name = "wikipedia", description="You must never search for multiple concepts at a single step, you need to search one at a time. When asked to compare two concepts for example, search for each one individually. Answer the question in Portuguese", func=wikipedia.run)
date_tool = Tool(name="Day", func = current_day, description = "Use when you need to know the current date")
tools = [wikipedia_tool, date_tool]


#
model_id = "meta-llama/Llama-3.2-3B-Instruct"
llm = HuggingFaceEndpoint(repo_id=model_id, temperature=0.1)


#
prompt = hub.pull("hwchase17/react")


#
agent = create_react_agent(llm = llm, tools = tools, prompt = prompt)
agent_executor = AgentExecutor.from_agent_and_tools(agent = agent, tools = tools, verbose = True, handling_parsing_errors = True)


#
print(prompt.template)
resp = agent_executor.invoke({"input": "Qual a população de Paris?"})