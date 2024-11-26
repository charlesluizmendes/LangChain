import os

from langchain_core.tools import Tool
from langchain_community.tools import WikipediaQueryRun, BaseTool
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub


from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


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
prompt = hub.pull("hwchase17/react")


#
llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)


#
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt, stop_sequence=True)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


#
resp = agent_executor.invoke({"input": "Qual é a população de Paris?"})