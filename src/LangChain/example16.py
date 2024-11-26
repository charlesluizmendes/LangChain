import os

from langchain_community.llms import HuggingFaceEndpoint
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv

# Variaveis
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Modelo e LLM
model_id = "meta-llama/Llama-3.2-3B-Instruct"
llm = HuggingFaceEndpoint(repo_id=model_id, temperature=0.1)

# Prompt
prompt = hub.pull("hwchase17/react")

# Tools
search = TavilySearchResults(max_results=2)
tools = [search]

# Agents
agent = create_react_agent(
    llm=llm, tools=tools, prompt=prompt, stop_sequence=True
)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

# Resposta
resp = agent_executor.invoke({"input": "Qual é o campeão da Libertadores de 2019?"})

# search_results = search.invoke("Qual é a populção de Paris?")
# print(search_results)
