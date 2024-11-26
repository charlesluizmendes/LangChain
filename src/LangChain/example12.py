from langchain_core.tools import Tool
from langchain_community.tools import WikipediaQueryRun, BaseTool
from langchain_community.utilities import WikipediaAPIWrapper

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=3000))

wikipedia_tool = Tool(name = "wikipedia",
                      description="You must never search for multiple concepts at a single step, you need to search one at a time. When asked to compare two concepts for example, search for each one individually. Answer the question in Portuguese",
                      func=wikipedia.run)

res = wikipedia_tool.run("Oscar Niemeyer")

print()
print(res)
print()