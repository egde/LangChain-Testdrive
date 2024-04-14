from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI
from langchain import hub

from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.youtube.search import YouTubeSearchTool


from decouple import config
from loguru import logger

# Defining the tools

youtube = YouTubeSearchTool()
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [youtube, wiki]

prompt = hub.pull("hwchase17/react")

memory = ConversationBufferMemory(memory_key="chat_history")
llm = OpenAI(api_key=config("OPENAI_API_KEY"))

agent = create_react_agent(llm=llm, prompt=prompt, tools=tools)

agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)

result = agent_chain.invoke(
        input={
            "input": "Tell me about LLMs and a set of links on Youtube to learn more"
        }
    )

logger.info(
    "Result: \n"
    + str(result.get('output'))
)
