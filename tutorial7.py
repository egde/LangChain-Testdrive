from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI
from langchain import hub
from langchain.tools import BaseTool
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.youtube.search import YouTubeSearchTool

from pyowm.owm import OWM
from pyowm.utils.config import get_default_config

from typing import Type
from pydantic import BaseModel, Field
from decouple import config
from loguru import logger

# Defining the tools
def get_weather(city):
    config_dict = get_default_config()
    config_dict['language'] = 'en'
    owm = OWM(api_key=config('OPENWEATHER_API_KEY'))
    mgr = owm.weather_manager()
    observation = mgr.weather_at_place(city)
    
    temperature = str(observation.weather.temperature("celsius")["temp"])+'°C'
    humidity = str(observation.weather.humidity)+'%'
    wind = str(observation.weather.wind)+'m/s'
    status = observation.weather.detailed_status

    return {
        'temperature': temperature,
        'humid': humidity,
        'wind': wind,
        'status': status
    }

class GetWeatherInput(BaseModel):
    """Inputs for get_weather"""
    city:str=Field(description='City name and country seperated by a comma')

class GetWeatherTool(BaseTool):
    name="get_weather"
    description="""
        Useful to get the weather details of a city.
        Mandatory input format is 'city,country'
        """
    args_schema = GetWeatherInput

    def _run(self, city:str):
        weather = get_weather(city=city)
        return weather


tools = [GetWeatherTool()]

prompt = hub.pull("hwchase17/react")

memory = ConversationBufferMemory(memory_key="chat_history")
llm = OpenAI(api_key=config("OPENAI_API_KEY"))

agent = create_react_agent(llm=llm, prompt=prompt, tools=tools)

agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)

result = agent_chain.invoke(
        input={
            "input": "What can I wear for tomorrow in Frankfurt am Main?"
        }
    )

logger.info(
    "Result: \n"
    + str(result.get('output'))
)
