from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.output_parsers import OutputFixingParser
from langchain.pydantic_v1 import BaseModel, Field
from decouple import config
from loguru import logger

# Defining the Output Parser
class Scorer(BaseModel):
    name:str=Field("The name of the player")
    club:str=Field("The name of the player's club ")
    goals:int=Field("The number of goals scored")

class ScorerResponse(BaseModel):
    season:str=Field(description="The Season")
    scorer:list[Scorer]=Field(description="The list of scorers of the season")


output_parser = PydanticOutputParser(pydantic_object=ScorerResponse)
format_instructions = output_parser.get_format_instructions()
logger.debug(f"format_instruction: {format_instructions}")

prompt = PromptTemplate(
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions},
    template="""Answer the following question as best as possible
{question}

{format_instructions}""")

llm = OpenAI(api_key=config("OPENAI_API_KEY"))

chain = prompt|llm|output_parser

logger.info(chain.invoke({'question': 'List the 5 top scorers, their clubs and the goals scored in the Champions League season 2020.'}))