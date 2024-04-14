from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from decouple import config
from loguru import logger


prompt = PromptTemplate.from_template(
    "Preprocess the given text by following the given steps in sequence. Follow only those steps that have a yes against them. Remove Number:{number},Remove punctuations:{punc},Word stemming:{stem} . Output just the preprocessed text. Text:{text}"
)

llm = OpenAI(openai_api_key=config("OPENAI_API_KEY"))
chain = LLMChain(llm=llm, prompt=prompt)

logger.info(chain.invoke({'text':'Hey!! I got 12 out of 20 in Swimming', 'number': 'yes', 'punc':'yes', 'stem':'no'})) 
logger.info(chain.invoke({'text':'22 13B is my flat no. Rohit will be joining us for the party', 'number':'yes', 'punc':'no', 'stem':'yes'}))