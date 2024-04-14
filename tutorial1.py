from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from decouple import config
from loguru import logger

prompt = PromptTemplate.from_template("Suggest {number} names for a {domain} startup?")
llm = OpenAI(openai_api_key=config("OPENAI_API_KEY"))
chain = LLMChain(llm=llm,
                 prompt=prompt
                 )

logger.info(chain.invoke({
    'number': '5',
    'domain': 'cooking'
}))

logger.info(chain.invoke({
    'number': '2',
    'domain': 'AI'
}))
