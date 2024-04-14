from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser

from operator import itemgetter
from decouple import config
from loguru import logger

prompt1 = ChatPromptTemplate.from_template(
    "Which country won the {game} world cup?")

prompt2 = ChatPromptTemplate.from_template(
    "Suggest the best {entity} from {country}"
)

llm = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"))

chain1 = prompt1 | llm | StrOutputParser()
chain = (
    {'country': chain1, 'entity': itemgetter('entity')} \
    | prompt2
    | llm
    | StrOutputParser()
)

logger.info('Result: ' + \
    chain.invoke({
        'game': 'soccer',
        'entity': 'wine'
    })
)