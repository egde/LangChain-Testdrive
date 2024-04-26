from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.callbacks import StdOutCallbackHandler
from decouple import config
from loguru import logger

template="""You are a chatbot having a conversation with a human.
{chat_history}
Human: {human}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=['chat_history', 'human'],
    template=template
)

# Adding Memory
memory=ConversationBufferMemory(memory_key="chat_history")
llm = OpenAI(api_key=config("OPENAI_API_KEY"))

#Adding a handler for debugging purposes
handler = StdOutCallbackHandler()

chain = LLMChain(
    prompt=prompt,
    llm=llm,
    memory=memory
)

logger.info(chain.run(human='Hi', callbacks=[handler]))

logger.info(chain.invoke({'human': 'Where can I see the latest Champions League games?'}))

logger.info(chain.invoke({'human': 'Can I check it on the radio?'}))