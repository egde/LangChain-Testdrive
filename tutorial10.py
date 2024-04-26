# Tutorial 10
# Build a RAG App

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.document_loaders.text import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.callbacks import StdOutCallbackHandler
from langchain_core.documents import Document
from decouple import config
from loguru import logger

handler = StdOutCallbackHandler()

def load_data(filename:str) -> list[Document]:
    # Text Loader
    loader = TextLoader(filename)
    data = loader.load()
    return data

data = load_data('README.md') + load_data('tutorial10.py')

# Document Transformer
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(data)

# Generate the Text Embedding
embeddings = OpenAIEmbeddings(openai_api_type=config("OPENAI_API_KEY"))
llm = OpenAI(api_key=config("OPENAI_API_KEY"))

# Vector DB using Chroma
doc_search = Chroma.from_documents(documents=texts, embedding=embeddings)
logger.debug('Documents prepared')

# Setting up the Retriever
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=doc_search.as_retriever()
)

logger.info(qa.run("Explain what the file tutorial10.py is about", callbacks=[handler]))
