import transformers
import torch
from langchain.chains import LLMChain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from loguru import logger

logger.info('Import successful')

model = "google/flan-t5-small"
tokenizer = transformers.AutoTokenizer.from_pretrained(model)

logger.info("Loading Tokenizer successful")

pipeline = transformers.pipeline(
    #task="text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

logger.info("Pipeline is prepared")

llm = HuggingFacePipeline(pipeline=pipeline)
prompt = PromptTemplate.from_template('Tell me about {something} in a 100 words.')
chain = LLMChain(llm=llm,
                 prompt=prompt
                 )
logger.info(chain.invoke({'something': 'bees'}))