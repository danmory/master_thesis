import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

import constants
import execution.rag as rag
import execution.iterational_llm as iterational_llm
import execution.only_llm as only_llm

API_KEY = ""

script_dir = os.path.dirname(os.path.abspath(__file__))


agreement_path = os.path.abspath(os.path.join(
    script_dir, '..', 'docs', 'short-term-vacation-lease-agreements', 'agreement_1.txt'))
model_path_str = os.path.abspath(os.path.join(
    script_dir, '..', 'models', constants.LLM_NAME))
templates_path = os.path.abspath(os.path.join(script_dir, '..', 'templates'))

with open(agreement_path, 'r') as file:
    rental_agreement = file.read()

# model = LlamaCpp(
#     model_path=model_path_str,
#     n_gpu_layers=-1,
#     n_ctx=8192,
#     max_tokens=2500,
# )

model = ChatOpenAI(
    api_key=SecretStr(API_KEY),
    base_url="https://api.proxyapi.ru/deepseek",
    temperature=0.8,
    model="deepseek-chat",
)


chain_type = constants.ChainType.ITERATIONAL

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n\d+\.", "\n\d+\."],
    is_separator_regex=True,
)
print(f"Creating {chain_type} chain...")
if chain_type == constants.ChainType.RAG:
    templates = rag.load_templates(templates_path)
    embeddings = SentenceTransformerEmbeddings(
    model_name=constants.EMBEDDINGS_MODEL_NAME, model_kwargs={'device': 'cuda'})
    vector_store = rag.create_vector_store(embeddings, templates)
    chain = rag.RAGChain().create_chain(model, vector_store, text_splitter)
elif chain_type == constants.ChainType.ITERATIONAL:
    chain = iterational_llm.IterationalLLMChain().create_chain(model, text_splitter)
elif chain_type == constants.ChainType.ONLY:
    chain = only_llm.OnlyLLMChain().create_chain(model)
else:
    raise ValueError(f"Unknown chain type: {chain_type}")

print(f"Invoking {chain_type} chain...")
response = chain.invoke({"agreement_to_convert": rental_agreement}, {
                        "recursion_limit": 100})
print("Response is generated")

print(response["answer"])
