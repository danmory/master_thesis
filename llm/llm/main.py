from operator import is_
import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp

import rag

script_dir = os.path.dirname(os.path.abspath(__file__))

agreement_path = os.path.abspath(os.path.join(
    script_dir, '..', 'docs', 'short-term-vacation-lease-agreements', 'agreement_1.txt'))
model_path_str = os.path.abspath(os.path.join(
    script_dir, '..', 'models', 'google_gemma-3-27b-it-qat-Q5_K_M.gguf'))
templates_path = os.path.abspath(os.path.join(script_dir, '..', 'templates'))

with open(agreement_path, 'r') as file:
    rental_agreement = file.read()

model = LlamaCpp(
    model_path=model_path_str,
    n_gpu_layers=-1,
    # n_batch=16,
    n_ctx=8192,
    max_tokens=2500,
)

templates = rag.load_templates(templates_path)
embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cuda'})
vector_store = rag.create_vector_store(embeddings, templates)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n\d+\.", "\n\d+\."],
    is_separator_regex=True,
)
rag_chain = rag.create_chain(vector_store, model, text_splitter)

print("Invoking RAG chain...")
response = rag_chain.invoke({"agreement_to_convert": rental_agreement}, {
                            "recursion_limit": 100})
print("Response is generated")

print(response["answer"])
