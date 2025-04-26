import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import TextSplitter
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


def load_templates(directory_path: str) -> list[Document]:
    print(f"Loading templates from: {directory_path}")
    loader = DirectoryLoader(
        directory_path, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
    documents = loader.load()
    print(f"Loaded {len(documents)} template documents.")
    return documents


def create_vector_store(embeddings: Embeddings, documents: list[Document]) -> InMemoryVectorStore:
    print("Creating vector store...")
    vectorstore = InMemoryVectorStore.from_documents(documents, embeddings)
    print("Vector store created successfully.")
    return vectorstore


class State(TypedDict):
    agreement_to_convert: str
    agreement_to_convert_chanks: list[str]
    context: list[Document]
    answer: str


def splitter_node(text_splitter: TextSplitter):
    def split(state: State):
        return {"agreement_to_convert_chanks": text_splitter.split_text(state["agreement_to_convert"])}
    return split


def retriever_node(vector_store: VectorStore):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    def retrieve(state: State):
        print("Retrieving relevant documents...")
        retrieved_docs_nested = retriever.batch(
            state["agreement_to_convert_chanks"],
        )
        # flatten
        all_docs = [doc for sublist in retrieved_docs_nested for doc in sublist]

        # remove duplicates
        unique_docs_by_path: dict[str, Document] = {}
        for doc in all_docs:
            source_path = doc.metadata.get('source')
            if source_path and source_path not in unique_docs_by_path:
                unique_docs_by_path[source_path] = doc

        print(
            f"Found {len(unique_docs_by_path)} unique documents based on source path.")

        processed_docs: list[Document] = []
        for index, (source_path, doc) in enumerate(unique_docs_by_path.items()):
            if source_path and source_path.endswith(".txt"):
                sol_path = source_path[:-4] + ".sol"
                if os.path.exists(sol_path):
                    try:
                        with open(sol_path, 'r') as f:
                            solidity_content = f.read()
                        combined_content = f"\n\n --- Agreement part №{index} ---\n\n {doc.page_content}\n\n--- Solidity Template for №{index} ---\n\n{solidity_content}"
                        print(
                            f"Successfully loaded and combined template for: {source_path}")
                        processed_docs.append(
                            Document(page_content=combined_content, metadata=doc.metadata))
                    except Exception as e:
                        print(f"Error reading Solidity file {sol_path}: {e}")
                else:
                    print(
                        f"No corresponding Solidity template found at: {sol_path}")

        print(f"Processed {len(processed_docs)} documents for context.")
        return {"context": processed_docs}

    return retrieve


def generator_node(prompt: PromptTemplate, model: BaseLLM):
    def generate(state: State):
        docs_content = "\n\n".join(
            doc.page_content for doc in state["context"])
        formatted_prompt = prompt.format(
            agreement_to_convert=state["agreement_to_convert"], context=docs_content
        )
        print("Prompt: ", formatted_prompt)
        response = model.invoke(formatted_prompt)
        return {"answer": response}
    return generate


def create_chain(vectorstore: VectorStore, model: BaseLLM, text_splitter: TextSplitter, template: str):
    print("Setting up RAG chain...")

    prompt = PromptTemplate.from_template(template)

    retriever = retriever_node(vectorstore)
    generator = generator_node(prompt, model)
    input_splitter = splitter_node(text_splitter)

    graph_builder = StateGraph(State)
    graph_builder.add_node("input_splitter", input_splitter)
    graph_builder.add_node("retriever", retriever)
    graph_builder.add_node("generator", generator)

    graph_builder.add_edge(START, "input_splitter")
    graph_builder.add_edge("input_splitter", "retriever")
    graph_builder.add_edge("retriever", "generator")
    graph_builder.add_edge("generator", END)

    print("RAG chain setup complete.")

    return graph_builder.compile()
