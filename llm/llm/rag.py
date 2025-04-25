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
    question: str
    question_chanks: list[str]
    context: list[Document]
    answer: str


def splitter_node(text_splitter: TextSplitter):
    def split(state: State):
        return {"question_chanks": text_splitter.split_text(state["question"])}
    return split


def retrieve_node(vector_store: VectorStore):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )

    def retrieve(state: State):
        print("Retrieving relevant documents...")
        retrieved_docs_nested = retriever.batch(
            state["question_chanks"],
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

        processed_docs = []
        for source_path, doc in unique_docs_by_path.items():
            if source_path and source_path.endswith(".txt"):
                sol_path = source_path[:-4] + ".sol"
                if os.path.exists(sol_path):
                    try:
                        with open(sol_path, 'r') as f:
                            solidity_content = f.read()
                        combined_content = f"{doc.page_content}\n\n--- Solidity Template ---\n\n{solidity_content}"
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


def generate_node(prompt: PromptTemplate, model: BaseLLM):
    def generate(state: State):
        docs_content = "\n\n".join(
            doc.page_content for doc in state["context"])
        messages = prompt.invoke(
            {"question": state["question"], "context": docs_content})
        response = model.invoke(messages)
        return {"answer": response}
    return generate


def create_chain(vectorstore: VectorStore, model: BaseLLM, text_splitter: TextSplitter):
    print("Setting up RAG chain...")

    template = """
        You are an AI assistant specialized in generating Solidity smart contracts.
        Use the following context, which consists of a text description from a legal agreement and its corresponding Solidity code template (separated by '--- Solidity Template ---'), to fulfill the user's request.
        Generate a complete Solidity smart contract based on the user's request and the provided context.
        Ensure the contract follows Solidity best practices. If the provided context is insufficient or irrelevant to the request, state that you cannot generate the contract based on the given information.

        Context:
        {context}

        User Request:
        {question}

        Generated Solidity Contract:
        """

    prompt = PromptTemplate.from_template(template)

    retriever = retrieve_node(vectorstore)
    generator = generate_node(prompt, model)
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
