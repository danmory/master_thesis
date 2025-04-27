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
import re


def get_default_prompt_template():
    return """
    You are a smart contract developer tasked with converting legal agreements into Solidity code.
    
    # Current Agreement Chunk:
    {current_chunk}
    
    # Relevant Templates and Their Solidity implementation:
    {context}
    
    # Previously Generated Code (modify only if needed):
    {generated_code}
    
    Instructions:
    1. Analyze the current agreement chunk and identify any contractual terms that should be implemented in Solidity.
    2. Use the provided templates and their implementation as reference for implementation patterns.
    3. If Previously Generated Code is empty, create a new Solidity contract with name 'Agreement' and appropriate structure.
    4. If code has already been generated, ONLY add or modify code of the 'Agreement' contract to implement the current chunk's requirements.
    5. Set state variables to values that you can find on the agreement, e.g. start dates, rent amount, etc.
    6. If the current chunk doesn't require any changes to the existing code, return the existing code unchanged.
    7. Ensure the contract remains syntactically valid and coherent at all times.
    8. Focus on implementing the specific terms from the current chunk only.
    
    Output ONLY the SINGLE complete Solidity contract code with your modifications.
    """


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
    agreement_chunks: list[str]
    current_chunk_index: int
    context: list[Document]
    generated_code: str
    answer: str
    done: bool


def splitter_node(text_splitter: TextSplitter):
    def split(state: State):
        chunks = text_splitter.split_text(state["agreement_to_convert"])
        return {
            "agreement_chunks": chunks,
            "current_chunk_index": 0,
            "generated_code": "",
            "done": False
        }
    return split


def retriever_node(vector_store: VectorStore):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    def retrieve(state: State):
        if state["done"]:
            return {}

        print(
            f"Processing chunk {state['current_chunk_index']+1}/{len(state['agreement_chunks'])}")

        retrieved_docs = retriever.invoke(
            state["agreement_chunks"][state["current_chunk_index"]])

        processed_docs: list[Document] = []
        for index, doc in enumerate(retrieved_docs):
            source_path = doc.metadata.get('source')
            if source_path and source_path.endswith(".txt"):
                sol_path = source_path[:-4] + ".sol"
                if os.path.exists(sol_path):
                    try:
                        with open(sol_path, 'r') as f:
                            solidity_content = f.read()
                        combined_content = f"\n\n --- Agreement part {index} ---\n\n {doc.page_content}\n\n--- Solidity Template for {index} ---\n\n{solidity_content}"
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
        if state["done"]:
            return {}

        docs_content = "\n\n".join(
            doc.page_content for doc in state["context"])

        formatted_prompt = prompt.format(
            current_chunk=state["agreement_chunks"][state["current_chunk_index"]],
            context=docs_content,
            generated_code=state["generated_code"]
        )

        print(
            f"Generating code for chunk {state['current_chunk_index']+1}/{len(state['agreement_chunks'])}")
        response = model.invoke(formatted_prompt)

        # Extract Solidity code block
        match = re.search(r"```solidity\n(.*?)\n```", response, re.DOTALL)
        if match:
            extracted_code = match.group(1).strip()
            print(f"Extracted Solidity code for chunk {state['current_chunk_index']+1}/{len(state['agreement_chunks'])}: \n\n {extracted_code}")
            return {"generated_code": extracted_code}
        else:
            print(f"Could not extract Solidity code from response for chunk {state['current_chunk_index']+1}/{len(state['agreement_chunks'])}. Using full response.")
            print(f"Full response: \n\n {response}")
            # Fallback to using the full response if extraction fails, or handle error as needed
            return {"generated_code": response}

    return generate


def router_node():
    def route(state: State):
        # Check if we've processed all chunks
        if state["current_chunk_index"] >= len(state["agreement_chunks"]) - 1:
            return {"done": True, "answer": state["generated_code"]}

        # Move to the next chunk
        next_index = state["current_chunk_index"] + 1
        return {
            "current_chunk_index": next_index,
        }

    return route


def create_chain(vectorstore: VectorStore, model: BaseLLM, text_splitter: TextSplitter, template: str = get_default_prompt_template()):
    print("Setting up iterative RAG chain...")

    prompt = PromptTemplate.from_template(template)

    # Define nodes
    input_splitter = splitter_node(text_splitter)
    retriever = retriever_node(vectorstore)
    generator = generator_node(prompt, model)
    router = router_node()

    # Build the graph
    graph_builder = StateGraph(State)

    graph_builder.add_node("input_splitter", input_splitter)
    graph_builder.add_node("retriever", retriever)
    graph_builder.add_node("generator", generator)
    graph_builder.add_node("router", router)

    # Define the workflow
    graph_builder.add_edge(START, "input_splitter")
    graph_builder.add_edge("input_splitter", "retriever")
    graph_builder.add_edge("retriever", "generator")
    graph_builder.add_edge("generator", "router")

    # Conditional edges
    graph_builder.add_conditional_edges(
        "router",
        lambda state: END if state["done"] else "retriever"
    )

    print("RAG chain setup complete.")

    return graph_builder.compile()
