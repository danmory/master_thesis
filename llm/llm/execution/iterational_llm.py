import os
import re
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import TextSplitter
from langgraph.graph import StateGraph, START, END

from .base_llm import (
    BaseState,
    LLMChain,
    invoke_with_retry,
    test_node,
    validation_node
)


DEFAULT_PROMPT = """
    You are a smart contract developer converting legal rental agreements to Solidity smart contracts.
    
    # Current Smart Contract State:
    {generated_code}
    
    # Agreement Chunk to Add into Smart Contract:
    {current_chunk}
    
    Instructions:
    1. As a lawyer, analyze agreement chunk and think what should be included into smart contract.
    2. As a lawyer, analyze how payments and changes to the signed agreement should be handled in smart contract and what checks are required.
    3. As an ordinary human, analyze what names should be used for state variables and methods in order to be understandable.
    4. As a smart contract developer, add the necessary state variables to smart contract. Add methods only if required.
    5. As a smart contract developer, make sure the contract is syntactically correct and aligns with the rules below:
    
    Rules:
    1. ONLY ADD new state variables and methods
    2. NEVER remove existing code!!!
    3. Initialize variables with exact values from the agreement chunk (dates, amounts, etc)
    4. Create methods for payments and other actions stated in the agreement considering:
       - require() checks for amounts/dates
       - access controls
       - corresponding events
    5. Skip creating methods to modify variables unless explicitly required
    6. Implement only what's in current chunk - ignore unrelated terms
    7. Keep all state variables at the top, and methods at the bottom
    8. Keep contract syntactically correct all the time.
    
    Output the complete 'Agreement' contract with your additions in plain Solidity code (no comments/explanations).
    """


class State(BaseState):
    agreement_chunks: list[str]
    current_chunk_index: int


def splitter_node(text_splitter: TextSplitter):
    def split(state: State):
        chunks = text_splitter.split_text(state["agreement_to_convert"])
        return {
            "agreement_chunks": chunks,
            "current_chunk_index": 0,
            "generated_code": "pragma solidity ^0.8.0;\n\ncontract Agreement {\n\n}",
            "done": False
        }
    return split


def router_node():
    def route(state: State):
        if state["current_chunk_index"] >= len(state["agreement_chunks"]) - 1:
            return {"done": True, "answer": state["generated_code"]}

        next_index = state["current_chunk_index"] + 1
        return {
            "current_chunk_index": next_index,
        }

    return route


def generator_node(prompt: PromptTemplate, model: BaseLanguageModel):
    def generate(state: State):
        if state["done"]:
            return {}

        formatted_prompt = prompt.format(
            current_chunk=state["agreement_chunks"][state["current_chunk_index"]],
            generated_code=state["generated_code"]
        )

        print(
            f"Generating code for chunk {state['current_chunk_index']+1}/{len(state['agreement_chunks'])}")
        response = invoke_with_retry(model, formatted_prompt)

        content = response.content if hasattr(
            response, 'content') else response
        match = re.search(r"```solidity\n(.*?)\n```", content, re.DOTALL)
        if match:
            extracted_code = match.group(1).strip()
            return {"generated_code": extracted_code}
        else:
            return {"generated_code": content}

    return generate


def logger_node(log_dir: str = "./logs"):
    def log(state: State):
        os.makedirs(log_dir, exist_ok=True)

        if state["done"]:
            # Log final state
            log_path = os.path.join(log_dir, "final.log")
            log_content = f"Final Generated Code:\n{state['generated_code']}\n"
        else:
            # Log intermediate state
            log_path = os.path.join(
                log_dir, f"step_{state['current_chunk_index'] + 1}.log")
            log_content = (
                f"Processing Step {state['current_chunk_index'] + 1}\n"
                f"{'=' * 50}\n\n"
                f"Current Agreement Chunk:\n{state['agreement_chunks'][state['current_chunk_index']]}\n\n"
                f"{'=' * 50}\n\n"
                f"Generated Solidity Code:\n{state['generated_code']}\n"
            )

        with open(log_path, "w") as f:
            f.write(log_content)

        return state

    return log


class IterationalLLMChain(LLMChain):
    def create_chain(self, model: BaseLanguageModel, text_splitter: TextSplitter, template: str = DEFAULT_PROMPT):
        print("Setting up iterative chain...")

        prompt = PromptTemplate.from_template(template)
        input_splitter = splitter_node(text_splitter)
        generator = generator_node(prompt, model)
        router = router_node()
        logger = logger_node()
        tester = test_node(model)
        validator = validation_node(model)

        graph_builder = StateGraph(State)

        graph_builder.add_node("input_splitter", input_splitter)
        graph_builder.add_node("generator", generator)
        graph_builder.add_node("router", router)
        graph_builder.add_node("logger", logger)
        graph_builder.add_node("tester", tester)
        graph_builder.add_node("validator", validator)

        graph_builder.add_edge(START, "input_splitter")
        graph_builder.add_edge("input_splitter", "generator")
        graph_builder.add_edge("generator", "router")
        graph_builder.add_edge("router", "validator")
        graph_builder.add_edge("validator", "logger")
        graph_builder.add_edge("logger", "tester")

        graph_builder.add_conditional_edges(
            "tester",
            lambda state: END if state["done"] else "generator"
        )

        print("Iterative chain setup complete.")
        return graph_builder.compile()
