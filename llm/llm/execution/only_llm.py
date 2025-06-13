from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END


from .base_llm import (
    BaseState,
    LLMChain,
    generator_node,
    logger_node,
    test_node,
    validation_node
)


DEFAULT_PROMPT = """
    You are a smart contract developer converting legal rental agreements to Solidity smart contracts.
    
    # Agreement to Convert:
    {agreement}
    
    Instructions:
    1. As a lawyer, analyze agreement and think what should be included into smart contract.
    2. As a lawyer, analyze how payments and changes to the signed agreement should be handled in smart contract and what checks are required.
    3. As an ordinary human, analyze what names should be used for state variables and methods in order to be understandable.
    4. As a smart contract developer, create a complete smart contract with all necessary state variables and methods.
    5. As a smart contract developer, make sure the contract is syntactically correct and aligns with the rules below:
    
    Rules:
    1. Initialize variables with exact values from the agreement (dates, amounts, etc)
    2. Create methods for payments and other actions stated in the agreement considering:
       - require() checks for amounts/dates
       - access controls
       - corresponding events
    3. Skip creating methods to modify variables unless explicitly required
    4. Keep all state variables at the top, and methods at the bottom
    5. Keep contract syntactically correct all the time.
    
    Output the complete 'Agreement' contract in plain Solidity code (no comments/explanations).
    """


class State(BaseState):
    pass


class OnlyLLMChain(LLMChain):
    def create_chain(self, model: BaseLanguageModel, template: str = DEFAULT_PROMPT):
        print("Setting up single-call chain...")

        prompt = PromptTemplate.from_template(template)
        generator = generator_node(prompt, model)
        logger = logger_node()
        tester = test_node(model)
        validator = validation_node(model)

        graph_builder = StateGraph(State)

        graph_builder.add_node("generator", generator)
        graph_builder.add_node("validator", validator)
        graph_builder.add_node("logger", logger)
        graph_builder.add_node("tester", tester)

        graph_builder.add_edge(START, "generator")
        graph_builder.add_edge("generator", "validator")
        graph_builder.add_edge("validator", "logger")
        graph_builder.add_edge("logger", "tester")
        graph_builder.add_edge("tester", END)

        print("Single-call chain setup complete.")
        return graph_builder.compile()
