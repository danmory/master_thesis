import os
import re
from typing import TypedDict, Any, Protocol
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from solcx import compile_source

from .test_generator import TestGenerator, TestConfig


class BaseState(TypedDict):
    agreement_to_convert: str
    generated_code: str
    answer: str
    done: bool
    test_results: str
    test_success: bool


class LLMChain(Protocol):
    def create_chain(self, model: BaseLanguageModel, **kwargs) -> Any:
        ...


def invoke_with_retry(model: BaseLanguageModel, prompt: str, max_retries: int = 3, retry_delay: float = 2.0, timeout: float = 180.0) -> Any:
    for attempt in range(max_retries):
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(model.invoke, prompt)
                return future.result(timeout=timeout)
        except TimeoutError:
            print(
                f"Model invocation timed out after {timeout} seconds (attempt {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                raise TimeoutError(
                    f"Model invocation timed out after {timeout} seconds after {max_retries} attempts")
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(
                f"Model invocation failed (attempt {attempt + 1}/{max_retries}): {str(e)}")

        print(f"Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)
        retry_delay *= 2


def generator_node(prompt: PromptTemplate, model: BaseLanguageModel):
    def generate(state: BaseState):
        formatted_prompt = prompt.format(
            agreement=state["agreement_to_convert"])
        response = invoke_with_retry(model, formatted_prompt)
        content = response.content if hasattr(
            response, 'content') else response
        match = re.search(r"```solidity\n(.*?)\n```", content, re.DOTALL)
        if match:
            extracted_code = match.group(1).strip()
            return {"generated_code": extracted_code, "answer": extracted_code, "done": True}
        else:
            return {"generated_code": content, "answer": content, "done": True}
    return generate


def logger_node(log_dir: str = "./logs"):
    def log(state: BaseState):
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "final.log")
        log_content = f"Generated Code:\n{state['generated_code']}\n"
        with open(log_path, "w") as f:
            f.write(log_content)
        return state
    return log


def test_node(model: BaseLanguageModel):
    def test(state: BaseState):
        if not state["done"] or not state["generated_code"]:
            return {}

        test_dir = Path(os.path.dirname(
            os.path.abspath(__file__))) / "temp_test"
        config = TestConfig(test_dir=test_dir)
        test_generator = TestGenerator(config)

        try:
            contract_path = test_generator.save_contract(
                state["generated_code"])
            print("Compiling contract...")
            contract_name_match = re.search(
                r"contract\s+(\w+)", state["generated_code"])
            if not contract_name_match:
                raise Exception("Could not find contract name in the code")
            contract_name = contract_name_match.group(1)

            temp_contract_path = os.path.join(test_dir, f"{contract_name}.sol")
            with open(temp_contract_path, "w") as f:
                f.write(state["generated_code"])

            try:
                compile_source(
                    state["generated_code"],
                    output_values=["abi", "bin"],
                    solc_version="0.8.20",
                    allow_paths=[test_dir]
                )
                print("Contract compiled successfully.")
            except Exception as e:
                error_msg = str(e)
                print(f"Compilation error: {error_msg}")
                fix_prompt = f"""
                    The generated Solidity contract failed to compile with these errors:
                    {error_msg}

                    Please fix the following contract:
                    {state["generated_code"]}

                    Instructions:
                    1. Collect all errors and analyze them(they start with Error: <description>).
                    2. For each error identify what part of the contract causes the error.
                    3. Fix the identified part of the contract.
                    4. Pay attention to duplicated constructors, methods and state variables. Initialize non-existing variables. Add visibility identifiers if needed. Use only Solidity types and syntax.
                    5. Make sure the contract is syntactically correct.
                    6. Repeat steps for all errors.
                
                    Output ONLY the fixed Solidity contract without any comments or explanations.
                """
                fixed_code = invoke_with_retry(model, fix_prompt)
                content = fixed_code.content if hasattr(
                    fixed_code, 'content') else fixed_code
                match = re.search(r"```solidity\n(.*?)\n```",
                                  content, re.DOTALL)
                if match:
                    fixed_code = match.group(1).strip()
                    test_generator.save_contract(fixed_code)
                    state["generated_code"] = fixed_code
                else:
                    state["generated_code"] = content

            test_prompt = test_generator.generate_test_prompt(
                state["generated_code"],
                state["agreement_to_convert"]
            )
            print("Invoking test prompt...")
            test_code = invoke_with_retry(model, test_prompt)

            content = test_code.content if hasattr(
                test_code, 'content') else test_code
            test_path = test_generator.save_test_file(content)

            if not test_generator.wait_for_human_review(contract_path, test_path):
                return {
                    "test_results": "Testing cancelled by user",
                    "test_success": False
                }

            result = test_generator.run_tests()
            print(
                f"Test success: {result['success']}, output: {result['output']}")
            return {
                "test_results": result["output"],
                "test_success": result["success"]
            }

        finally:
            ...

    return test


def validation_node(model: BaseLanguageModel):
    def validate(state: BaseState):
        if not state["done"] or not state["generated_code"]:
            return {}

        validation_prompt = f"""
        Analyze the following smart contract for correctness and potential issues:
        
        Contract Code:
        {state["generated_code"]}
        
        Original Agreement:
        {state["agreement_to_convert"]}
        
        Perform the following validations:
        1. Semantic validation - check if all terms are properly translated
        2. Security pattern validation
        3. Gas optimization analysis
        4. State variable validation
        5. Function validation
        6. Documentation validation
        7. Cross-reference validation
        8. Business logic validation
        9. Integration testing considerations
        
        For each validation category, provide:
        - Pass/Fail status
        - Detailed explanation of any issues found
        - Recommendations for improvement
        
        Output the validation results in a structured format.
        """

        print("Invoking validation prompt...")
        validation_results = invoke_with_retry(model, validation_prompt)
        content = validation_results.content if hasattr(
            validation_results, 'content') else validation_results
        print(f"Validation results: {content}")

        fix_prompt = f"""
        Auditors found issues in the contract:
        {content}
        
        Please fix the issues in the contract:
        {state["generated_code"]}

        Instructions:
        1. Analyze the issues and fix them.
        2. Pay attention to duplicated constructors, methods and state variables. Initialize non-existing variables. Add visibility identifiers if needed. Use only Solidity types and syntax.
        3. Make sure the contract is syntactically correct.
        4. Repeat steps for all issues.
        5. Output ONLY the fixed Solidity contract without any comments or explanations.
        """
        print("Fixing contract...")
        fixed_code = invoke_with_retry(model, fix_prompt)
        content = fixed_code.content if hasattr(
            fixed_code, 'content') else fixed_code
        print(f"Fixed code: {content}")
        match = re.search(r"```solidity\n(.*?)\n```", content, re.DOTALL)
        if match:
            fixed_code = match.group(1).strip()
            return {"generated_code": fixed_code}
        else:
            return {"generated_code": content}

    return validate
