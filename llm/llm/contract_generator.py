from typing import List, Dict, Any
from pathlib import Path
from ctransformers import AutoModelForCausalLM
from .document_processor import DocumentProcessor
from .rag_pipeline import RAGPipeline

class ContractGenerator:
    def __init__(self, llm_model_path: str, templates_dir: str):
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_path,
            model_type="llama",
            gpu_layers=50,
            context_length=4096
        )
        self.doc_processor = DocumentProcessor()
        self.rag_pipeline = RAGPipeline()
        self.rag_pipeline.load_contract_templates(templates_dir)
        
    def _prepare_prompt(self, section: Dict[str, str], similar_contracts: List[Dict[str, Any]]) -> str:
        prompt = f"""Based on the following legal document section and similar smart contract parts,
        generate a Solidity smart contract section that implements the requirements:

        Document Section:
        {section['title']}
        {section['content']}

        Similar Contract Parts:
        """
        
        for contract in similar_contracts:
            prompt += f"\n{contract['content']}"
            
        prompt += "\n\nGenerate the appropriate Solidity code section:"
        return prompt
    
    def generate_contract(self, document_path: str) -> str:
        # Process the document into sections
        sections = self.doc_processor.process_document(document_path)
        
        # Detect document type
        doc_type = self.rag_pipeline.detect_document_type(sections)
        
        # Generate embeddings and find similar contract parts for each section
        contract_sections = []
        for section in sections:
            # Get section type based on document type
            section_type = self.doc_processor.get_section_type(section)
            
            # Find similar contract parts using the section content with document type context
            similar_parts = self.rag_pipeline.find_similar_contract_parts(f"{doc_type} {section['content']}")
            
            # Generate contract section using LLM with document type context
            prompt = f"Based on the following {doc_type} agreement section and similar smart contract parts,"
            prompt += "generate a Solidity smart contract section that implements the requirements:\n\n"
            prompt += f"Document Section ({doc_type} agreement):\n"
            prompt += f"{section['title']}\n{section['content']}\n\n"
            prompt += "Similar Contract Parts:\n"
            
            for contract in similar_parts:
                prompt += f"\n{contract['content']}"
                
            prompt += "\n\nGenerate the appropriate Solidity code section:"
            
            generated_section = self.llm(prompt, temperature=0.2, top_p=0.95, max_new_tokens=1000)
            
            contract_sections.append({
                'type': section_type,
                'content': generated_section
            })
        
        # Assemble the final contract
        contract = "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.0;\n\n"
        contract += f"contract {doc_type.capitalize()}Agreement {{\n"
        
        # Add state variables
        for section in contract_sections:
            if 'state variables' in section['content'].lower():
                contract += section['content'] + "\n"
        
        # Add events
        for section in contract_sections:
            if 'event' in section['content'].lower():
                contract += section['content'] + "\n"
        
        # Add constructor and functions
        for section in contract_sections:
            if 'function' in section['content'].lower() or 'constructor' in section['content'].lower():
                contract += section['content'] + "\n"
        
        contract += "}\n"
        return contract