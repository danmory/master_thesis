from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

class RAGPipeline:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vector_store = None
        self.contract_templates = {}
    
    def process_document(self, document_path: str) -> List[Dict[str, Any]]:
        """Process a legal document and split it into sections using LangChain."""
        loader = TextLoader(document_path)
        documents = loader.load()
        texts = self.text_splitter.split_documents(documents)
        
        processed_sections = []
        for doc in texts:
            processed_sections.append({
                'text': doc.page_content,
                'metadata': doc.metadata
            })
        return processed_sections
    
    def find_similar_contract_parts(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find most similar contract parts using FAISS vector store."""
        if not self.vector_store:
            return []
            
        docs_and_scores = self.vector_store.similarity_search_with_score(query_text, k=top_k)
        return [{
            'content': doc.page_content,
            'metadata': doc.metadata,
            'score': score
        } for doc, score in docs_and_scores]
    
    def load_contract_templates(self, templates_dir: str):
        """Load and embed contract templates using LangChain."""
        template_path = Path(templates_dir)
        documents = []
        
        for template_file in template_path.glob('*.sol'):
            loader = TextLoader(str(template_file))
            documents.extend(loader.load())
            
        texts = self.text_splitter.split_documents(documents)
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
    
    def assemble_contract(self, document_sections: List[Dict[str, Any]]) -> str:
        """Assemble final smart contract from matched sections using LangChain."""
        contract_parts = {}
        contract_variables = {}
        
        # Extract key information from sections
        for section in document_sections:
            section_text = section['text'].lower()
            
            # Extract key variables
            if 'rent' in section_text or 'payment' in section_text:
                contract_variables['rentAmount'] = self._extract_amount(section_text)
                contract_variables['paymentFrequency'] = 30  # Default to monthly
            
            if 'property' in section_text or 'premises' in section_text:
                contract_variables['propertyAddress'] = self._extract_address(section_text)
            
            if 'term' in section_text or 'duration' in section_text:
                contract_variables['agreementEndDate'] = self._extract_duration(section_text)
            
            # Find similar contract parts
            similar_parts = self.find_similar_contract_parts(section['text'])
            if similar_parts:
                section_type = self._get_section_type(section['text'])
                if section_type not in contract_parts:
                    contract_parts[section_type] = similar_parts[0]['content']
        
        # Assemble contract with proper structure
        contract = "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.0;\n\n"
        contract += "contract RentalAgreement {\n"
        
        # Add state variables
        contract += "    address public lessor;\n"
        contract += "    address public lessee;\n"
        contract += f"    string public propertyAddress = \"{contract_variables.get('propertyAddress', '')}\";"
        contract += f"    uint256 public rentAmount = {contract_variables.get('rentAmount', '0')};"
        contract += f"    uint256 public paymentFrequency = {contract_variables.get('paymentFrequency', '30')};"
        contract += f"    uint256 public agreementEndDate = {contract_variables.get('agreementEndDate', '0')};"
        contract += "    mapping(uint => bool) payments;\n\n"
        
        # Add events
        contract += "    event RentPaid(address indexed sender, uint256 amount);\n"
        contract += "    event AgreementTerminated(address indexed sender);\n"
        contract += "    event PaymentVerified(address indexed sender, uint256 paymentNumber);\n\n"
        
        # Add constructor and functions
        for part in contract_parts.values():
            if 'constructor' in part or 'function' in part:
                contract += part.strip() + "\n\n"
        
        contract += "}\n"
        return contract

    def _extract_amount(self, text: str) -> str:
        """Extract rental amount from text."""
        import re
        amounts = re.findall(r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)', text)
        return amounts[0].replace(',', '') if amounts else '0'
    
    def _extract_address(self, text: str) -> str:
        """Extract property address from text."""
        import re
        # Simple address extraction - can be improved
        address_match = re.search(r'(?:located at|address[:]?\s+)([^.\n]+)', text)
        return address_match.group(1).strip() if address_match else ''
    
    def _extract_duration(self, text: str) -> str:
        """Extract agreement duration in timestamp."""
        import re
        from datetime import datetime, timedelta
        
        # Try to find duration in months or years
        duration_match = re.search(r'(\d+)\s*(month|year)s?', text)
        if duration_match:
            number = int(duration_match.group(1))
            unit = duration_match.group(2)
            days = number * 365 if unit == 'year' else number * 30
            future_date = datetime.now() + timedelta(days=days)
            return str(int(future_date.timestamp()))
        return '0'
    
    def detect_document_type(self, document_sections: List[Dict[str, Any]]) -> str:
        """Detect the type of legal document based on its content."""
        keywords = {
            'rental': ['rent', 'lease', 'tenant', 'landlord', 'property', 'premises'],
            'sale': ['purchase', 'buyer', 'seller', 'sale', 'price', 'property'],
            'service': ['service', 'provider', 'client', 'scope', 'deliverables'],
            'employment': ['employer', 'employee', 'salary', 'work', 'employment'],
            'license': ['license', 'licensor', 'licensee', 'intellectual property', 'rights']
        }
        
        counts = {doc_type: 0 for doc_type in keywords}
        for section in document_sections:
            text = (section['text'] + section.get('title', '')).lower()
            for doc_type, kwords in keywords.items():
                counts[doc_type] += sum(1 for kw in kwords if kw in text)
        
        return max(counts.items(), key=lambda x: x[1])[0]
    
    def _get_section_type(self, text: str, doc_type: str = 'rental') -> str:
        """Identify the type of contract section based on document type."""
        text = text.lower()
        section_types = {
            'rental': {
                'payment': ['payment', 'rent', 'fee'],
                'term': ['term', 'duration', 'period'],
                'property': ['property', 'premises', 'location'],
                'termination': ['termination', 'end', 'cancel']
            },
            'sale': {
                'payment': ['payment', 'price', 'consideration'],
                'item': ['item', 'property', 'asset', 'goods'],
                'transfer': ['transfer', 'delivery', 'possession'],
                'termination': ['termination', 'cancellation', 'rescission']
            },
            'service': {
                'payment': ['payment', 'fee', 'compensation'],
                'scope': ['scope', 'services', 'deliverables'],
                'term': ['term', 'duration', 'period'],
                'termination': ['termination', 'cancellation']
            }
        }
        
        if doc_type not in section_types:
            doc_type = 'rental'  # Default to rental if type not recognized
            
        for section_type, keywords in section_types[doc_type].items():
            if any(word in text for word in keywords):
                return section_type
        return 'other'