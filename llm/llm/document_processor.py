from typing import List, Dict, Any
from pathlib import Path
import re

class DocumentProcessor:
    def __init__(self):
        self.section_patterns = [
            r'(?i)^\s*ARTICLE\s+[\d]+[.:)]\s*[\w\s]+',
            r'(?i)^\s*\d+[.:)]\s*[\w\s]+',
            r'(?i)^\s*[A-Z][.:)]\s*[\w\s]+',
            r'(?i)^\s*SECTION\s+[\d]+[.:)]\s*[\w\s]+'
        ]
    
    def extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extract sections from legal document text."""
        sections = []
        current_section = {'title': '', 'content': []}
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            is_section_header = any(re.match(pattern, line) for pattern in self.section_patterns)
            
            if is_section_header:
                if current_section['content']:
                    sections.append({
                        'title': current_section['title'],
                        'content': '\n'.join(current_section['content'])
                    })
                current_section = {'title': line, 'content': []}
            else:
                current_section['content'].append(line)
        
        if current_section['content']:
            sections.append({
                'title': current_section['title'],
                'content': '\n'.join(current_section['content'])
            })
        
        return sections
    
    def process_document(self, document_path: str) -> List[Dict[str, Any]]:
        """Process a legal document and return its sections."""
        doc_text = Path(document_path).read_text()
        return self.extract_sections(doc_text)
    
    def get_section_type(self, section: Dict[str, str]) -> str:
        """Identify the type of contract section based on its content."""
        title = section['title'].lower()
        content = section['content'].lower()
        
        section_types = {
            'payment': ['payment', 'rent', 'fee', 'deposit'],
            'term': ['term', 'duration', 'period'],
            'parties': ['parties', 'lessor', 'lessee', 'landlord', 'tenant'],
            'property': ['property', 'premises', 'location'],
            'termination': ['termination', 'end', 'cancel'],
            'maintenance': ['maintenance', 'repair', 'upkeep'],
            'rights': ['rights', 'obligations', 'responsibilities'],
            'default': ['default', 'breach', 'violation']
        }
        
        for type_name, keywords in section_types.items():
            if any(keyword in title or keyword in content for keyword in keywords):
                return type_name
        
        return 'other'