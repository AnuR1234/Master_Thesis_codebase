#!/usr/bin/env python3
"""
Enhanced ABAP Metrics - Factual Accuracy and Template Compliance
"""

import re
import logging
from typing import Dict, List, Set

class EnhancedABAPMetrics:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Define expected template sections for different ABAP types
        self.template_sections = {
            'class': [
                'overview', 'class definition', 'implementation overview',
                'method dependencies', 'database tables used', 'method implementation details'
            ],
            'test_class': [
                'overview', 'class definition', 'test methods', 'test data',
                'test implementation overview', 'method implementation details'
            ],
            'report': [
                'overview', 'program flow', 'data structures', 'selection screen',
                'main processing logic', 'database interaction'
            ],
            'function': [
                'overview', 'function module definition', 'interface',
                'implementation overview', 'processing logic', 'database interaction'
            ]
        }
        
        # ABAP keywords for validation
        self.valid_abap_keywords = {
            'CLASS', 'METHOD', 'FUNCTION', 'SELECT', 'INSERT', 'UPDATE', 'DELETE',
            'DATA', 'TYPES', 'CONSTANTS', 'PARAMETERS', 'IMPORTING', 'EXPORTING',
            'CHANGING', 'TABLES', 'EXCEPTIONS', 'LOOP', 'IF', 'ENDIF', 'ENDLOOP'
        }
        
        self.logger.info("✅ Enhanced ABAP metrics initialized")

    def calculate_enhanced_sample_metrics(self, sample: Dict) -> Dict[str, float]:
        """Calculate enhanced metrics for one sample."""
        abap_code = sample.get('abap_code', '')
        generated_doc = sample['generated_documentation']
        file_type = sample.get('file_type', 'class')
        
        all_metrics = {}
        
        # Calculate factual accuracy metrics
        all_metrics.update(self.calculate_factual_accuracy(abap_code, generated_doc))
        
        # Calculate template compliance metrics
        all_metrics.update(self.calculate_template_compliance(generated_doc, file_type))
        
        return all_metrics

    def calculate_factual_accuracy(self, abap_code: str, generated_doc: str) -> Dict[str, float]:
        """Calculate factual accuracy metrics."""
        try:
            # Extract facts from code and claims from documentation
            code_facts = self.extract_code_facts(abap_code)
            doc_claims = self.extract_documentation_claims(generated_doc)
            
            accuracy_metrics = {}
            
            # Method accuracy
            if code_facts['methods']:
                documented_methods = len(code_facts['methods'].intersection(doc_claims['methods']))
                accuracy_metrics['method_accuracy'] = documented_methods / len(code_facts['methods'])
                
                # Hallucination rate for methods
                hallucinated_methods = doc_claims['methods'] - code_facts['methods']
                if doc_claims['methods']:
                    accuracy_metrics['method_hallucination_rate'] = len(hallucinated_methods) / len(doc_claims['methods'])
                else:
                    accuracy_metrics['method_hallucination_rate'] = 0.0
            else:
                accuracy_metrics['method_accuracy'] = 1.0
                accuracy_metrics['method_hallucination_rate'] = 0.0
            
            # ABAP construct accuracy
            accuracy_metrics['abap_construct_accuracy'] = self.validate_abap_constructs(generated_doc)
            
            # Overall factual accuracy
            accuracy_values = [v for k, v in accuracy_metrics.items() if not k.endswith('_rate')]
            accuracy_metrics['overall_factual_accuracy'] = sum(accuracy_values) / len(accuracy_values)
            
            return accuracy_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating factual accuracy: {e}")
            return {
                'method_accuracy': 0.0,
                'abap_construct_accuracy': 0.0,
                'overall_factual_accuracy': 0.0,
                'method_hallucination_rate': 1.0
            }

    def calculate_template_compliance(self, generated_doc: str, file_type: str = 'class') -> Dict[str, float]:
        """Calculate template compliance metrics."""
        try:
            expected_sections = self.template_sections.get(file_type, self.template_sections['class'])
            doc_sections = self.extract_documentation_sections(generated_doc)
            
            compliance_metrics = {}
            
            # Section presence
            present_sections = 0
            for expected_section in expected_sections:
                if self.is_section_present(expected_section, doc_sections):
                    present_sections += 1
            
            compliance_metrics['section_coverage'] = present_sections / len(expected_sections)
            
            # Section completeness (not just headers, but content)
            complete_sections = 0
            for expected_section in expected_sections:
                if self.is_section_complete(expected_section, doc_sections):
                    complete_sections += 1
            
            compliance_metrics['section_completeness'] = complete_sections / len(expected_sections)
            
            # Markdown format compliance
            compliance_metrics['markdown_compliance'] = self.validate_markdown_format(generated_doc)
            
            # Table structure compliance
            compliance_metrics['table_structure_compliance'] = self.validate_table_structures(generated_doc)
            
            # Overall template compliance
            compliance_values = list(compliance_metrics.values())
            compliance_metrics['overall_template_compliance'] = sum(compliance_values) / len(compliance_values)
            
            return compliance_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating template compliance: {e}")
            return {
                'section_coverage': 0.0,
                'section_completeness': 0.0,
                'markdown_compliance': 0.0,
                'table_structure_compliance': 0.0,
                'overall_template_compliance': 0.0
            }

    def extract_code_facts(self, abap_code: str) -> Dict[str, Set[str]]:
        """Extract factual information from ABAP code."""
        facts = {
            'methods': set(),
            'parameters': set(),
            'tables': set(),
            'classes': set()
        }
        
        code_upper = abap_code.upper()
        
        # Extract methods
        method_matches = re.findall(r'METHOD\s+(\w+)', code_upper)
        facts['methods'].update(method_matches)
        
        # Extract parameters
        param_matches = re.findall(r'(?:IMPORTING|EXPORTING|CHANGING)\s+(\w+)', code_upper)
        facts['parameters'].update(param_matches)
        
        # Extract table names
        table_matches = re.findall(r'(?:FROM|INTO|UPDATE|DELETE\s+FROM)\s+(\w+)', code_upper)
        facts['tables'].update(table_matches)
        
        # Extract class names
        class_matches = re.findall(r'CLASS\s+(\w+)', code_upper)
        facts['classes'].update(class_matches)
        
        return facts

    def extract_documentation_claims(self, doc: str) -> Dict[str, Set[str]]:
        """Extract claims/mentions from documentation."""
        claims = {
            'methods': set(),
            'parameters': set(),
            'tables': set(),
            'classes': set()
        }
        
        doc_upper = doc.upper()
        
        # Look for method mentions
        method_patterns = [
            r'METHOD[:\s]+(\w+)',
            r'`(\w+)`.*METHOD',
            r'\*\*(\w+)\*\*.*METHOD'
        ]
        
        for pattern in method_patterns:
            matches = re.findall(pattern, doc_upper)
            claims['methods'].update(matches)
        
        # Look for table mentions
        table_section = re.search(r'DATABASE TABLES.*?(?=##|$)', doc_upper, re.DOTALL)
        if table_section:
            table_matches = re.findall(r'\|\s*(\w+)\s*\|', table_section.group())
            claims['tables'].update(table_matches)
        
        return claims

    def validate_abap_constructs(self, doc: str) -> float:
        """Validate that ABAP constructs mentioned in doc are real."""
        doc_upper = doc.upper()
        
        # Find all ABAP-like keywords in the documentation
        potential_abap_words = re.findall(r'\b[A-Z_]{2,}\b', doc_upper)
        
        if not potential_abap_words:
            return 1.0
        
        valid_mentions = sum(1 for word in potential_abap_words if word in self.valid_abap_keywords)
        return valid_mentions / len(potential_abap_words)

    def extract_documentation_sections(self, doc: str) -> Dict[str, str]:
        """Extract sections from documentation based on headers."""
        sections = {}
        
        # Split by markdown headers
        header_pattern = r'^(#{1,6})\s*(.+?)$'
        lines = doc.split('\n')
        current_section = ""
        current_content = []
        
        for line in lines:
            header_match = re.match(header_pattern, line.strip())
            if header_match:
                if current_section:
                    sections[current_section.lower()] = '\n'.join(current_content)
                current_section = header_match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)
        
        if current_section:
            sections[current_section.lower()] = '\n'.join(current_content)
        
        return sections

    def is_section_present(self, expected_section: str, doc_sections: Dict[str, str]) -> bool:
        """Check if a section is present in the documentation."""
        expected_lower = expected_section.lower()
        for section_name in doc_sections.keys():
            if expected_lower in section_name or section_name in expected_lower:
                return True
        return False

    def is_section_complete(self, expected_section: str, doc_sections: Dict[str, str]) -> bool:
        """Check if a section has meaningful content."""
        if not self.is_section_present(expected_section, doc_sections):
            return False
        
        # Find the section content
        expected_lower = expected_section.lower()
        section_content = ""
        for section_name, content in doc_sections.items():
            if expected_lower in section_name or section_name in expected_lower:
                section_content = content
                break
        
        # Check if content is meaningful
        placeholders = ['[', 'TODO', 'TBD', 'PLACEHOLDER', 'DESCRIPTION']
        content_words = section_content.strip().split()
        
        if len(content_words) < 3:
            return False
        
        placeholder_count = sum(1 for word in content_words if any(ph in word.upper() for ph in placeholders))
        return placeholder_count / len(content_words) < 0.5

    def validate_markdown_format(self, doc: str) -> float:
        """Validate markdown formatting quality."""
        issues = 0
        total_checks = 3
        
        # Check for proper header hierarchy
        if not self.has_proper_header_hierarchy(doc):
            issues += 1
        
        # Check for proper code block formatting
        if not self.has_proper_code_blocks(doc):
            issues += 1
        
        # Check for consistent formatting
        if not self.has_consistent_formatting(doc):
            issues += 1
        
        return 1 - (issues / total_checks)

    def validate_table_structures(self, doc: str) -> float:
        """Validate markdown table formatting."""
        table_pattern = r'\|.*\|'
        table_lines = re.findall(table_pattern, doc)
        
        if not table_lines:
            return 1.0
        
        # Simple validation - check if we have header separators
        valid_tables = 0
        total_tables = 1  # Assume at least one table group
        
        for i, line in enumerate(table_lines):
            if i > 0 and re.match(r'\|[\s\-:]+\|', line):
                valid_tables += 1
        
        return min(1.0, valid_tables / total_tables)

    def has_proper_header_hierarchy(self, doc: str) -> bool:
        """Check header hierarchy."""
        headers = re.findall(r'^(#{1,6})', doc, re.MULTILINE)
        if not headers:
            return True
        
        header_levels = [len(h) for h in headers]
        for i in range(1, len(header_levels)):
            if header_levels[i] - header_levels[i-1] > 1:
                return False
        
        return True

    def has_proper_code_blocks(self, doc: str) -> bool:
        """Check code block formatting."""
        fence_pattern = r'```'
        fences = re.findall(fence_pattern, doc)
        return len(fences) % 2 == 0

    def has_consistent_formatting(self, doc: str) -> bool:
        """Check for consistent formatting."""
        # Simple check for consistent list formatting
        bullet_types = []
        if re.search(r'^\s*-\s', doc, re.MULTILINE):
            bullet_types.append('-')
        if re.search(r'^\s*\*\s', doc, re.MULTILINE):
            bullet_types.append('*')
        if re.search(r'^\s*\+\s', doc, re.MULTILINE):
            bullet_types.append('+')
        
        return len(bullet_types) <= 1  # Consistent if using only one bullet type