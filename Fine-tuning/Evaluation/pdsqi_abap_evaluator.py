#!/usr/bin/env python3
"""
PDSQI-9 Rubric Evaluator for ABAP Documentation
"""

import re
import logging
from typing import Dict, List, Set

class PDSQI9_ABAP_Evaluator:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # ABAP validation patterns
        self.abap_patterns = {
            'methods': r'(?i)method\s+(\w+)',
            'parameters': r'(?i)(?:importing|exporting|changing)\s+(\w+)',
            'tables': r'(?i)(?:select|from|into|update|delete\s+from)\s+(\w+)',
            'classes': r'(?i)class\s+(\w+)'
        }
        
        # Required sections by file type
        self.required_sections = {
            'class': ['overview', 'class definition', 'method implementation'],
            'test_class': ['overview', 'test methods', 'test data'],
            'report': ['overview', 'program flow', 'selection screen'],
            'function': ['overview', 'interface', 'processing logic']
        }
        
        # Valid ABAP keywords
        self.valid_abap_keywords = {
            'CLASS', 'METHOD', 'FUNCTION', 'SELECT', 'INSERT', 'UPDATE', 'DELETE',
            'DATA', 'TYPES', 'CONSTANTS', 'PARAMETERS', 'IMPORTING', 'EXPORTING',
            'CHANGING', 'TABLES', 'EXCEPTIONS', 'LOOP', 'IF', 'ENDIF', 'ENDLOOP',
            'FORM', 'PERFORM', 'CALL', 'BAPI', 'RFC', 'SY-SUBRC', 'SY-TABIX'
        }

    def calculate_pdsqi9_scores(self, sample: Dict) -> Dict[str, float]:
        """Calculate all PDSQI-9 scores for a sample."""
        abap_code = sample.get('abap_code', '')
        generated_doc = sample['generated_documentation']
        file_type = sample.get('file_type', 'class')
        
        scores = {}
        
        # Evaluate all 9 dimensions
        scores['pdsqi_accuracy'] = self.evaluate_accuracy(abap_code, generated_doc)
        scores['pdsqi_completeness'] = self.evaluate_completeness(abap_code, generated_doc, file_type)
        scores['pdsqi_hallucinations'] = self.evaluate_hallucinations(abap_code, generated_doc)
        scores['pdsqi_clarity'] = self.evaluate_clarity(generated_doc)
        scores['pdsqi_organization'] = self.evaluate_organization(generated_doc, file_type)
        scores['pdsqi_usefulness'] = self.evaluate_usefulness(generated_doc)
        scores['pdsqi_omissions'] = self.evaluate_omissions(abap_code, generated_doc)
        scores['pdsqi_coherence'] = self.evaluate_coherence(generated_doc)
        scores['pdsqi_conciseness'] = self.evaluate_conciseness(generated_doc)
        
        # Calculate weighted overall score (critical dimensions weighted higher)
        weights = {
            'pdsqi_accuracy': 1.5,
            'pdsqi_completeness': 1.5,
            'pdsqi_hallucinations': 1.5,
            'pdsqi_clarity': 1.0,
            'pdsqi_organization': 1.0,
            'pdsqi_usefulness': 1.0,
            'pdsqi_omissions': 1.2,
            'pdsqi_coherence': 0.8,
            'pdsqi_conciseness': 0.8
        }
        
        weighted_sum = sum(scores[dim] * weights[dim] for dim in scores)
        total_weight = sum(weights.values())
        scores['pdsqi_overall'] = weighted_sum / total_weight
        
        return scores

    def evaluate_accuracy(self, abap_code: str, generated_doc: str) -> float:
        """PDSQI-9 Dimension 1: Accuracy."""
        try:
            score_components = []
            
            # Extract code elements and doc claims
            code_elements = self._extract_code_elements(abap_code)
            doc_claims = self._extract_doc_claims(generated_doc)
            
            # Method accuracy
            if code_elements['methods']:
                method_accuracy = len(code_elements['methods'].intersection(doc_claims['methods'])) / len(code_elements['methods'])
                score_components.append(method_accuracy)
            else:
                score_components.append(1.0)
            
            # Check for hallucinations
            hallucinated_methods = doc_claims['methods'] - code_elements['methods']
            if doc_claims['methods']:
                hallucination_penalty = len(hallucinated_methods) / len(doc_claims['methods'])
                score_components.append(1.0 - hallucination_penalty)
            else:
                score_components.append(1.0)
            
            # ABAP syntax accuracy
            syntax_score = self._validate_abap_syntax(generated_doc)
            score_components.append(syntax_score)
            
            return (sum(score_components) / len(score_components)) * 5
            
        except Exception as e:
            self.logger.error(f"Error evaluating accuracy: {e}")
            return 1.0

    def evaluate_completeness(self, abap_code: str, generated_doc: str, file_type: str) -> float:
        """PDSQI-9 Dimension 2: Completeness."""
        try:
            score_components = []
            
            # Section coverage
            required_sections = self.required_sections.get(file_type, self.required_sections['class'])
            doc_sections = self._extract_sections(generated_doc)
            
            section_coverage = 0
            for section in required_sections:
                if any(section.lower() in doc_section.lower() for doc_section in doc_sections):
                    section_coverage += 1
            
            section_score = section_coverage / len(required_sections)
            score_components.append(section_score)
            
            # Method coverage
            code_methods = self._extract_code_elements(abap_code)['methods']
            if code_methods:
                documented_methods = len([m for m in code_methods if m.lower() in generated_doc.lower()])
                method_coverage = documented_methods / len(code_methods)
                score_components.append(method_coverage)
            else:
                score_components.append(1.0)
            
            return (sum(score_components) / len(score_components)) * 5
            
        except Exception as e:
            self.logger.error(f"Error evaluating completeness: {e}")
            return 1.0

    def evaluate_hallucinations(self, abap_code: str, generated_doc: str) -> float:
        """PDSQI-9 Dimension 3: Hallucinations (reverse scored - higher is better)."""
        try:
            code_elements = self._extract_code_elements(abap_code)
            doc_claims = self._extract_doc_claims(generated_doc)
            
            total_claims = 0
            hallucinated_claims = 0
            
            # Check method hallucinations
            if doc_claims['methods']:
                total_claims += len(doc_claims['methods'])
                hallucinated_methods = doc_claims['methods'] - code_elements['methods']
                hallucinated_claims += len(hallucinated_methods)
            
            # Check for impossible ABAP constructs
            impossible_constructs = self._find_impossible_constructs(generated_doc)
            hallucinated_claims += len(impossible_constructs)
            total_claims += len(impossible_constructs)
            
            if total_claims == 0:
                hallucination_rate = 0
            else:
                hallucination_rate = hallucinated_claims / total_claims
            
            # Return score (1 - hallucination_rate) * 5
            return (1 - hallucination_rate) * 5
            
        except Exception as e:
            self.logger.error(f"Error evaluating hallucinations: {e}")
            return 3.0

    def evaluate_clarity(self, generated_doc: str) -> float:
        """PDSQI-9 Dimension 4: Clarity."""
        try:
            score_components = []
            
            # Sentence length check
            word_count = len(generated_doc.split())
            sentence_count = len(re.findall(r'[.!?]+', generated_doc))
            
            if sentence_count > 0:
                avg_words_per_sentence = word_count / sentence_count
                if 10 <= avg_words_per_sentence <= 20:
                    score_components.append(1.0)
                elif avg_words_per_sentence > 25:
                    score_components.append(0.6)
                else:
                    score_components.append(0.8)
            else:
                score_components.append(0.7)
            
            # Structure clarity
            headers = re.findall(r'^#+\s', generated_doc, re.MULTILINE)
            if len(headers) >= 3:
                score_components.append(1.0)
            else:
                score_components.append(0.6)
            
            # Technical term explanation
            technical_terms = re.findall(r'\b[A-Z]{2,}\b', generated_doc)
            if technical_terms:
                explained_terms = len(re.findall(r'(?:\b[A-Z]{2,}\b).*?(?:is|means|refers to)', generated_doc))
                if explained_terms / len(set(technical_terms)) > 0.2:
                    score_components.append(1.0)
                else:
                    score_components.append(0.7)
            else:
                score_components.append(0.9)
            
            return (sum(score_components) / len(score_components)) * 5
            
        except Exception as e:
            self.logger.error(f"Error evaluating clarity: {e}")
            return 3.0

    def evaluate_organization(self, generated_doc: str, file_type: str) -> float:
        """PDSQI-9 Dimension 5: Organization."""
        try:
            score_components = []
            
            # Header hierarchy
            headers = re.findall(r'^(#+)\s(.+)', generated_doc, re.MULTILINE)
            if self._is_proper_header_hierarchy([len(h[0]) for h in headers]):
                score_components.append(1.0)
            else:
                score_components.append(0.6)
            
            # Starts with overview
            if "overview" in generated_doc.lower()[:500]:
                score_components.append(1.0)
            else:
                score_components.append(0.7)
            
            # Section order
            doc_sections = self._extract_sections(generated_doc)
            required_sections = self.required_sections.get(file_type, self.required_sections['class'])
            section_order_score = self._check_section_order(doc_sections, required_sections)
            score_components.append(section_order_score)
            
            return (sum(score_components) / len(score_components)) * 5
            
        except Exception as e:
            self.logger.error(f"Error evaluating organization: {e}")
            return 3.0

    def evaluate_usefulness(self, generated_doc: str) -> float:
        """PDSQI-9 Dimension 6: Usefulness."""
        try:
            score_components = []
            
            # Actionable information
            actionable_keywords = ['how to', 'to use', 'example', 'step', 'procedure']
            actionable_count = sum(1 for keyword in actionable_keywords if keyword in generated_doc.lower())
            
            if actionable_count >= 3:
                score_components.append(1.0)
            elif actionable_count >= 1:
                score_components.append(0.8)
            else:
                score_components.append(0.5)
            
            # Business context
            context_indicators = ['purpose', 'used for', 'enables', 'supports', 'business']
            context_count = sum(1 for indicator in context_indicators if indicator in generated_doc.lower())
            
            if context_count >= 2:
                score_components.append(1.0)
            else:
                score_components.append(0.7)
            
            # Developer-focused content
            dev_keywords = ['parameter', 'method', 'class', 'function', 'error', 'exception']
            dev_count = sum(1 for keyword in dev_keywords if keyword in generated_doc.lower())
            
            if dev_count >= 4:
                score_components.append(1.0)
            else:
                score_components.append(0.8)
            
            return (sum(score_components) / len(score_components)) * 5
            
        except Exception as e:
            self.logger.error(f"Error evaluating usefulness: {e}")
            return 3.0

    def evaluate_omissions(self, abap_code: str, generated_doc: str) -> float:
        """PDSQI-9 Dimension 7: Omissions (reverse scored)."""
        try:
            score_components = []
            
            # Check for undocumented methods
            code_elements = self._extract_code_elements(abap_code)
            if code_elements['methods']:
                undocumented_methods = []
                for method in code_elements['methods']:
                    if method.lower() not in generated_doc.lower():
                        undocumented_methods.append(method)
                
                if not undocumented_methods:
                    score_components.append(1.0)
                else:
                    omission_rate = len(undocumented_methods) / len(code_elements['methods'])
                    score_components.append(1.0 - omission_rate)
            else:
                score_components.append(1.0)
            
            # Check for error handling documentation
            has_error_handling = any(keyword in abap_code.lower() for keyword in ['error', 'exception', 'sy-subrc'])
            documents_errors = any(keyword in generated_doc.lower() for keyword in ['error', 'exception', 'handling'])
            
            if has_error_handling:
                if documents_errors:
                    score_components.append(1.0)
                else:
                    score_components.append(0.5)
            else:
                score_components.append(1.0)
            
            return (sum(score_components) / len(score_components)) * 5
            
        except Exception as e:
            self.logger.error(f"Error evaluating omissions: {e}")
            return 3.0

    def evaluate_coherence(self, generated_doc: str) -> float:
        """PDSQI-9 Dimension 8: Coherence."""
        try:
            score_components = []
            
            # Consistent terminology
            abap_terms = re.findall(r'\b(?:METHOD|CLASS|FUNCTION|SELECT)\b', generated_doc)
            if len(set(abap_terms)) > 0:
                consistency_score = min(1.0, len(abap_terms) / len(set(abap_terms)) / 2)
                score_components.append(consistency_score)
            else:
                score_components.append(0.9)
            
            # Logical flow
            sections = self._extract_sections(generated_doc)
            if len(sections) >= 3:
                score_components.append(1.0)
            else:
                score_components.append(0.7)
            
            # Cross-references
            if re.search(r'(?:see|refer|mentioned|above|below)', generated_doc, re.IGNORECASE):
                score_components.append(1.0)
            else:
                score_components.append(0.8)
            
            return (sum(score_components) / len(score_components)) * 5
            
        except Exception as e:
            self.logger.error(f"Error evaluating coherence: {e}")
            return 3.0

    def evaluate_conciseness(self, generated_doc: str) -> float:
        """PDSQI-9 Dimension 9: Conciseness."""
        try:
            score_components = []
            
            # Length appropriateness
            word_count = len(generated_doc.split())
            
            if 200 <= word_count <= 800:
                score_components.append(1.0)
            elif word_count > 1200:
                score_components.append(0.6)
            elif word_count < 100:
                score_components.append(0.5)
            else:
                score_components.append(0.8)
            
            # Redundancy check
            sentences = re.split(r'[.!?]+', generated_doc)
            if len(sentences) > 5:
                repeated_phrases = 0
                for i, sentence in enumerate(sentences):
                    for j, other_sentence in enumerate(sentences[i+1:], i+1):
                        if len(sentence.split()) > 3 and sentence.strip() == other_sentence.strip():
                            repeated_phrases += 1
                
                if repeated_phrases == 0:
                    score_components.append(1.0)
                else:
                    score_components.append(0.7)
            else:
                score_components.append(0.9)
            
            return (sum(score_components) / len(score_components)) * 5
            
        except Exception as e:
            self.logger.error(f"Error evaluating conciseness: {e}")
            return 3.0

    # Helper methods
    def _extract_code_elements(self, code: str) -> Dict[str, set]:
        """Extract elements from ABAP code."""
        elements = {pattern: set() for pattern in self.abap_patterns}
        
        for element_type, pattern in self.abap_patterns.items():
            matches = re.findall(pattern, code)
            elements[element_type].update(matches)
        
        return elements

    def _extract_doc_claims(self, doc: str) -> Dict[str, set]:
        """Extract claims from documentation."""
        return self._extract_code_elements(doc)

    def _extract_sections(self, doc: str) -> List[str]:
        """Extract section headers."""
        headers = re.findall(r'^#+\s+(.+)', doc, re.MULTILINE)
        return [h.strip().lower() for h in headers]

    def _validate_abap_syntax(self, doc: str) -> float:
        """Validate ABAP syntax mentions."""
        abap_keywords = re.findall(r'\b[A-Z]{2,}\b', doc)
        
        if not abap_keywords:
            return 1.0
        
        valid_count = sum(1 for kw in abap_keywords if kw in self.valid_abap_keywords)
        return valid_count / len(abap_keywords)

    def _find_impossible_constructs(self, doc: str) -> List[str]:
        """Find impossible ABAP constructs."""
        impossible = []
        
        # Simple check for impossible syntax
        if re.search(r'SELECT.*FROM.*INTO.*WHERE.*GROUP BY.*ORDER BY.*HAVING', doc, re.IGNORECASE):
            impossible.append("Impossible SQL syntax order")
        
        return impossible

    def _is_proper_header_hierarchy(self, header_levels: List[int]) -> bool:
        """Check header hierarchy."""
        if not header_levels:
            return True
        
        for i in range(1, len(header_levels)):
            if header_levels[i] - header_levels[i-1] > 1:
                return False
        
        return True

    def _check_section_order(self, sections: List[str], required: List[str]) -> float:
        """Check section ordering."""
        if not sections or not required:
            return 1.0
        
        order_score = 0
        for i, req_section in enumerate(required):
            for j, doc_section in enumerate(sections):
                if req_section in doc_section:
                    expected_position = i / len(required)
                    actual_position = j / len(sections)
                    if abs(expected_position - actual_position) < 0.3:
                        order_score += 1
                    break
        
        return order_score / len(required)