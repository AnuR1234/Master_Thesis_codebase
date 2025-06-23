#!/usr/bin/env python3
"""
ABAP RAG Evaluation System - Main Evaluator
Author: Your Name
Purpose: Compare open-source vs closed-source RAG systems using LLM-as-Judge methodology
Updated with current model names as of June 2025
"""

import json
import random
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime
import statistics
from collections import defaultdict
import time
import os
from dotenv import load_dotenv
RAG_ENV_PATH = r".env"
load_dotenv(RAG_ENV_PATH)

@dataclass
class EvaluationResult:
    """Single evaluation result with blind tracking"""
    evaluation_id: str
    question: str
    question_type: str
    judge_model: str
    
    # Original responses
    open_source_response: str
    closed_source_response: str
    
    # Blinded presentation
    response_a: str
    response_b: str
    position_mapping: Dict[str, str]
    
    # Judge decision
    judge_choice: str
    actual_winner: str
    confidence: float
    reasoning: str
    
    # Detailed scores
    scores_a: Dict[str, int]
    scores_b: Dict[str, int]
    
    timestamp: str

class ABAPRAGEvaluator:
    """Main evaluator for ABAP RAG system comparison"""
    
    def __init__(self, judge_models: List[str] = None):
        if judge_models is None:
            # Updated with available model names
            judge_models = ["gemini-1.5-pro", "grok-3", "qwen-2.5-coder-32b-instruct"]
        
        self.judge_models = judge_models
        self.results: List[EvaluationResult] = []
        self.question_types_found = set()
        self.balancing_report = None
        
        # Import API handler
        try:
            from api_handlers import RealJudgeAPIs
            self.api_handler = None  # Will be initialized when needed
        except ImportError:
            print("⚠️  API handlers not found. Using simulation mode.")
            self.api_handler = None
    
    def setup_apis(self, gemini_key: Optional[str] = None, grok_key: Optional[str] = None, 
                   qwen_provider: str = "together", qwen_key: Optional[str] = None):
        """Setup real API connections"""
        try:
            from api_handlers import RealJudgeAPIs
            self.api_handler = RealJudgeAPIs(
                gemini_api_key=gemini_key,
                grok_api_key=grok_key,
                qwen_api_provider=qwen_provider,
                qwen_api_key=qwen_key
            )
            print("✅ Real APIs initialized")
        except Exception as e:
            print(f"⚠️  Could not initialize APIs: {e}")
            print("Using simulation mode")
    
    def load_and_balance_data(self, open_source_file: str, closed_source_file: str,
                             questions_per_type: int = None) -> List[Tuple[Dict, Dict]]:
        """Load data and optionally balance by question type"""
        
        # Verify files exist
        if not os.path.exists(open_source_file):
            raise FileNotFoundError(f"Open source file not found: {open_source_file}")
        if not os.path.exists(closed_source_file):
            raise FileNotFoundError(f"Closed source file not found: {closed_source_file}")
        
        # Load both files
        with open(open_source_file, 'r', encoding='utf-8') as f:
            open_source_data = json.load(f)
        
        with open(closed_source_file, 'r', encoding='utf-8') as f:
            closed_source_data = json.load(f)
        
        # Verify same number of questions
        if len(open_source_data) != len(closed_source_data):
            raise ValueError(f"Files have different lengths: {len(open_source_data)} vs {len(closed_source_data)}")
        
        # Group by question type
        grouped_data = defaultdict(list)
        
        for i, (open_item, closed_item) in enumerate(zip(open_source_data, closed_source_data)):
            # Verify questions match
            if open_item["Question"] != closed_item["Question"]:
                raise ValueError(f"Question mismatch at index {i}:\nOpen: {open_item['Question']}\nClosed: {closed_item['Question']}")
            
            question_type = open_item.get("Question_type", "Unknown")
            grouped_data[question_type].append((open_item, closed_item))
            self.question_types_found.add(question_type)
        
        print(f"\n{'='*60}")
        print(f"DATASET ANALYSIS")
        print(f"{'='*60}")
        print(f"Found question types: {list(self.question_types_found)}")
        
        total_available = 0
        for qtype, items in grouped_data.items():
            print(f"  - {qtype}: {len(items)} questions available")
            total_available += len(items)
        
        print(f"Total questions in dataset: {total_available}")
        
        # Balance data if requested
        if questions_per_type is not None:
            print(f"\n{'='*60}")
            print(f"BALANCING TO {questions_per_type} QUESTIONS PER TYPE")
            print(f"{'='*60}")
            
            balanced_data = []
            balancing_report = {}
            
            for qtype, items in grouped_data.items():
                if len(items) >= questions_per_type:
                    # Randomly select N questions from this type
                    selected = random.sample(items, questions_per_type)
                    balanced_data.extend(selected)
                    balancing_report[qtype] = {
                        "available": len(items),
                        "selected": len(selected),
                        "status": "✓ Balanced"
                    }
                    print(f"  ✓ {qtype}: Selected {len(selected)} from {len(items)} available")
                    
                elif len(items) > 0:
                    # Use all available questions if less than N
                    balanced_data.extend(items)
                    balancing_report[qtype] = {
                        "available": len(items),
                        "selected": len(items),
                        "status": f"⚠ Used all {len(items)} (less than {questions_per_type})"
                    }
                    print(f"  ⚠ {qtype}: Using all {len(items)} questions (less than {questions_per_type})")
                    
                else:
                    # No questions of this type
                    balancing_report[qtype] = {
                        "available": 0,
                        "selected": 0,
                        "status": "❌ No questions available"
                    }
                    print(f"  ❌ {qtype}: No questions available")
            
            print(f"\n{'='*60}")
            print(f"BALANCING SUMMARY")
            print(f"{'='*60}")
            print(f"Total questions selected: {len(balanced_data)}")
            print(f"Questions per type requested: {questions_per_type}")
            
            # Show final distribution
            final_distribution = defaultdict(int)
            for open_item, closed_item in balanced_data:
                qtype = open_item.get("Question_type", "Unknown")
                final_distribution[qtype] += 1
            
            print(f"\nFinal distribution:")
            for qtype, count in final_distribution.items():
                status = "✓ Perfect" if count == questions_per_type else f"⚠ Only {count}"
                print(f"  - {qtype}: {count} questions ({status})")
            
            # Store balancing report for later reference
            self.balancing_report = balancing_report
            return balanced_data
        
        else:
            # Use all data
            all_data = []
            for items in grouped_data.values():
                all_data.extend(items)
                
            print(f"\n{'='*60}")
            print(f"USING FULL DATASET")
            print(f"{'='*60}")
            print(f"Total questions: {len(all_data)}")
            
            # Show distribution
            print(f"\nDistribution:")
            for qtype, items in grouped_data.items():
                print(f"  - {qtype}: {len(items)} questions")
            
            return all_data
    
    def evaluate_question_pair(self, open_item: Dict, closed_item: Dict, 
                              judge_model: str, num_position_switches: int = 3) -> List[EvaluationResult]:
        """Evaluate a single question with position switching"""
        
        question = open_item["Question"]
        reference_context = open_item["reference_context"]
        question_type = open_item["Question_type"]
        
        # Extract responses
        open_response = self._extract_response(open_item)
        closed_response = self._extract_response(closed_item)
        
        switch_results = []
        
        for switch_round in range(num_position_switches):
            eval_id = str(uuid.uuid4())
            
            # Randomly assign positions
            if random.random() < 0.5:
                response_a = open_response
                response_b = closed_response
                position_mapping = {"A": "open_source", "B": "closed_source"}
            else:
                response_a = closed_response
                response_b = open_response
                position_mapping = {"A": "closed_source", "B": "open_source"}
            
            # Create judge prompt
            prompt = self._create_judge_prompt(
                question, reference_context, question_type, response_a, response_b
            )
            
            # Get judge evaluation
            judge_response = self._call_judge_api(prompt, judge_model)
            
            # Determine actual winner
            judge_choice = judge_response["winner"]
            if judge_choice == "Tie":
                actual_winner = "tie"
            else:
                actual_winner = position_mapping[judge_choice]
            
            # Create result record
            result = EvaluationResult(
                evaluation_id=eval_id,
                question=question,
                question_type=question_type,
                judge_model=judge_model,
                open_source_response=open_response,
                closed_source_response=closed_response,
                response_a=response_a,
                response_b=response_b,
                position_mapping=position_mapping,
                judge_choice=judge_choice,
                actual_winner=actual_winner,
                confidence=judge_response["confidence"],
                reasoning=judge_response["reasoning"],
                scores_a=judge_response["scores"]["A"],
                scores_b=judge_response["scores"]["B"],
                timestamp=datetime.now().isoformat()
            )
            
            switch_results.append(result)
            self.results.append(result)
        
        return switch_results
    
    def _extract_response(self, item: Dict) -> str:
        """Extract response from item, handling different field names"""
        possible_fields = ["RAG_response_1", "RAG_response_2", "response", "answer", "output"]
        
        for field in possible_fields:
            if field in item:
                return item[field]
        
        # Look for any field with "response" in the name
        for key, value in item.items():
            if "response" in key.lower() and isinstance(value, str):
                return value
        
        raise ValueError(f"Could not find response field in item: {list(item.keys())}")
    
    def _create_judge_prompt(self, question: str, reference_context: str,
                           question_type: str, response_a: str, response_b: str) -> str:
        """Create judge prompt for evaluation"""
        
        # Question type specific guidance
        type_guidance = {
            "Simple": "Focus on basic accuracy and clarity. Simple questions should have direct, correct answers.",
            "Complex": "Evaluate comprehensive coverage, technical depth, and handling of multiple concepts.",
            "Intermediate": "Look for good balance of accuracy and completeness without overwhelming detail.",
            "Advanced": "Assess advanced technical knowledge, edge cases, and expert-level insights.",
            "Situational": "Evaluate practical application in real-world ABAP development scenarios.",
            "Conversational": "Assess natural language understanding and context-aware responses.",
            "Double": "Evaluate ability to handle multi-part questions and provide comprehensive coverage."
        }
        
        guidance = type_guidance.get(question_type, "Evaluate based on technical accuracy and practical value.")
        
        prompt = f"""[EVALUATION TASK]
You are evaluating two responses to an SAP ABAP technical question. Determine which response is better based on technical accuracy, completeness, and ABAP domain expertise.

[QUESTION CLASSIFICATION]
This is a {question_type} level question. {guidance}

[REFERENCE CONTEXT]
{reference_context}

[QUESTION]
{question}

[RESPONSE A]
{response_a}

[RESPONSE B]
{response_b}

[EVALUATION CRITERIA]
Rate each response on a scale of 1-10 for:

1. **Technical Accuracy**: Correct ABAP syntax, method names, function modules
2. **SAP Domain Knowledge**: Understanding of SAP-specific concepts and terminology  
3. **Completeness**: How well the response covers all aspects of the question
4. **Clarity**: How clear and well-structured the explanation is
5. **Practical Value**: How useful this would be for an ABAP developer

[QUESTION TYPE SPECIFIC CONSIDERATIONS]
For {question_type} questions:
- {guidance}
- Consider the appropriate level of detail for this complexity level
- Evaluate if the response matches the expected depth for a {question_type} question

[OUTPUT FORMAT]
Respond with valid JSON in this exact format:
{{
    "winner": "A" | "B" | "Tie",
    "confidence": 0.0-1.0,
    "scores": {{
        "A": {{
            "technical_accuracy": 1-10,
            "sap_domain_knowledge": 1-10,
            "completeness": 1-10,
            "clarity": 1-10,
            "practical_value": 1-10
        }},
        "B": {{
            "technical_accuracy": 1-10,
            "sap_domain_knowledge": 1-10,
            "completeness": 1-10,
            "clarity": 1-10,
            "practical_value": 1-10
        }}
    }},
    "reasoning": "Detailed explanation considering this is a {question_type} level question",
    "question_type_assessment": "How well each response addresses the {question_type} complexity level"
}}

Your evaluation:"""
        
        return prompt
    
    def _call_judge_api(self, prompt: str, judge_model: str) -> Dict:
        """Call judge API - real or simulation"""
        
        if self.api_handler:
            try:
                if judge_model == "gemini-1.5-pro":  # Updated to match available model
                    return self.api_handler.call_gemini_2_5_pro(prompt)
                elif judge_model == "grok-3":
                    return self.api_handler.call_grok_3(prompt)
                elif judge_model == "qwen-2.5-coder-32b-instruct":
                    return self.api_handler.call_qwen_2_5_72b(prompt)
            except Exception as e:
                print(f"⚠️  API call failed for {judge_model}: {e}")
                return self._simulate_judge_response(prompt, judge_model)
        
        return self._simulate_judge_response(prompt, judge_model)
    
    def _simulate_judge_response(self, prompt: str, judge_model: str) -> Dict:
        """Simulate judge response for testing"""
        
        choices = ["A", "B", "Tie"]
        weights = [0.4, 0.4, 0.2]
        winner = random.choices(choices, weights=weights)[0]
        
        base_score = random.randint(6, 8)
        scores_a = {
            "technical_accuracy": base_score + random.randint(-1, 2),
            "sap_domain_knowledge": base_score + random.randint(-1, 2),
            "completeness": base_score + random.randint(-1, 2),
            "clarity": base_score + random.randint(-1, 2),
            "practical_value": base_score + random.randint(-1, 2)
        }
        
        scores_b = {
            "technical_accuracy": base_score + random.randint(-1, 2),
            "sap_domain_knowledge": base_score + random.randint(-1, 2),
            "completeness": base_score + random.randint(-1, 2),
            "clarity": base_score + random.randint(-1, 2),
            "practical_value": base_score + random.randint(-1, 2)
        }
        
        # Ensure valid range
        for scores in [scores_a, scores_b]:
            for key in scores:
                scores[key] = max(1, min(10, scores[key]))
        
        return {
            "winner": winner,
            "confidence": random.uniform(0.6, 0.9),
            "scores": {"A": scores_a, "B": scores_b},
            "reasoning": f"{judge_model} evaluation: Response {winner} better addresses the technical requirements.",
            "question_type_assessment": f"Both responses appropriate for question complexity level."
        }
    
    def evaluate_full_dataset(self, open_source_file: str, closed_source_file: str,
                             questions_per_type: int = None, num_position_switches: int = 3) -> Dict:
        """Evaluate complete dataset"""
        
        # Load and optionally balance data
        paired_data = self.load_and_balance_data(
            open_source_file, closed_source_file, questions_per_type
        )
        
        total_evaluations = len(paired_data) * len(self.judge_models) * num_position_switches
        print(f"\nStarting evaluation:")
        print(f"  - Questions: {len(paired_data)}")
        print(f"  - Judges: {len(self.judge_models)}")
        print(f"  - Position switches: {num_position_switches}")
        print(f"  - Total evaluations: {total_evaluations}")
        
        # Evaluate each question
        for i, (open_item, closed_item) in enumerate(paired_data):
            question_preview = open_item["Question"][:60] + "..." if len(open_item["Question"]) > 60 else open_item["Question"]
            question_type = open_item["Question_type"]
            
            print(f"\n[{i+1}/{len(paired_data)}] {question_type}: {question_preview}")
            
            for judge_model in self.judge_models:
                print(f"  - {judge_model}")
                self.evaluate_question_pair(
                    open_item, closed_item, judge_model, num_position_switches
                )
        
        print(f"\nEvaluation complete! Total evaluations: {len(self.results)}")
        return self.analyze_results()
    
    def analyze_results(self) -> Dict:
        """Comprehensive analysis of evaluation results"""
        
        if not self.results:
            return {"error": "No evaluations completed"}
        
        total_evals = len(self.results)
        total_questions = len(set(r.question for r in self.results))
        
        # Overall results
        open_wins = sum(1 for r in self.results if r.actual_winner == "open_source")
        closed_wins = sum(1 for r in self.results if r.actual_winner == "closed_source")
        ties = sum(1 for r in self.results if r.actual_winner == "tie")
        
        # Detailed question-type analysis
        by_question_type = {}
        for qtype in self.question_types_found:
            type_results = [r for r in self.results if r.question_type == qtype]
            
            if type_results:
                type_open_wins = sum(1 for r in type_results if r.actual_winner == "open_source")
                type_closed_wins = sum(1 for r in type_results if r.actual_winner == "closed_source")
                type_ties = sum(1 for r in type_results if r.actual_winner == "tie")
                type_total = len(type_results)
                
                by_question_type[qtype] = {
                    "total_evaluations": type_total,
                    "unique_questions": len(set(r.question for r in type_results)),
                    "open_source_wins": type_open_wins,
                    "closed_source_wins": type_closed_wins,
                    "ties": type_ties,
                    "open_source_win_rate": type_open_wins / type_total,
                    "closed_source_win_rate": type_closed_wins / type_total,
                    "tie_rate": type_ties / type_total,
                    "average_confidence": statistics.mean([r.confidence for r in type_results])
                }
        
        return {
            "summary": {
                "total_evaluations": total_evals,
                "total_questions": total_questions,
                "question_types": list(self.question_types_found),
                "judges_used": self.judge_models,
                "open_source_wins": open_wins,
                "closed_source_wins": closed_wins,
                "ties": ties,
                "open_source_win_rate": open_wins / total_evals,
                "closed_source_win_rate": closed_wins / total_evals,
                "tie_rate": ties / total_evals
            },
            "by_question_type": by_question_type
        }
    
    def export_results(self, filename: str = None) -> str:
        """Export complete results"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"abap_rag_evaluation_{timestamp}.json"
        
        export_data = {
            "metadata": {
                "evaluation_timestamp": datetime.now().isoformat(),
                "total_evaluations": len(self.results),
                "total_questions": len(set(r.question for r in self.results)),
                "judge_models": self.judge_models,
                "methodology": "LLM-as-Judge with position switching for bias reduction",
                "balancing_report": self.balancing_report
            },
            "analysis": self.analyze_results(),
            "detailed_results": [asdict(result) for result in self.results]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return filename

if __name__ == "__main__":
    # Example usage
    evaluator = ABAPRAGEvaluator()
    
    # Setup APIs if you have keys (uncomment and add your keys)
    evaluator.setup_apis(
         gemini_key=os.getenv("GEMINI_API_KEY"),
         grok_key=os.getenv("GROK_API_KEY"), 
         qwen_provider="together",  # Fixed: this should be the provider name, not the API key
         qwen_key=os.getenv("QWEN_API_KEY")
     )
    
    print("ABAP RAG Evaluator initialized!")
    print("Use dataset_analyzer.py to check your data first.")
    print("Use run_evaluation.py to run the full evaluation.")