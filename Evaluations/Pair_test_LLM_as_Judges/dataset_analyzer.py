#!/usr/bin/env python3
"""
Dataset Analyzer for ABAP RAG Evaluation
Analyzes your dataset structure and suggests optimal evaluation parameters
"""

import json
import os
from collections import defaultdict
from typing import Dict, List

class DatasetAnalyzer:
    """Analyze dataset structure and provide recommendations"""
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_files(self, open_source_file: str, closed_source_file: str) -> Dict:
        """Comprehensive analysis of dataset files"""
        
        print(f"\n{'='*60}")
        print(f"DATASET ANALYSIS")
        print(f"{'='*60}")
        
        # Check if files exist
        if not os.path.exists(open_source_file):
            return {"error": f"Open source file not found: {open_source_file}"}
        
        if not os.path.exists(closed_source_file):
            return {"error": f"Closed source file not found: {closed_source_file}"}
        
        try:
            # Load files
            with open(open_source_file, 'r', encoding='utf-8') as f:
                open_data = json.load(f)
            
            with open(closed_source_file, 'r', encoding='utf-8') as f:
                closed_data = json.load(f)
            
            # Basic validation
            validation_results = self._validate_structure(open_data, closed_data)
            if validation_results["errors"]:
                return validation_results
            
            # Detailed analysis
            structure_analysis = self._analyze_structure(open_data, closed_data)
            balance_analysis = self._analyze_balance(open_data)
            recommendations = self._generate_recommendations(structure_analysis, balance_analysis)
            
            results = {
                "files": {
                    "open_source_file": open_source_file,
                    "closed_source_file": closed_source_file
                },
                "validation": validation_results,
                "structure": structure_analysis,
                "balance": balance_analysis,
                "recommendations": recommendations
            }
            
            self.analysis_results = results
            self._print_analysis(results)
            
            return results
            
        except json.JSONDecodeError as e:
            return {"error": f"JSON parsing error: {e}"}
        except Exception as e:
            return {"error": f"Analysis error: {e}"}
    
    def _validate_structure(self, open_data: List, closed_data: List) -> Dict:
        """Validate basic file structure"""
        
        errors = []
        warnings = []
        
        # Check if both are lists
        if not isinstance(open_data, list):
            errors.append("Open source file should contain a JSON array")
        if not isinstance(closed_data, list):
            errors.append("Closed source file should contain a JSON array")
        
        if errors:
            return {"errors": errors, "warnings": warnings}
        
        # Check lengths match
        if len(open_data) != len(closed_data):
            errors.append(f"File lengths don't match: {len(open_data)} vs {len(closed_data)}")
        
        if not open_data:
            errors.append("Files are empty")
            return {"errors": errors, "warnings": warnings}
        
        # Check required fields
        required_fields = ["Question", "reference_context", "Question_type"]
        response_fields = ["RAG_response_1", "RAG_response_2", "response", "answer", "output"]
        
        for i, (open_item, closed_item) in enumerate(zip(open_data, closed_data)):
            # Check required fields
            for field in required_fields:
                if field not in open_item:
                    errors.append(f"Open source item {i}: Missing field '{field}'")
                if field not in closed_item:
                    errors.append(f"Closed source item {i}: Missing field '{field}'")
            
            # Check response fields
            open_has_response = any(field in open_item for field in response_fields)
            closed_has_response = any(field in closed_item for field in response_fields)
            
            if not open_has_response:
                errors.append(f"Open source item {i}: No response field found")
            if not closed_has_response:
                errors.append(f"Closed source item {i}: No response field found")
            
            # Check questions match
            if open_item.get("Question") != closed_item.get("Question"):
                errors.append(f"Item {i}: Questions don't match")
                if len(errors) < 5:  # Don't spam with too many errors
                    errors.append(f"  Open: {open_item.get('Question', 'N/A')[:100]}")
                    errors.append(f"  Closed: {closed_item.get('Question', 'N/A')[:100]}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _analyze_structure(self, open_data: List, closed_data: List) -> Dict:
        """Analyze file structure details"""
        
        if not open_data:
            return {}
        
        # Analyze fields
        open_fields = set()
        closed_fields = set()
        response_fields_open = set()
        response_fields_closed = set()
        
        for item in open_data:
            if isinstance(item, dict):
                open_fields.update(item.keys())
                # Find response fields
                for key in item.keys():
                    if "response" in key.lower():
                        response_fields_open.add(key)
        
        for item in closed_data:
            if isinstance(item, dict):
                closed_fields.update(item.keys())
                # Find response fields
                for key in item.keys():
                    if "response" in key.lower():
                        response_fields_closed.add(key)
        
        return {
            "total_questions": len(open_data),
            "open_source_fields": sorted(list(open_fields)),
            "closed_source_fields": sorted(list(closed_fields)),
            "response_fields_open": sorted(list(response_fields_open)),
            "response_fields_closed": sorted(list(response_fields_closed)),
            "common_fields": sorted(list(open_fields & closed_fields)),
            "unique_to_open": sorted(list(open_fields - closed_fields)),
            "unique_to_closed": sorted(list(closed_fields - open_fields))
        }
    
    def _analyze_balance(self, open_data: List) -> Dict:
        """Analyze question type balance"""
        
        type_counts = defaultdict(int)
        type_questions = defaultdict(list)
        
        for i, item in enumerate(open_data):
            if isinstance(item, dict):
                qtype = item.get("Question_type", "Unknown")
                type_counts[qtype] += 1
                type_questions[qtype].append({
                    "index": i,
                    "question": item.get("Question", "")[:100] + "..." if len(item.get("Question", "")) > 100 else item.get("Question", "")
                })
        
        # Calculate balance metrics
        counts = list(type_counts.values())
        min_count = min(counts) if counts else 0
        max_count = max(counts) if counts else 0
        total_count = sum(counts)
        
        balance_score = min_count / max_count if max_count > 0 else 0
        
        return {
            "question_types": dict(type_counts),
            "total_questions": total_count,
            "min_questions_per_type": min_count,
            "max_questions_per_type": max_count,
            "balance_score": balance_score,  # 1.0 = perfectly balanced, 0.0 = very unbalanced
            "type_details": dict(type_questions)
        }
    
    def _generate_recommendations(self, structure: Dict, balance: Dict) -> Dict:
        """Generate recommendations based on analysis"""
        
        recommendations = {
            "evaluation_strategies": [],
            "balancing_options": [],
            "potential_issues": [],
            "next_steps": []
        }
        
        total_questions = structure.get("total_questions", 0)
        min_per_type = balance.get("min_questions_per_type", 0)
        balance_score = balance.get("balance_score", 0)
        
        # Evaluation strategy recommendations
        if total_questions >= 50:
            recommendations["evaluation_strategies"].append({
                "strategy": "Full Dataset Evaluation",
                "description": f"Use all {total_questions} questions for maximum statistical power",
                "pros": ["Maximum data utilization", "Strong statistical power"],
                "cons": ["Unbalanced question types", "Longer evaluation time"]
            })
        
        if min_per_type >= 3:
            recommendations["evaluation_strategies"].append({
                "strategy": "Balanced Evaluation",
                "description": f"Use {min_per_type} questions per type for balanced comparison",
                "pros": ["Fair comparison across question types", "Clear patterns"],
                "cons": ["Reduced dataset size", "Some data unused"]
            })
        
        # Balancing options
        if balance_score < 0.5:  # Unbalanced
            recommendations["balancing_options"].append({
                "option": f"Balance to {min_per_type} questions per type",
                "total_questions": min_per_type * len(balance["question_types"]),
                "description": "Perfect balance using minimum available"
            })
            
            if min_per_type < 3:
                recommendations["balancing_options"].append({
                    "option": "Balance to 3 questions per type (if possible)",
                    "total_questions": 3 * len(balance["question_types"]),
                    "description": "Good balance for pattern detection",
                    "warning": "Some types have fewer than 3 questions"
                })
        
        # Identify potential issues
        if total_questions < 10:
            recommendations["potential_issues"].append("Very small dataset - results may not be statistically significant")
        
        if balance_score < 0.3:
            recommendations["potential_issues"].append("Highly unbalanced dataset - consider creating more questions for underrepresented types")
        
        if len(balance["question_types"]) == 1:
            recommendations["potential_issues"].append("Only one question type - cannot analyze performance differences across complexity levels")
        
        # Next steps
        recommendations["next_steps"] = [
            "1. Review the analysis above",
            "2. Choose an evaluation strategy based on your thesis goals",
            "3. Set up API keys for judge models (Gemini, Grok, Qwen)",
            "4. Run evaluation using run_evaluation.py",
            "5. Analyze results for thesis findings"
        ]
        
        return recommendations
    
    def _print_analysis(self, results: Dict):
        """Print analysis results in a readable format"""
        
        structure = results["structure"]
        balance = results["balance"]
        recommendations = results["recommendations"]
        
        print(f"\n📊 DATASET OVERVIEW")
        print(f"   Total Questions: {structure['total_questions']}")
        print(f"   Question Types: {len(balance['question_types'])}")
        
        print(f"\n📋 QUESTION TYPE DISTRIBUTION")
        for qtype, count in balance["question_types"].items():
            percentage = (count / balance["total_questions"]) * 100
            print(f"   {qtype}: {count} questions ({percentage:.1f}%)")
        
        print(f"\n⚖️  BALANCE ANALYSIS")
        print(f"   Balance Score: {balance['balance_score']:.2f} (1.0 = perfect balance)")
        print(f"   Min per type: {balance['min_questions_per_type']}")
        print(f"   Max per type: {balance['max_questions_per_type']}")
        
        if balance['balance_score'] >= 0.8:
            print(f"   Status: ✅ Well balanced")
        elif balance['balance_score'] >= 0.5:
            print(f"   Status: ⚠️  Moderately balanced")
        else:
            print(f"   Status: ❌ Unbalanced")
        
        print(f"\n🔧 FIELD ANALYSIS")
        print(f"   Open source fields: {', '.join(structure['open_source_fields'])}")
        print(f"   Closed source fields: {', '.join(structure['closed_source_fields'])}")
        print(f"   Response fields (open): {', '.join(structure['response_fields_open'])}")
        print(f"   Response fields (closed): {', '.join(structure['response_fields_closed'])}")
        
        if recommendations["potential_issues"]:
            print(f"\n⚠️  POTENTIAL ISSUES")
            for issue in recommendations["potential_issues"]:
                print(f"   - {issue}")
        
        print(f"\n💡 RECOMMENDATIONS")
        for strategy in recommendations["evaluation_strategies"]:
            print(f"   📋 {strategy['strategy']}")
            print(f"      {strategy['description']}")
            print(f"      Pros: {', '.join(strategy['pros'])}")
            print(f"      Cons: {', '.join(strategy['cons'])}")
        
        print(f"\n📝 NEXT STEPS")
        for step in recommendations["next_steps"]:
            print(f"   {step}")
    
    def preview_balancing(self, questions_per_type: int) -> Dict:
        """Preview what balancing would look like"""
        
        if not self.analysis_results:
            return {"error": "Run analyze_files first"}
        
        balance = self.analysis_results["balance"]
        preview = {
            "requested_per_type": questions_per_type,
            "type_analysis": {},
            "total_after_balancing": 0,
            "warnings": []
        }
        
        for qtype, count in balance["question_types"].items():
            will_select = min(count, questions_per_type)
            preview["type_analysis"][qtype] = {
                "available": count,
                "will_select": will_select,
                "status": "perfect" if count >= questions_per_type else "insufficient"
            }
            
            if count < questions_per_type:
                preview["warnings"].append(f"{qtype}: Only {count} available, requested {questions_per_type}")
            
            preview["total_after_balancing"] += will_select
        
        print(f"\n🔍 BALANCING PREVIEW ({questions_per_type} per type)")
        print(f"   Total questions after balancing: {preview['total_after_balancing']}")
        
        for qtype, analysis in preview["type_analysis"].items():
            status_icon = "✅" if analysis["status"] == "perfect" else "⚠️"
            print(f"   {status_icon} {qtype}: {analysis['available']} → {analysis['will_select']}")
        
        if preview["warnings"]:
            print(f"\n   Warnings:")
            for warning in preview["warnings"]:
                print(f"   - {warning}")
        
        return preview
    
    def suggest_optimal_balance(self) -> Dict:
        """Suggest optimal balancing strategy"""
        
        if not self.analysis_results:
            return {"error": "Run analyze_files first"}
        
        balance = self.analysis_results["balance"]
        suggestions = []
        
        min_count = balance["min_questions_per_type"]
        type_counts = balance["question_types"]
        
        # Perfect balance option
        if min_count > 0:
            suggestions.append({
                "questions_per_type": min_count,
                "total_questions": min_count * len(type_counts),
                "description": f"Perfect balance using all available from smallest type ({min_count})",
                "recommendation": "Best for balanced comparison" if min_count >= 3 else "Limited statistical power"
            })
        
        # Good balance options
        for target in [3, 5, 10]:
            if min_count >= target:
                suggestions.append({
                    "questions_per_type": target,
                    "total_questions": target * len(type_counts),
                    "description": f"Balanced evaluation with {target} questions per type",
                    "recommendation": "Good balance with adequate statistical power"
                })
        
        # Full dataset option
        total_questions = sum(type_counts.values())
        suggestions.append({
            "questions_per_type": None,
            "total_questions": total_questions,
            "description": "Use all available questions (unbalanced)",
            "recommendation": "Maximum statistical power, but unbalanced"
        })
        
        print(f"\n💡 BALANCING SUGGESTIONS")
        for i, suggestion in enumerate(suggestions, 1):
            per_type = suggestion["questions_per_type"] or "all available"
            print(f"   {i}. {per_type} questions per type")
            print(f"      → {suggestion['total_questions']} total questions")
            print(f"      → {suggestion['description']}")
            print(f"      → {suggestion['recommendation']}")
        
        return {
            "current_distribution": dict(type_counts),
            "suggestions": suggestions,
            "recommended": suggestions[0] if suggestions else None
        }

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze ABAP RAG evaluation dataset")
    parser.add_argument("open_source_file", help="Path to open source responses JSON file")
    parser.add_argument("closed_source_file", help="Path to closed source responses JSON file")
    parser.add_argument("--preview-balance", type=int, metavar="N", 
                       help="Preview balancing with N questions per type")
    
    args = parser.parse_args()
    
    analyzer = DatasetAnalyzer()
    
    # Run analysis
    results = analyzer.analyze_files(args.open_source_file, args.closed_source_file)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    # Show balancing suggestions
    analyzer.suggest_optimal_balance()
    
    # Preview specific balance if requested
    if args.preview_balance:
        analyzer.preview_balancing(args.preview_balance)

if __name__ == "__main__":
    main()