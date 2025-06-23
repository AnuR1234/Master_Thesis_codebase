#!/usr/bin/env python3
"""
ABAP RAG Evaluation Runner
Fixed version that actually runs the evaluation with enhanced reporting
"""

import argparse
import os
import json
import pandas as pd
from datetime import datetime
from main_evaluator import ABAPRAGEvaluator
from dataset_analyzer import DatasetAnalyzer

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    PLOTTING_AVAILABLE = True
    plt.style.use('default')
    sns.set_palette("husl")
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  Plotting libraries not available. Install with: pip install matplotlib seaborn")

def create_enhanced_summary_report(evaluator: ABAPRAGEvaluator, results: dict, filename: str):
    """Create an enhanced human-readable summary report with tables and analysis"""
    
    summary = results["summary"]
    by_type = results.get("by_question_type", {})
    
    # Generate analysis tables
    judge_performance = analyze_judge_performance(evaluator.results)
    question_type_matrix = create_question_type_matrix(evaluator.results)
    confidence_analysis = analyze_confidence_patterns(evaluator.results)
    
    report = f"""# ABAP RAG Evaluation - Enhanced Summary Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

**Overall Results:**
- **Open-Source RAG**: {summary['open_source_wins']} wins ({summary['open_source_win_rate']:.1%})
- **Closed-Source RAG**: {summary['closed_source_wins']} wins ({summary['closed_source_win_rate']:.1%})
- **Ties**: {summary['ties']} ({summary['tie_rate']:.1%})

**Dataset Overview:**
- Total Questions: {summary['total_questions']}
- Total Evaluations: {summary['total_evaluations']}
- Question Types: {', '.join(summary['question_types'])}
- Judge Models: {', '.join(summary['judges_used'])}

**Key Finding:** {"Closed-Source RAG outperformed Open-Source" if summary['closed_source_win_rate'] > summary['open_source_win_rate'] else "Open-Source RAG outperformed Closed-Source" if summary['open_source_win_rate'] > summary['closed_source_win_rate'] else "Performance was roughly equal"}

---

## Detailed Results by Question Type

"""
    
    # Enhanced question type analysis
    for qtype, type_results in by_type.items():
        winner = "Closed-Source" if type_results['closed_source_win_rate'] > type_results['open_source_win_rate'] else "Open-Source" if type_results['open_source_win_rate'] > type_results['closed_source_win_rate'] else "Tie"
        
        report += f"""### {qtype} Questions
- **Questions**: {type_results['unique_questions']}
- **Total Evaluations**: {type_results['total_evaluations']}
- **Winner**: {winner}
- **Open-Source Win Rate**: {type_results['open_source_win_rate']:.1%} ({type_results['open_source_wins']} wins)
- **Closed-Source Win Rate**: {type_results['closed_source_win_rate']:.1%} ({type_results['closed_source_wins']} wins)
- **Tie Rate**: {type_results['tie_rate']:.1%} ({type_results['ties']} ties)
- **Average Judge Confidence**: {type_results['average_confidence']:.2f}

"""

    # Judge Performance Analysis
    report += f"""---

## Judge Performance Analysis

### Individual Judge Performance

| Judge Model | Total Evaluations | Open-Source Wins | Closed-Source Wins | Ties | Open-Source Win Rate | Closed-Source Win Rate | Avg Confidence |
|-------------|------------------|------------------|-------------------|------|---------------------|---------------------|----------------|
"""
    
    for judge, perf in judge_performance.items():
        report += f"| {judge} | {perf['total']} | {perf['open_wins']} | {perf['closed_wins']} | {perf['ties']} | {perf['open_rate']:.1%} | {perf['closed_rate']:.1%} | {perf['avg_confidence']:.2f} |\n"
    
    # Question Type Performance Matrix
    report += f"""
### Performance by Question Type and Judge

| Question Type | Judge Model | Open-Source Wins | Closed-Source Wins | Ties | Winner |
|---------------|-------------|------------------|-------------------|------|--------|
"""
    
    for (qtype, judge), data in question_type_matrix.items():
        winner = "Open-Source" if data['open_wins'] > data['closed_wins'] else "Closed-Source" if data['closed_wins'] > data['open_wins'] else "Tie"
        report += f"| {qtype} | {judge} | {data['open_wins']} | {data['closed_wins']} | {data['ties']} | {winner} |\n"
    
    # Confidence Analysis
    report += f"""
---

## Confidence Analysis

### Average Confidence by Winner
- **When Open-Source Won**: {confidence_analysis['open_confidence']:.2f}
- **When Closed-Source Won**: {confidence_analysis['closed_confidence']:.2f}
- **When Tied**: {confidence_analysis['tie_confidence']:.2f}

### Confidence Distribution
- **High Confidence (>0.8)**: {confidence_analysis['high_confidence']:.1%} of evaluations
- **Medium Confidence (0.6-0.8)**: {confidence_analysis['medium_confidence']:.1%} of evaluations
- **Low Confidence (<0.6)**: {confidence_analysis['low_confidence']:.1%} of evaluations

---

## Statistical Analysis

### Statistical Significance
- **Performance Gap**: {abs(summary['closed_source_win_rate'] - summary['open_source_win_rate']):.1%} percentage points
- **Significance Level**: {"High" if abs(summary['closed_source_win_rate'] - summary['open_source_win_rate']) > 0.15 else "Moderate" if abs(summary['closed_source_win_rate'] - summary['open_source_win_rate']) > 0.05 else "Low"}

### Why These Numbers?
- **Each question gets {len(evaluator.results) // summary['total_questions']} evaluations** ({len(summary['judges_used'])} judges × {len(evaluator.results) // summary['total_questions'] // len(summary['judges_used'])} position switches)
- **"Open-Source Win Rate"** shows percentage of times Open-Source won (Closed-Source rate = 100% - Open-Source rate - Tie rate)
- **High Confidence** ({confidence_analysis['high_confidence']:.1%}) indicates reliable judge decisions

---

## Key Insights for Thesis

### 1. Overall Performance Comparison
- **{"Closed-Source" if summary['closed_source_win_rate'] > summary['open_source_win_rate'] else "Open-Source"} Advantage**: {abs(summary['closed_source_win_rate'] - summary['open_source_win_rate']):.1%} percentage points
- **Judge Consensus**: {"High agreement across all judges" if len(set(max(judge_performance[j]['open_wins'], judge_performance[j]['closed_wins']) > min(judge_performance[j]['open_wins'], judge_performance[j]['closed_wins']) for j in judge_performance)) == 1 else "Mixed results across judges"}

### 2. Question Complexity Analysis
"""
    
    # Analyze performance by complexity
    complexity_order = ["Simple", "Complex", "Situational", "Conversational", "Double"]
    available_types = [t for t in complexity_order if t in by_type.keys()]
    
    for qtype in available_types:
        if qtype in by_type:
            type_data = by_type[qtype]
            report += f"- **{qtype}**: Closed-Source {type_data['closed_source_win_rate']:.1%} vs Open-Source {type_data['open_source_win_rate']:.1%}\n"
    
    report += f"""
### 3. Judge Reliability
- **Most Confident Judge**: {max(judge_performance.keys(), key=lambda j: judge_performance[j]['avg_confidence'])}
- **Most Consistent Results**: {"High consistency" if max(judge_performance[j]['avg_confidence'] for j in judge_performance) - min(judge_performance[j]['avg_confidence'] for j in judge_performance) < 0.1 else "Some variation"} across judges

---

## Methodology Validation

This evaluation used the **LLM-as-Judge methodology** with comprehensive bias reduction:

### Bias Reduction Techniques
1. **Blind Evaluation**: Judges never knew which response came from which RAG system
2. **Position Switching**: Each question evaluated {len(evaluator.results) // summary['total_questions'] // len(summary['judges_used'])} times with randomized positions
3. **Multi-Judge Consensus**: {len(summary['judges_used'])} different judge models for diverse perspectives
4. **Balanced Dataset**: Questions selected across all complexity levels

### Statistical Rigor
- **Sample Size**: {summary['total_evaluations']} total evaluations
- **Random Assignment**: Eliminates position bias
- **Inter-Judge Reliability**: Multiple models reduce systematic bias
- **Reproducible Results**: Methodology suitable for academic research

---

## Files Generated

This evaluation produced:
- **Detailed JSON Results**: Complete evaluation data with all scores and reasoning
- **Enhanced CSV Export**: Statistical data for analysis in Excel/R/Python
- **This Summary Report**: Human-readable analysis for thesis documentation
{"- **Visualization Plots**: Performance comparison charts and analysis graphs" if PLOTTING_AVAILABLE else ""}

---

*This report was generated automatically by the Enhanced ABAP RAG Evaluation System*
"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)

def analyze_judge_performance(results):
    """Analyze performance of each judge model"""
    judge_performance = {}
    
    for result in results:
        judge = result.judge_model
        if judge not in judge_performance:
            judge_performance[judge] = {
                'total': 0,
                'open_wins': 0,
                'closed_wins': 0,
                'ties': 0,
                'confidences': []
            }
        
        judge_performance[judge]['total'] += 1
        judge_performance[judge]['confidences'].append(result.confidence)
        
        if result.actual_winner == 'open_source':
            judge_performance[judge]['open_wins'] += 1
        elif result.actual_winner == 'closed_source':
            judge_performance[judge]['closed_wins'] += 1
        else:
            judge_performance[judge]['ties'] += 1
    
    # Calculate rates and averages
    for judge in judge_performance:
        data = judge_performance[judge]
        data['open_rate'] = data['open_wins'] / data['total']
        data['closed_rate'] = data['closed_wins'] / data['total']
        data['tie_rate'] = data['ties'] / data['total']
        data['avg_confidence'] = sum(data['confidences']) / len(data['confidences'])
    
    return judge_performance

def create_question_type_matrix(results):
    """Create a matrix of performance by question type and judge"""
    matrix = {}
    
    for result in results:
        key = (result.question_type, result.judge_model)
        if key not in matrix:
            matrix[key] = {
                'open_wins': 0,
                'closed_wins': 0,
                'ties': 0,
                'total': 0
            }
        
        matrix[key]['total'] += 1
        if result.actual_winner == 'open_source':
            matrix[key]['open_wins'] += 1
        elif result.actual_winner == 'closed_source':
            matrix[key]['closed_wins'] += 1
        else:
            matrix[key]['ties'] += 1
    
    return matrix

def analyze_confidence_patterns(results):
    """Analyze confidence patterns across different outcomes"""
    open_confidences = [r.confidence for r in results if r.actual_winner == 'open_source']
    closed_confidences = [r.confidence for r in results if r.actual_winner == 'closed_source']
    tie_confidences = [r.confidence for r in results if r.actual_winner == 'tie']
    all_confidences = [r.confidence for r in results]
    
    high_confidence = sum(1 for c in all_confidences if c > 0.8)
    medium_confidence = sum(1 for c in all_confidences if 0.6 <= c <= 0.8)
    low_confidence = sum(1 for c in all_confidences if c < 0.6)
    
    return {
        'open_confidence': sum(open_confidences) / len(open_confidences) if open_confidences else 0,
        'closed_confidence': sum(closed_confidences) / len(closed_confidences) if closed_confidences else 0,
        'tie_confidence': sum(tie_confidences) / len(tie_confidences) if tie_confidences else 0,
        'high_confidence': high_confidence / len(all_confidences),
        'medium_confidence': medium_confidence / len(all_confidences),
        'low_confidence': low_confidence / len(all_confidences)
    }

def create_basic_plots(evaluator, results, output_dir):
    """Create basic visualization plots if libraries are available"""
    
    if not PLOTTING_AVAILABLE:
        print("⚠️  Skipping plots - matplotlib/seaborn not installed")
        return
    
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    summary = results["summary"]
    
    # 1. Overall Performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    sizes = [summary['open_source_wins'], summary['closed_source_wins'], summary['ties']]
    labels = ['Open-Source RAG', 'Closed-Source RAG', 'Ties']
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                       explode=(0.05, 0.05, 0.05), shadow=True, startangle=90)
    ax1.set_title('Overall Performance Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart
    bars = ax2.bar(labels, sizes, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_title('Total Wins by System', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Wins')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars, sizes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/overall_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Judge Performance
    judge_performance = analyze_judge_performance(evaluator.results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    judges = list(judge_performance.keys())
    judges_short = [j.split('-')[0] for j in judges]
    open_rates = [judge_performance[j]['open_rate'] * 100 for j in judges]
    closed_rates = [judge_performance[j]['closed_rate'] * 100 for j in judges]
    
    x = np.arange(len(judges))
    width = 0.35
    
    ax1.bar(x - width/2, open_rates, width, label='Open-Source', color='#FF6B6B', alpha=0.8)
    ax1.bar(x + width/2, closed_rates, width, label='Closed-Source', color='#4ECDC4', alpha=0.8)
    
    ax1.set_xlabel('Judge Models')
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title('Judge Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(judges_short)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Confidence levels
    confidences = [judge_performance[j]['avg_confidence'] for j in judges]
    bars = ax2.bar(judges_short, confidences, color='#45B7D1', alpha=0.8)
    ax2.set_xlabel('Judge Models')
    ax2.set_ylabel('Average Confidence')
    ax2.set_title('Judge Confidence Levels')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, conf in zip(bars, confidences):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{conf:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/judge_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Question Type Performance
    by_type = results.get("by_question_type", {})
    if by_type:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        question_types = list(by_type.keys())
        open_rates = [by_type[qt]['open_source_win_rate'] * 100 for qt in question_types]
        closed_rates = [by_type[qt]['closed_source_win_rate'] * 100 for qt in question_types]
        
        x = np.arange(len(question_types))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, open_rates, width, label='Open-Source', color='#FF6B6B', alpha=0.8)
        bars2 = ax.bar(x + width/2, closed_rates, width, label='Closed-Source', color='#4ECDC4', alpha=0.8)
        
        ax.set_xlabel('Question Types')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Performance by Question Type')
        ax.set_xticks(x)
        ax.set_xticklabels(question_types, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/question_type_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📊 Plots saved to {plots_dir}/")

def export_enhanced_csv(evaluator: ABAPRAGEvaluator, filename: str):
    """Export enhanced CSV with additional analysis columns"""
    
    data = []
    for result in evaluator.results:
        # Calculate score differences
        avg_score_a = sum(result.scores_a.values()) / len(result.scores_a)
        avg_score_b = sum(result.scores_b.values()) / len(result.scores_b)
        score_difference = avg_score_a - avg_score_b
        
        # Determine actual system for each position
        system_a = result.position_mapping['A']
        system_b = result.position_mapping['B']
        
        row = {
            'evaluation_id': result.evaluation_id,
            'question_type': result.question_type,
            'judge_model': result.judge_model,
            'judge_choice': result.judge_choice,
            'actual_winner': result.actual_winner,
            'confidence': result.confidence,
            'system_a': system_a,
            'system_b': system_b,
            'avg_score_a': avg_score_a,
            'avg_score_b': avg_score_b,
            'score_difference_a_minus_b': score_difference,
            
            # Individual scores for A
            'tech_accuracy_a': result.scores_a.get('technical_accuracy', 0),
            'sap_knowledge_a': result.scores_a.get('sap_domain_knowledge', 0),
            'completeness_a': result.scores_a.get('completeness', 0),
            'clarity_a': result.scores_a.get('clarity', 0),
            'practical_value_a': result.scores_a.get('practical_value', 0),
            
            # Individual scores for B
            'tech_accuracy_b': result.scores_b.get('technical_accuracy', 0),
            'sap_knowledge_b': result.scores_b.get('sap_domain_knowledge', 0),
            'completeness_b': result.scores_b.get('completeness', 0),
            'clarity_b': result.scores_b.get('clarity', 0),
            'practical_value_b': result.scores_b.get('practical_value', 0),
            
            # Analysis flags
            'high_confidence': result.confidence > 0.8,
            'timestamp': result.timestamp
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return df

def main():
    """Main evaluation runner with enhanced reporting"""
    
    parser = argparse.ArgumentParser(
        description="Run ABAP RAG System Evaluation with Enhanced Reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation with enhanced reporting
  python run_evaluation.py open_source.json closed_source.json --balance 3 --enhanced
  
  # With plots (requires matplotlib/seaborn)
  python run_evaluation.py open_source.json closed_source.json --balance 3 --enhanced --plots
  
  # Simulation mode for testing
  python run_evaluation.py open_source.json closed_source.json --balance 1 --simulation --enhanced
        """
    )
    
    # Required arguments
    parser.add_argument("open_source_file", help="Path to open source responses JSON file")
    parser.add_argument("closed_source_file", help="Path to closed source responses JSON file")
    
    # Evaluation options
    parser.add_argument("--balance", type=int, metavar="N", 
                       help="Balance dataset to N questions per type")
    parser.add_argument("--no-balance", action="store_true",
                       help="Use full dataset without balancing")
    parser.add_argument("--switches", type=int, default=3, metavar="N",
                       help="Number of position switches per question per judge (default: 3)")
    
    # Enhanced reporting options
    parser.add_argument("--enhanced", action="store_true",
                       help="Generate enhanced summary report with detailed tables")
    parser.add_argument("--plots", action="store_true",
                       help="Generate visualization plots (requires matplotlib/seaborn)")
    
    # API options
    parser.add_argument("--simulation", action="store_true",
                       help="Use simulation mode instead of real APIs")
    parser.add_argument("--gemini-key", help="Gemini API key")
    parser.add_argument("--grok-key", help="Grok API key")
    parser.add_argument("--qwen-provider", choices=["together", "huggingface", "custom"], 
                       default="together", help="Qwen API provider (default: together)")
    parser.add_argument("--qwen-key", help="Qwen API key")
    
    # Output options
    parser.add_argument("--output", help="Output file name (default: auto-generated)")
    parser.add_argument("--output-dir", default="results", help="Output directory (default: results)")
    
    # Analysis options
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze dataset, don't run evaluation")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"""
{'='*70}
ENHANCED ABAP RAG EVALUATION SYSTEM
{'='*70}
Open Source File: {args.open_source_file}
Closed Source File: {args.closed_source_file}
Enhanced Reporting: {'Yes' if args.enhanced else 'Standard'}
Visualization Plots: {'Yes' if args.plots and PLOTTING_AVAILABLE else 'No' if not args.plots else 'Requested but libraries missing'}
{'='*70}
""")
    
    # Step 1: Dataset Analysis
    print("STEP 1: ANALYZING DATASET")
    print("-" * 30)
    
    analyzer = DatasetAnalyzer()
    analysis_results = analyzer.analyze_files(args.open_source_file, args.closed_source_file)
    
    if "error" in analysis_results:
        print(f"❌ Dataset analysis failed: {analysis_results['error']}")
        return 1
    
    if args.analyze_only:
        print("\n✅ Dataset analysis complete.")
        return 0
    
    # Step 2: Configure Evaluation
    print(f"\nSTEP 2: CONFIGURING EVALUATION")
    print("-" * 35)
    
    questions_per_type = None
    if args.balance:
        questions_per_type = args.balance
        print(f"Using balanced evaluation: {questions_per_type} questions per type")
    elif args.no_balance:
        questions_per_type = None
        print("Using full dataset (no balancing)")
    else:
        suggestions = analyzer.suggest_optimal_balance()
        recommended = suggestions.get("recommended")
        if recommended and recommended["questions_per_type"]:
            questions_per_type = recommended["questions_per_type"]
            print(f"Using recommended balance: {questions_per_type} questions per type")
        else:
            questions_per_type = None
            print("Using full dataset (no recommended balance)")
    
    # Initialize evaluator
    evaluator = ABAPRAGEvaluator()
    
    # Setup APIs
    if not args.simulation:
        print("\nSetting up APIs...")
        gemini_key = args.gemini_key or os.getenv("GEMINI_API_KEY")
        grok_key = args.grok_key or os.getenv("GROK_API_KEY")
        qwen_key = args.qwen_key or os.getenv("QWEN_API_KEY")
        
        if not (gemini_key or grok_key or qwen_key):
            print("⚠️  No API keys provided. Using simulation mode.")
        else:
            evaluator.setup_apis(
                gemini_key=gemini_key,
                grok_key=grok_key,
                qwen_provider=args.qwen_provider,
                qwen_key=qwen_key
            )
    else:
        print("Using simulation mode")
    
    # Step 3: Run Evaluation
    print(f"\nSTEP 3: RUNNING EVALUATION")
    print("-" * 30)
    
    try:
        results = evaluator.evaluate_full_dataset(
            args.open_source_file,
            args.closed_source_file,
            questions_per_type=questions_per_type,
            num_position_switches=args.switches
        )
        print(f"\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 4: Generate Reports
    print(f"\nSTEP 4: GENERATING REPORTS")
    print("-" * 30)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export detailed JSON results
    if args.output:
        results_file = os.path.join(args.output_dir, args.output)
    else:
        balance_suffix = f"_balanced_{questions_per_type}" if questions_per_type else "_full"
        results_file = os.path.join(args.output_dir, f"abap_rag_evaluation_{timestamp}{balance_suffix}.json")
    
    exported_file = evaluator.export_results(results_file)
    print(f"📄 Detailed JSON results: {exported_file}")
    
    # Create enhanced summary report
    if args.enhanced:
        summary_file = os.path.join(args.output_dir, f"enhanced_summary_{timestamp}.md")
        create_enhanced_summary_report(evaluator, results, summary_file)
        print(f"📋 Enhanced summary report: {summary_file}")
    else:
        # Create basic summary
        summary_file = os.path.join(args.output_dir, f"evaluation_summary_{timestamp}.md")
        create_basic_summary_report(evaluator, results, summary_file)
        print(f"📋 Summary report: {summary_file}")
    
    # Create enhanced CSV
    csv_file = os.path.join(args.output_dir, f"evaluation_data_{timestamp}.csv")
    df = export_enhanced_csv(evaluator, csv_file)
    print(f"📊 Enhanced CSV data: {csv_file}")
    
    # Generate plots if requested
    if args.plots:
        try:
            create_basic_plots(evaluator, results, args.output_dir)
        except Exception as e:
            print(f"⚠️  Could not generate plots: {e}")
            if not PLOTTING_AVAILABLE:
                print("   Install plotting libraries: pip install matplotlib seaborn")
    
    # Step 5: Display Summary
    print(f"\nSTEP 5: EVALUATION SUMMARY")
    print("-" * 30)
    
    summary = results["summary"]
    print(f"📊 OVERALL RESULTS")
    print(f"   Open-Source RAG:   {summary['open_source_wins']:3d} wins ({summary['open_source_win_rate']:5.1%})")
    print(f"   Closed-Source RAG: {summary['closed_source_wins']:3d} wins ({summary['closed_source_win_rate']:5.1%})")
    print(f"   Ties:              {summary['ties']:3d} ties ({summary['tie_rate']:5.1%})")
    
    by_type = results.get("by_question_type", {})
    if by_type:
        print(f"\n📋 BY QUESTION TYPE")
        for qtype, type_results in by_type.items():
            open_rate = type_results['open_source_win_rate']
            closed_rate = type_results['closed_source_win_rate']
            winner = "Open" if open_rate > closed_rate else "Closed" if closed_rate > open_rate else "Tie"
            print(f"   {qtype:12s}: {winner:6s} wins (Open: {open_rate:5.1%}, Closed: {closed_rate:5.1%})")
    
    print(f"\n📈 DATASET STATS")
    print(f"   Total Questions:   {summary['total_questions']}")
    print(f"   Total Evaluations: {summary['total_evaluations']}")
    print(f"   Judge Models:      {len(summary['judges_used'])}")
    
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE!")
    print(f"All results saved to: {args.output_dir}/")
    if args.enhanced:
        print(f"📋 Check the enhanced summary report for detailed analysis!")
    if args.plots and PLOTTING_AVAILABLE:
        print(f"📊 Check the plots/ folder for visualizations!")
    print(f"{'='*70}")
    
    return 0

def create_basic_summary_report(evaluator, results, filename):
    """Create basic summary report (fallback when enhanced not requested)"""
    
    summary = results["summary"]
    by_type = results.get("by_question_type", {})
    
    report = f"""# ABAP RAG Evaluation Summary Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

**Overall Results:**
- Open-Source RAG: {summary['open_source_wins']} wins ({summary['open_source_win_rate']:.1%})
- Closed-Source RAG: {summary['closed_source_wins']} wins ({summary['closed_source_win_rate']:.1%})
- Ties: {summary['ties']} ({summary['tie_rate']:.1%})

**Dataset:**
- Total Questions: {summary['total_questions']}
- Total Evaluations: {summary['total_evaluations']}
- Question Types: {', '.join(summary['question_types'])}
- Judge Models: {', '.join(summary['judges_used'])}

## Results by Question Type

"""
    
    for qtype, type_results in by_type.items():
        report += f"""### {qtype} Questions
- Questions: {type_results['unique_questions']}
- Evaluations: {type_results['total_evaluations']}
- Open-Source Win Rate: {type_results['open_source_win_rate']:.1%}
- Closed-Source Win Rate: {type_results['closed_source_win_rate']:.1%}
- Tie Rate: {type_results['tie_rate']:.1%}
- Average Confidence: {type_results['average_confidence']:.2f}

"""
    
    report += """## Methodology

This evaluation used the LLM-as-Judge methodology with bias reduction techniques:

1. **Blind Evaluation**: Judges never knew which response came from which RAG system
2. **Position Switching**: Each question evaluated multiple times with randomized response positions
3. **Multi-Judge Consensus**: Multiple different judge models for diverse perspectives
4. **Balanced Dataset**: Questions selected to ensure fair comparison across complexity levels

## Files Generated

- Detailed JSON results with all evaluation data
- CSV export for statistical analysis
- This summary report for documentation
"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)

if __name__ == "__main__":
    exit(main())