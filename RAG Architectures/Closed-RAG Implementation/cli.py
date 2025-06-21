"""
Command-line interface for the RAG pipeline
"""
import asyncio
import argparse
import logging
import json
import sys
from typing import Optional, Dict, Any, List

from config import (
    USE_HYBRID_DEFAULT,
    USE_RERANKER_DEFAULT,
    DEFAULT_TOP_K
)
from pipeline import RAGPipeline
from eval import RAGEvaluator, create_test_case

# Configure logging
logger = logging.getLogger(__name__)

async def process_single_query(query: str,
                             use_hybrid: bool = USE_HYBRID_DEFAULT,
                             use_reranker: bool = USE_RERANKER_DEFAULT,
                             top_k: int = DEFAULT_TOP_K,
                             output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a single query and display the result
    
    Args:
        query: The query to process
        use_hybrid: Whether to use hybrid retrieval
        use_reranker: Whether to use reranker
        top_k: Number of top contexts to use
        output_file: Optional file to save results to
        
    Returns:
        Result dictionary
    """
    print(f"\nProcessing query: {query}")
    print(f"Using hybrid retrieval: {use_hybrid}")
    print(f"Using reranker: {use_reranker}")
    print(f"Top contexts: {top_k}")
    print("\nRetrieving information...")
    
    # Initialize and use the pipeline
    rag_pipeline = RAGPipeline()
    result = await rag_pipeline.process_query(
        query=query,
        use_hybrid=use_hybrid,
        use_reranker=use_reranker,
        top_k=top_k
    )
    
    # Display the result
    print("\n" + "="*50)
    print("GENERATED RESPONSE:")
    print("="*50)
    print(result["response"])
    print("\n" + "="*50)
    print(f"Based on {result['context_count']} documents")
    print("="*50)
    
    # Display the contexts briefly
    print("\nReference documents:")
    for i, ctx in enumerate(result.get("contexts", [])):
        print(f"  {i+1}. {ctx.get('title', 'Untitled')} (Score: {ctx.get('score', 0):.4f})")
    
    # Save to file if requested
    if output_file:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nComplete result saved to {output_file}")
    
    return result

async def run_batch_queries(queries_file: str, 
                           output_file: str = "batch_results.json",
                           use_hybrid: bool = USE_HYBRID_DEFAULT,
                           use_reranker: bool = USE_RERANKER_DEFAULT,
                           top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """
    Run a batch of queries from a file
    
    Args:
        queries_file: File containing queries (one per line)
        output_file: File to save results to
        use_hybrid: Whether to use hybrid retrieval
        use_reranker: Whether to use reranker
        top_k: Number of top contexts to use
        
    Returns:
        List of result dictionaries
    """
    # Read queries from file
    with open(queries_file, "r") as f:
        queries = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(queries)} queries from {queries_file}")
    print(f"Using hybrid retrieval: {use_hybrid}")
    print(f"Using reranker: {use_reranker}")
    print(f"Top contexts: {top_k}")
    
    # Initialize pipeline
    rag_pipeline = RAGPipeline()
    results = []
    
    # Process each query
    for i, query in enumerate(queries):
        print(f"\nProcessing query {i+1}/{len(queries)}: {query}")
        
        result = await rag_pipeline.process_query(
            query=query,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
            top_k=top_k
        )
        
        results.append(result)
        print(f"Response: {result['response'][:100]}...")
    
    # Save all results to file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAll results saved to {output_file}")
    return results

async def run_evaluation(test_cases_file: str,
                       output_file: str = "evaluation_results.json") -> Dict[str, Any]:
    """
    Run evaluation on a set of test cases
    
    Args:
        test_cases_file: File containing test cases in JSON format
        output_file: File to save evaluation results to
        
    Returns:
        Evaluation results dictionary
    """
    # Load test cases from file
    with open(test_cases_file, "r") as f:
        test_cases = json.load(f)
    
    print(f"Running evaluation on {len(test_cases)} test cases from {test_cases_file}")
    
    # Initialize pipeline and evaluator
    rag_pipeline = RAGPipeline()
    evaluator = RAGEvaluator(rag_pipeline)
    
    # Run evaluation
    results = await evaluator.run_evaluation(test_cases)
    
    # Display aggregate metrics
    print("\nEvaluation Results:")
    print("="*50)
    for metric, value in results["aggregate_metrics"].items():
        print(f"{metric}: {value:.4f}")
    
    # Save results to file
    evaluator.save_evaluation_results(results, output_file)
    print(f"\nDetailed evaluation results saved to {output_file}")
    
    return results

async def interactive_mode():
    """
    Run the RAG pipeline in interactive mode
    """
    print("\nSAP ABAP Code Documentation RAG System")
    print("Type 'exit' or 'quit' to end the session")
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline()
    
    # Default settings
    settings = {
        "use_hybrid": USE_HYBRID_DEFAULT,
        "use_reranker": USE_RERANKER_DEFAULT,
        "top_k": DEFAULT_TOP_K
    }
    
    # Display current settings
    print("\nCurrent settings:")
    for key, value in settings.items():
        print(f"  {key}: {value}")
    
    # Interactive loop
    while True:
        print("\nEnter your question (or 'settings' to change settings, 'exit' to quit):")
        query = input("> ")
        
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        elif query.lower() == "settings":
            print("\nChange settings (press Enter to keep current value):")
            
            # Update use_hybrid
            use_hybrid_input = input(f"Use hybrid retrieval (y/n) [{settings['use_hybrid']}]: ")
            if use_hybrid_input.lower() in ["y", "yes"]:
                settings["use_hybrid"] = True
            elif use_hybrid_input.lower() in ["n", "no"]:
                settings["use_hybrid"] = False
            
            # Update use_reranker
            use_reranker_input = input(f"Use reranker (y/n) [{settings['use_reranker']}]: ")
            if use_reranker_input.lower() in ["y", "yes"]:
                settings["use_reranker"] = True
            elif use_reranker_input.lower() in ["n", "no"]:
                settings["use_reranker"] = False
            
            # Update top_k
            top_k_input = input(f"Number of top contexts [{settings['top_k']}]: ")
            if top_k_input.strip() and top_k_input.isdigit():
                settings["top_k"] = int(top_k_input)
            
            print("\nUpdated settings:")
            for key, value in settings.items():
                print(f"  {key}: {value}")
            
        elif query.strip():
            # Process the query
            result = await rag_pipeline.process_query(
                query=query,
                use_hybrid=settings["use_hybrid"],
                use_reranker=settings["use_reranker"],
                top_k=settings["top_k"]
            )
            
            # Display the response
            print("\n" + "="*50)
            print("RESPONSE:")
            print("="*50)
            print(result["response"])
            print("\n" + "="*50)
            
            # Display the contexts briefly
            print(f"Based on {result['context_count']} documents:")
            for i, ctx in enumerate(result.get("contexts", [])):
                print(f"  {i+1}. {ctx.get('title', 'Untitled')} (Score: {ctx.get('score', 0):.4f})")

async def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(description="RAG System for SAP ABAP Code Documentation")
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Single query command
    query_parser = subparsers.add_parser("query", help="Process a single query")
    query_parser.add_argument("--query", "-q", help="Query to process")
    query_parser.add_argument("--no-hybrid", action="store_true", help="Disable hybrid retrieval")
    query_parser.add_argument("--no-rerank", action="store_true", help="Disable reranker")
    query_parser.add_argument("--top", "-t", type=int, default=DEFAULT_TOP_K, help="Number of top contexts to use")
    query_parser.add_argument("--output", "-o", help="Output file for results (JSON)")
    
    # Batch processing command
    batch_parser = subparsers.add_parser("batch", help="Process a batch of queries from a file")
    batch_parser.add_argument("--file", "-f", required=True, help="File containing queries (one per line)")
    batch_parser.add_argument("--output", "-o", default="batch_results.json", help="Output file for results (JSON)")
    batch_parser.add_argument("--no-hybrid", action="store_true", help="Disable hybrid retrieval")
    batch_parser.add_argument("--no-rerank", action="store_true", help="Disable reranker")
    batch_parser.add_argument("--top", "-t", type=int, default=DEFAULT_TOP_K, help="Number of top contexts to use")
    
    # Evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the RAG system using test cases")
    eval_parser.add_argument("--file", "-f", required=True, help="JSON file containing test cases")
    eval_parser.add_argument("--output", "-o", default="evaluation_results.json", help="Output file for evaluation results")
    
    # Interactive mode command
    subparsers.add_parser("interactive", help="Run in interactive mode")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == "query":
        query = args.query
        if not query:
            print("Enter your question about SAP ABAP code:")
            query = input("> ")
        
        await process_single_query(
            query=query,
            use_hybrid=not args.no_hybrid,
            use_reranker=not args.no_rerank,
            top_k=args.top,
            output_file=args.output
        )
    
    elif args.command == "batch":
        await run_batch_queries(
            queries_file=args.file,
            output_file=args.output,
            use_hybrid=not args.no_hybrid,
            use_reranker=not args.no_rerank,
            top_k=args.top
        )
    
    elif args.command == "evaluate":
        await run_evaluation(
            test_cases_file=args.file,
            output_file=args.output
        )
    
    elif args.command == "interactive":
        await interactive_mode()
    
    else:
        # Default to interactive mode if no command specified
        print("No command specified, starting interactive mode.")
        await interactive_mode()

if __name__ == "__main__":
    asyncio.run(main())