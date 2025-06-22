def sanitize_for_json(text):
    """
    Sanitize a string to ensure it can be properly encoded in JSON.
    
    Args:
        text: String to sanitize
        
    Returns:
        Sanitized string
    """
    if not isinstance(text, str):
        return text
    
    # Replace control characters that might break JSON encoding
    replacements = {
        '\n': '\\n',
        '\r': '\\r',
        '\t': '\\t',
        '\b': '\\b',
        '\f': '\\f',
        '"': '\\"',
        '\\': '\\\\'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Replace any remaining control characters that could break JSON
    result = ""
    for char in text:
        if ord(char) < 32:  # Control characters
            result += f"\\u{ord(char):04x}"
        else:
            result += char
    
    return result#!/usr/bin/env python3
"""
Script to process RAG evaluation results and format them into the desired output format.

This script:
1. Loads a JSON file with RAG evaluation results
2. Extracts the required fields (Question, response, question_type, latency)
3. Combines all retrieved_docs into a single retrieved_context field
4. Includes only the filename field from retrieved_docs
5. Saves the processed results to a new JSON file
"""

import json
import argparse
import sys
from typing import Dict, List, Any


def process_rag_results(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process RAG results and format them according to requirements
    
    Args:
        data: Input data containing RAG results
        
    Returns:
        Processed list of question entries in the desired format
    """
    processed_results = []
    
    # Extract the results array
    if "results" in data:
        results = data["results"]
    else:
        # If the data is already an array or doesn't have a results field
        results = data if isinstance(data, list) else [data]
    
    # Process each question entry
    for entry in results:
        # Skip if entry doesn't contain a Question
        if "Question" not in entry:
            continue
            
        # Create a new entry with only the required fields
        new_entry = {
            "Question": entry.get("Question", ""),
            "response": entry.get("response", ""),
            "question_type": entry.get("question_type", ""),
            "latency": entry.get("latency", 0.0)
        }
        
        # Get retrieved_docs
        retrieved_docs = entry.get("retrieved_docs", [])
        
        # Extract filenames
        filenames = []
        for doc in retrieved_docs:
            filename = doc.get("filename", "")
            if filename and filename not in filenames:
                filenames.append(filename)
        
        # Join filenames with comma if multiple
        new_entry["filename"] = ", ".join(filenames) if filenames else ""
        
        # Combine docs into retrieved_context
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            # Extract document information
            doc_id = i
            title = doc.get("title", "No Title")
            text = doc.get("text", "")
            code_snippet = doc.get("code_snippet", "")
            
            # Format document header
            doc_header = f"Document {doc_id}: Title: {title}"
            
            # Combine text and code snippet
            doc_content = text
            if code_snippet and code_snippet.strip():
                if doc_content:
                    doc_content += f"\n\nCode Snippet:\n{code_snippet}"
                else:
                    doc_content = f"Code Snippet:\n{code_snippet}"
            
            # Create the complete document section
            if doc_content:
                context_parts.append(f"{doc_header}\n\n{doc_content}")
            else:
                context_parts.append(doc_header)
        
        # Join all documents with double newlines
        new_entry["retrieved_context"] = "\n\n".join(context_parts)
        
        processed_results.append(new_entry)
    
    return processed_results


def process_json_file(input_file: str, output_file: str):
    """
    Process the JSON file and create output in the desired format
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
    """
    try:
        # Load input JSON
        print(f"Loading input file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Processing RAG results...")
        
        # Process the data
        processed_results = process_rag_results(data)
        
        # Save output JSON - Use json.dumps and json.loads to verify
        print(f"Saving output file: {output_file}")
        
        # Write JSON directly using Python's built-in tools
        # This ensures proper escaping of all characters
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_results, f, indent=2, ensure_ascii=False)
            
        # Verify file is readable
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                test_load = json.load(f)
                print(f"Successfully verified JSON output (loaded {len(test_load)} entries)")
        except json.JSONDecodeError as e:
            print(f"Warning: Generated JSON file may have issues: {e}")
            
        print(f"Successfully processed {len(processed_results)} questions")
        print(f"Output saved to: {output_file}")
        
        # Print sample of first entry for verification
        if processed_results:
            first_entry = processed_results[0]
            print(f"\nSample processed entry:")
            print(f"Question: {first_entry.get('Question', '')[:50]}...")
            print(f"Response: {first_entry.get('response', '')[:50]}...")
            print(f"Question Type: {first_entry.get('question_type', '')}")
            print(f"Latency: {first_entry.get('latency', 0)}")
            print(f"Filename: {first_entry.get('filename', '')}")
            # For retrieved context, show just the beginning without newlines
            context = first_entry.get('retrieved_context', '')
            if context:
                # Display for readability in terminal
                context_preview = context.replace("\\n", " ").replace("\\r", " ")[:200]
                print(f"Retrieved Context (first 200 chars): {context_preview}...")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Process RAG evaluation results into the desired format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_rag_results.py input.json output.json
  python process_rag_results.py rag_results.json processed_results.json
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Path to input JSON file with RAG evaluation results"
    )
    
    parser.add_argument(
        "output_file", 
        help="Path to output JSON file with processed results"
    )
    
    args = parser.parse_args()
    
    # Process the file
    process_json_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()