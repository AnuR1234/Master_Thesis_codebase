#!/usr/bin/env python3
"""
Script to combine retrieved_docs into a single retrieved_context field

This script:
1. Loads a JSON file with RAG evaluation results
2. Combines all retrieved_docs into a single retrieved_context field
3. Keeps all other fields unchanged
4. Saves the modified results to a new JSON file
"""

import json
import argparse
import sys
from typing import Dict, List, Any


def combine_retrieved_docs(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Combine retrieved_docs into retrieved_context for each entry
    Remove expected_response and retrieved_docs, keep only filename from retrieved_docs
    
    Args:
        data: List of evaluation result dictionaries
        
    Returns:
        Modified list with retrieved_context field added and unwanted fields removed
    """
    modified_data = []
    
    for entry in data:
        # Create a copy of the entry
        new_entry = entry.copy()
        
        # Get retrieved_docs
        retrieved_docs = entry.get("retrieved_docs", [])
        
        # Extract filenames for the filename field
        filenames = []
        for doc in retrieved_docs:
            filename = doc.get("filename", "")
            if filename and filename not in filenames:
                filenames.append(filename)
        
        # Combine docs into retrieved_context
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            # Extract document information
            doc_id = doc.get("id", "Unknown")
            title = doc.get("title", "No Title")
            text = doc.get("text", "")
            code_snippet = doc.get("code_snippet", "")
            
            # Format document header
            doc_header = f"Document {doc_id}: Title: {title}"
            
            # Combine text and code snippet if both exist
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
        retrieved_context = "\n\n".join(context_parts)
        
        # Add retrieved_context to the entry
        new_entry["retrieved_context"] = retrieved_context
        
        # Add filename as a separate field (join multiple filenames with comma if more than one)
        new_entry["filename"] = ", ".join(filenames) if filenames else ""
        
        # Remove unwanted fields
        if "expected_response" in new_entry:
            del new_entry["expected_response"]
        if "retrieved_docs" in new_entry:
            del new_entry["retrieved_docs"]
        
        modified_data.append(new_entry)
    
    return modified_data


def process_json_file(input_file: str, output_file: str):
    """
    Process the JSON file and add retrieved_context field
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
    """
    try:
        # Load input JSON
        print(f"Loading input file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different input formats
        if isinstance(data, dict):
            # Single entry
            data = [data]
        elif not isinstance(data, list):
            raise ValueError("Input JSON must be a list of objects or a single object")
        
        print(f"Processing {len(data)} entries...")
        
        # Combine retrieved docs
        modified_data = combine_retrieved_docs(data)
        
        # Save output JSON
        print(f"Saving output file: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(modified_data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully processed {len(modified_data)} entries")
        print(f"Removed 'expected_response' and 'retrieved_docs' fields")
        print(f"Added 'retrieved_context' and 'filename' fields")
        print(f"Output saved to: {output_file}")
        
        # Print sample of first entry for verification
        if modified_data:
            first_context = modified_data[0].get("retrieved_context", "")
            first_filename = modified_data[0].get("filename", "")
            print(f"\nSample filename: {first_filename}")
            print(f"Sample retrieved_context (first 200 chars):")
            print(f"{first_context[:200]}...")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Combine retrieved_docs into retrieved_context field",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python combine_docs.py input.json output.json
  python combine_docs.py rag_results.json rag_results_combined.json
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Path to input JSON file with retrieved_docs"
    )
    
    parser.add_argument(
        "output_file", 
        help="Path to output JSON file with retrieved_context"
    )
    
    args = parser.parse_args()
    
    # Process the file
    process_json_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()