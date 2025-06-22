import json
import argparse
from typing import Dict, List, Any

def load_json_file(filepath: str) -> List[Dict[str, Any]]:
    """Load JSON file and return the data."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{filepath}': {e}")
        return []

def merge_datasets(golden_data: List[Dict], rag_data: List[Dict]) -> List[Dict]:
    """
    Merge golden dataset and RAG response dataset into the new format.
    
    Args:
        golden_data: List of entries from golden dataset
        rag_data: List of entries from RAG response dataset
    
    Returns:
        List of merged entries in the new format
    """
    merged_data = []
    
    # Create a lookup dictionary for RAG responses by question
    rag_lookup = {entry.get('Question', '').strip(): entry for entry in rag_data}
    
    # Debug: Print all RAG questions for verification
    print("DEBUG: First 3 RAG questions:")
    for i, entry in enumerate(rag_data[:3]):
        rag_q = entry.get('Question', '').strip()
        print(f"  {i+1}. '{rag_q}' (length: {len(rag_q)})")
    
    for golden_entry in golden_data:
        question = golden_entry.get('question', '').strip()
        
        # Debug: Print the current golden question being processed
        print(f"\nDEBUG: Processing golden question: '{question}' (length: {len(question)})")
        
        # Find corresponding RAG entry
        rag_entry = rag_lookup.get(question)
        
        # Debug: Check if match was found
        if rag_entry:
            print("DEBUG: ✓ Match found!")
        else:
            print("DEBUG: ✗ No match found!")
            # Try to find partial matches for debugging
            possible_matches = [k for k in rag_lookup.keys() if k.lower().startswith(question[:20].lower())]
            if possible_matches:
                print(f"DEBUG: Possible partial matches: {possible_matches[:2]}")
            else:
                print("DEBUG: No partial matches found either")
        
        if rag_entry:
            # Extract number of chunks referenced from retrieved_docs
            retrieved_docs = rag_entry.get('retrieved_docs', [])
            num_chunks = len(retrieved_docs) if retrieved_docs else 0
            
            # Create merged entry - using retrieved_context from RAG dataset
            merged_entry = {
                "Question": question,
                "reference_context": rag_entry.get('retrieved_context', ''),  # Changed: now from RAG dataset
                "golden_answer": golden_entry.get('reference_answer', ''),
                "RAG_response": rag_entry.get('response', ''),
                "Question_type": golden_entry.get('question_type', '').title(),
                "Difficulty": golden_entry.get('difficulty', 'Medium').title(),
                "Number_of_chunks_referenced": num_chunks,
                "Latency": rag_entry.get('latency', 0.0)
            }
            
            merged_data.append(merged_entry)
        else:
            print(f"Warning: No matching RAG response found for question: '{question[:50]}...'")
            
            # Create entry with missing RAG data
            merged_entry = {
                "Question": question,
                "reference_context": '',  # Changed: empty since no RAG entry found
                "golden_answer": golden_entry.get('reference_answer', ''),
                "RAG_response": None,
                "Question_type": golden_entry.get('question_type', '').title(),
                "Difficulty": golden_entry.get('difficulty', 'Medium').title(),
                "Number_of_chunks_referenced": 0,
                "Latency": None
            }
            
            merged_data.append(merged_entry)
    
    return merged_data

def save_merged_data(merged_data: List[Dict], output_filepath: str):
    """Save merged data to JSON file."""
    try:
        with open(output_filepath, 'w', encoding='utf-8') as file:
            json.dump(merged_data, file, indent=2, ensure_ascii=False)
        print(f"Successfully saved merged data to '{output_filepath}'")
    except Exception as e:
        print(f"Error saving file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Merge golden dataset and RAG response dataset')
    parser.add_argument('golden_file', help='Path to golden dataset JSON file')
    parser.add_argument('rag_file', help='Path to RAG response dataset JSON file')
    parser.add_argument('output_file', help='Path to output merged dataset JSON file')
    
    args = parser.parse_args()
    
    # Load the datasets
    print("Loading golden dataset...")
    golden_data = load_json_file(args.golden_file)
    
    print("Loading RAG response dataset...")
    rag_data = load_json_file(args.rag_file)
    
    if not golden_data or not rag_data:
        print("Error: Could not load one or both datasets. Exiting.")
        return
    
    print(f"Loaded {len(golden_data)} entries from golden dataset")
    print(f"Loaded {len(rag_data)} entries from RAG dataset")
    
    # Merge the datasets
    print("Merging datasets...")
    merged_data = merge_datasets(golden_data, rag_data)
    
    print(f"Successfully merged {len(merged_data)} entries")
    
    # Save the merged data
    save_merged_data(merged_data, args.output_file)
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Total merged entries: {len(merged_data)}")
    print(f"Entries with RAG responses: {sum(1 for entry in merged_data if entry['RAG_response'] is not None)}")
    print(f"Entries missing RAG responses: {sum(1 for entry in merged_data if entry['RAG_response'] is None)}")

if __name__ == "__main__":
    # Example usage if running without command line arguments
    import sys
    
    if len(sys.argv) == 1:
        print("Example usage:")
        print("python merge_datasets.py golden_dataset.json rag_dataset.json merged_output.json")
        print("\nAlternatively, you can modify the script to use hardcoded file paths:")
        print("golden_file = 'golden_dataset.json'")
        print("rag_file = 'rag_dataset.json'")
        print("output_file = 'merged_dataset.json'")
        
        # Uncomment the following lines to use hardcoded file paths
        # golden_data = load_json_file('golden_dataset.json')
        # rag_data = load_json_file('rag_dataset.json')
        # merged_data = merge_datasets(golden_data, rag_data)
        # save_merged_data(merged_data, 'merged_dataset.json')
    else:
        main()