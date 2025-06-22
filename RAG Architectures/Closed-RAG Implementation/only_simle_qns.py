import json

def filter_simple_questions(input_file, output_file):
    """
    Filter questions with Question_type 'simple' and save to a new JSON file.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to the output JSON file
    """
    try:
        # Read the input JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filter questions with Question_type 'simple'
        simple_questions = []
        
        # Check if data is a list
        if isinstance(data, list):
            for item in data:
                if item.get('Question_type') == 'Simple':
                    simple_questions.append(item)
        else:
            print("Warning: Input data is not a list format")
            return
        
        # Save filtered questions to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(simple_questions, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully filtered and saved {len(simple_questions)} simple questions to {output_file}")
        
        if len(simple_questions) == 0:
            print("Warning: No questions with Question_type 'simple' were found")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{input_file}'")
    except Exception as e:
        print(f"Error: {str(e)}")

# Main execution
if __name__ == "__main__":
    input_filename = "/home/user/Desktop/RAG_pipeline_claude_eval_v2/merged_output_file_reports1.json"
    output_filename = "/home/user/Desktop/RAG_pipeline_claude_eval_v2/simple_questions_only_reports1.json"
    
    print("Filtering questions with Question_type 'simple'...")
    filter_simple_questions(input_filename, output_filename)