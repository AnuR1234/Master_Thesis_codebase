import os
import shutil
import argparse
import re
import sys

def rename_abap_files(source_dir, target_dir=None, pattern=".prog.abap", dry_run=False):
    """
    Rename or copy ABAP files from source_dir to target_dir, changing extensions
    from .prog.abap to .abap.
    
    Args:
        source_dir (str): Directory containing the source files
        target_dir (str, optional): Directory to save renamed files. If None, files are renamed in place.
        pattern (str): File pattern to look for (default: ".prog.abap")
        dry_run (bool): If True, only show what would be done without making changes
    
    Returns:
        int: Number of files processed
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return 0
    
    if target_dir and not os.path.exists(target_dir):
        if not dry_run:
            os.makedirs(target_dir)
            print(f"Created target directory: {target_dir}")
        else:
            print(f"Would create target directory: {target_dir}")
    
    # Find all files with the specified pattern
    abap_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(pattern):
                abap_files.append(os.path.join(root, file))
    
    if not abap_files:
        print(f"No files with pattern '{pattern}' found in {source_dir}")
        return 0
    
    print(f"Found {len(abap_files)} files with pattern '{pattern}'")
    
    # Process each file
    processed_count = 0
    for old_path in abap_files:
        # Determine new filename
        filename = os.path.basename(old_path)
        new_filename = filename.replace(pattern, ".abap")
        
        # Determine new path
        if target_dir:
            rel_path = os.path.relpath(old_path, source_dir)
            # Extract just the directory part, without the filename
            rel_dir = os.path.dirname(rel_path)
            target_subdir = os.path.join(target_dir, rel_dir)
            new_path = os.path.join(target_subdir, new_filename)
            
            # Create subdirectories if needed
            if not os.path.exists(target_subdir) and not dry_run:
                os.makedirs(target_subdir, exist_ok=True)
        else:
            # In-place rename
            new_path = os.path.join(os.path.dirname(old_path), new_filename)
        
        # Copy or rename the file
        if dry_run:
            print(f"Would {'rename' if not target_dir else 'copy'}: {old_path} to {new_path}")
        else:
            try:
                if target_dir:
                    # Copy to new location
                    shutil.copy2(old_path, new_path)
                    print(f"Copied: {old_path} to {new_path}")
                else:
                    # Rename in place
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} to {new_path}")
                processed_count += 1
            except Exception as e:
                print(f"Error processing {old_path}: {e}")
    
    return processed_count

def main():
    parser = argparse.ArgumentParser(description="Rename ABAP files from .prog.abap to .abap")
    parser.add_argument("source_dir", help="Directory containing the source files")
    parser.add_argument("--target-dir", help="Directory to save renamed files (if omitted, renames in place)")
    parser.add_argument("--pattern", default=".prog.abap", help="File pattern to look for (default: .prog.abap)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    
    # Add option to process other ABAP file types
    parser.add_argument("--all-types", action="store_true", 
                        help="Process all ABAP type extensions (clas.abap, fugr.abap, etc)")
    
    args = parser.parse_args()
    
    # Process .prog.abap files
    count = rename_abap_files(args.source_dir, args.target_dir, args.pattern, args.dry_run)
    
    # Process other ABAP file types if requested
    if args.all_types:
        other_patterns = [".clas.abap", ".fugr.abap", ".intf.abap", ".dtel.abap", ".tabl.abap"]
        for pattern in other_patterns:
            print(f"\nProcessing files with pattern: {pattern}")
            count += rename_abap_files(args.source_dir, args.target_dir, pattern, args.dry_run)
    
    print(f"\nTotal files processed: {count}")
    
    # Show user instructions
    if count > 0 and not args.dry_run:
        print("\n=== Next Steps ===")
        if args.target_dir:
            print(f"Files have been copied to: {args.target_dir}")
            print("Use these files with your chunking pipeline:")
            print(f"python chunking_main.py --code-dir {args.target_dir} [other options]")
        else:
            print("Files have been renamed in place.")
            print("You can now run your chunking pipeline:")
            print(f"python chunking_main.py --code-dir {args.source_dir} [other options]")

if __name__ == "__main__":
    main()