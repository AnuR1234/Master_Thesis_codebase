#doc_chunker.py
import os
import re
import json
import logging
import pandas as pd
import time
import concurrent.futures
from base_chunker import BaseChunker
from llm_chunker import SAPLLMMultiDeploymentChunker

class DocumentationChunker:
    """Chunker for ABAP documentation and code files"""
    
    def __init__(self, max_chars=2000, overlap_chars=200, llm_chunkers=None, deployment_configs=None, use_llm=False):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.min_chunk_size = 100  # Minimum chunk size to avoid tiny chunks
        
        # Initialize with multiple LLM chunkers or create them if provided configurations
        if llm_chunkers:
            self.llm_chunkers = llm_chunkers
        elif deployment_configs:
            self.llm_chunkers = SAPLLMMultiDeploymentChunker(deployment_configs=deployment_configs)
        else:
            self.llm_chunkers = None
            
        self.use_llm = use_llm and self.llm_chunkers is not None
        self.file_specific_counter = {}  # Track counters per file
        self.base_chunker = BaseChunker(max_chars=max_chars, overlap_chars=overlap_chars)
        
        # Create thread pool for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        
    def process_file(self, doc_file, code_file, deployment_id=None, instance_id=0):
        """Process a single ABAP documentation and code file pair completely"""
        start_time = time.time()
        logging.info(f"Instance {instance_id}: Processing {os.path.basename(doc_file)}")

        # Read input files
        try:
            with open(doc_file, 'r', encoding='utf-8') as f:
                doc_content = f.read()
            with open(code_file, 'r', encoding='utf-8') as f:
                code_content = f.read()

            doc_filename = os.path.basename(doc_file)
            code_filename = os.path.basename(code_file)
            
            # Initialize file-specific counter if not exists
            if doc_filename not in self.file_specific_counter:
                self.file_specific_counter[doc_filename] = 0
                
        except Exception as e:
            logging.error(f"Error reading files: {e}")
            return pd.DataFrame()

        # Extract all methods from the ABAP code
        methods = self.base_chunker._extract_all_abap_methods(code_content)

        # Create chunks from documentation with associated code - pass the deployment_id
        chunks = self._create_chunks(doc_content, methods, code_filename, doc_filename, instance_id, deployment_id)
        
        # Merge small chunks
        chunks = self._merge_small_chunks(chunks)
        
        result_df = pd.DataFrame(chunks)

        # Add overlap between consecutive chunks from same section
        result_df = self._ensure_chunk_overlap(result_df)

        elapsed_time = time.time() - start_time
        logging.info(f"Instance {instance_id}: Processed {doc_filename} in {elapsed_time:.2f}s, generated {len(result_df)} chunks")

        return result_df
        
    def process_file_parallel_sections(self, doc_file, code_file, deployment_id=None, instance_id=0):
        """Process file with parallel section processing for better performance"""
        start_time = time.time()
        logging.info(f"Instance {instance_id}: Processing {os.path.basename(doc_file)} with parallel sections")

        # Read input files
        try:
            with open(doc_file, 'r', encoding='utf-8') as f:
                doc_content = f.read()
            with open(code_file, 'r', encoding='utf-8') as f:
                code_content = f.read()

            doc_filename = os.path.basename(doc_file)
            code_filename = os.path.basename(code_file)
            
            # Initialize file-specific counter if not exists
            if doc_filename not in self.file_specific_counter:
                self.file_specific_counter[doc_filename] = 0
                
        except Exception as e:
            logging.error(f"Error reading files: {e}")
            return pd.DataFrame()

        # Extract all methods from the ABAP code
        methods = self.base_chunker._extract_all_abap_methods(code_content)
        
        # Split documentation by headers to process sections in parallel
        header_sections = self.base_chunker._split_by_headers(doc_content)
        
        # Handle special sections (Overview, Class Definition)
        overview_section = None
        class_def_section = None
        for i, (header, content) in enumerate(header_sections):
            if "## Overview" in header:
                overview_section = (header, content)
            elif "## Class Definition" in header:
                class_def_section = (header, content)
        
        # Remove special sections from the main processing list
        processed_sections = []
        if overview_section:
            try:
                header_sections.remove(overview_section)
                processed_sections.append(overview_section)
            except:
                pass
        if class_def_section:
            try:
                header_sections.remove(class_def_section)
                processed_sections.append(class_def_section)
            except:
                pass
                
        # Create special chunks for overview and class definition together
        combined_chunks = []
        if overview_section and class_def_section:
            combined_content = f"{overview_section[0]}\n{overview_section[1]}\n{class_def_section[0]}\n{class_def_section[1]}"
            file_prefix = ""
            if doc_filename:
                # Extract just the filename without extension and clean it
                base_filename = os.path.splitext(doc_filename)[0]
                clean_filename = re.sub(r'[^a-zA-Z0-9]', '', base_filename)
                file_prefix = f"{clean_filename}_"
                
            # Include both file prefix and code filename in the chunk_id for absolute uniqueness
            code_file_id = re.sub(r'[^a-zA-Z0-9]', '', os.path.splitext(code_filename)[0])
            chunk_counter = self.file_specific_counter.get(doc_filename, 0)
            chunk_id = f"{file_prefix}{code_file_id}_instance_{instance_id}_chunk_{chunk_counter}"
            self.file_specific_counter[doc_filename] = chunk_counter + 1
            
            metadata = {
                "previous_chunk_id": None,
                "next_chunk_id": None,
                "code_snippet": "",
                "length": len(combined_content),
                "filename": code_filename
            }
            combined_chunks.append({
                "chunk_id": chunk_id,
                "doc": combined_content,
                "title": "Overview",
                "metadata": json.dumps(metadata)
            })
            
        # Process non-empty sections in parallel 
        valid_sections = [(h, c) for h, c in header_sections if c.strip()]
        
        # Only use parallel processing if we have enough sections
        if len(valid_sections) > 3 and len(valid_sections) < 100:
            # Process sections in parallel using thread pool
            future_to_section = {}
            futures = []
            
            # Submit tasks to process each section
            for header, content in valid_sections:
                section_text = f"{header}\n{content}" if header else content
                future = self.executor.submit(
                    self._process_section,
                    section_text=section_text,
                    code_methods=methods,
                    code_filename=code_filename,
                    doc_filename=doc_filename,
                    instance_id=instance_id,
                    deployment_id=deployment_id
                )
                futures.append(future)
                future_to_section[future] = (header, content)
            
            # Collect results as they complete
            section_chunks = []
            for future in concurrent.futures.as_completed(futures):
                section = future_to_section[future]
                try:
                    # Get chunks for this section
                    result = future.result()
                    if result:
                        section_chunks.extend(result)
                except Exception as e:
                    logging.error(f"Error processing section: {e}")
                    # Try to process this section sequentially if parallel processing failed
                    header, content = section
                    section_text = f"{header}\n{content}" if header else content
                    try:
                        result = self._process_section(
                            section_text=section_text,
                            code_methods=methods,
                            code_filename=code_filename,
                            doc_filename=doc_filename,
                            instance_id=instance_id,
                            deployment_id=deployment_id
                        )
                        if result:
                            section_chunks.extend(result)
                    except Exception as e2:
                        logging.error(f"Error in fallback processing: {e2}")
            
            # Combine all chunks
            all_chunks = combined_chunks + section_chunks
        else:
            # For a small number of sections, process sequentially
            all_chunks = combined_chunks
            for header, content in valid_sections:
                section_text = f"{header}\n{content}" if header else content
                result = self._process_section(
                    section_text=section_text,
                    code_methods=methods,
                    code_filename=code_filename,
                    doc_filename=doc_filename,
                    instance_id=instance_id,
                    deployment_id=deployment_id
                )
                if result:
                    all_chunks.extend(result)
                    
        # Merge small chunks
        all_chunks = self._merge_small_chunks(all_chunks)
        
        # Create dataframe
        result_df = pd.DataFrame(all_chunks)
        
        # Fix chunk links by sorting and updating next/previous IDs
        if not result_df.empty:
            result_df = self._update_chunk_links(result_df)
        
        # Add overlap between consecutive chunks from same section
        result_df = self._ensure_chunk_overlap(result_df)

        elapsed_time = time.time() - start_time
        logging.info(f"Instance {instance_id}: Processed {doc_filename} in {elapsed_time:.2f}s, generated {len(result_df)} chunks")

        return result_df
    
    def _update_chunk_links(self, df):
        """Update the next/previous chunk IDs to maintain proper sequence"""
        if len(df) <= 1:
            return df
            
        # Extract chunking IDs which should have sequence numbers at the end
        chunk_ids = df['chunk_id'].tolist()
        
        # Try to sort based on chunk ID sequence numbers
        try:
            # Sort the dataframe by chunk_id if possible
            df = df.sort_values(by='chunk_id', key=lambda x: [
                # Extract the prefix and number
                (re.sub(r'c\d+$', '', id), 
                 int(re.search(r'c(\d+)$', id).group(1)) if re.search(r'c(\d+)$', id) else 0) 
                for id in x
            ])
            
            # Update next/previous links
            for i in range(len(df)):
                metadata = json.loads(df.iloc[i]['metadata'])
                
                # Update previous link (except for first chunk)
                if i > 0:
                    metadata['previous_chunk_id'] = df.iloc[i-1]['chunk_id']
                else:
                    metadata['previous_chunk_id'] = None
                    
                # Update next link (except for last chunk)
                if i < len(df) - 1:
                    metadata['next_chunk_id'] = df.iloc[i+1]['chunk_id']
                else:
                    metadata['next_chunk_id'] = None
                    
                df.iloc[i, df.columns.get_loc('metadata')] = json.dumps(metadata)
        except Exception as e:
            logging.error(f"Error updating chunk links: {e}")
            
        return df
        
    def _process_section(self, section_text, code_methods, code_filename, doc_filename, instance_id, deployment_id=None):
        """Process a single section of documentation"""
        if not section_text.strip():
            return []
            
        # Extract method name and code snippet
        method_name = None
        code_snippet = ""
        parent_title = None
        
        # Check for header to extract title information
        header_match = re.search(r'(#{2,4})\s+([^\n]+)', section_text)
        if header_match:
            header_level = len(header_match.group(1))
            header_text = header_match.group(0)
            
            # For level 2 headers (##), extract section title
            if header_level == 2:
                section_title = header_match.group(2).strip()
                # Clean up section title if it contains table formatting
                if "|" in section_title:
                    # Extract just the part before the first pipe
                    section_title = section_title.split("|", 1)[0].strip()
                
                # Check if this is a meaningful section name (not a table header)
                if section_title and not section_title.strip("-|").strip() == "":
                    parent_title = section_title
            
            # For level 4 headers (####), extract method name
            elif header_level == 4:
                full_method = header_match.group(2).strip()
                # If method has tilde, extract the part after tilde
                if '~' in full_method:
                    method_name = full_method.split('~')[-1].strip()
                else:
                    method_name = full_method
                # Set as parent title for multi-part chunks
                parent_title = method_name
                
            # Find matching code for this header
            code_snippet = self.base_chunker._match_method_to_header(header_text, code_methods) or ""
        
        # Extract title based on content
        if method_name:
            title = method_name  # Use the clean method name
        elif parent_title:
            title = parent_title  # Use the section title
        else:
            # Extract from content
            title = self._extract_title_from_content(section_text)
            
        # File prefix for chunk IDs
        file_prefix = ""
        if doc_filename:
            base_filename = os.path.splitext(doc_filename)[0]
            file_prefix = re.sub(r'[^a-zA-Z0-9]', '', base_filename) + "_"
        
        # Code file hash for chunk ID
        code_file_hash = str(abs(hash(code_filename)) % 10000)
        
        # Get chunk counter for this file
        with self._get_counter_lock():
            chunk_counter = self.file_specific_counter.get(doc_filename, 0)
            self.file_specific_counter[doc_filename] = chunk_counter + 1
        
        # If section is small enough, keep as single chunk
        if len(section_text) <= self.max_chars:
            chunk_id = f"{file_prefix}i{instance_id}_h{code_file_hash}_c{chunk_counter}"
            metadata = {
                "previous_chunk_id": None,  # Will be updated later
                "next_chunk_id": None,
                "code_snippet": code_snippet,
                "length": len(section_text),
                "filename": code_filename
            }
            
            # Clean up title by removing table formatting
            if "|" in title:
                parts = title.split("|", 1)
                title = parts[0].strip()
            
            return [{
                "chunk_id": chunk_id,
                "doc": section_text,
                "title": title,
                "metadata": json.dumps(metadata)
            }]
            
        # For longer sections, split into chunks
        try:
            # Choose chunking method based on settings
            if self.use_llm and self.llm_chunkers:
                # Determine which deployment to use
                deployment_id_to_use = deployment_id
                if not deployment_id_to_use and hasattr(self.llm_chunkers, 'deployment_configs') and self.llm_chunkers.deployment_configs:
                    deployment_id_to_use = self.llm_chunkers.deployment_configs[0]["deployment_id"]
                    
                # Use LLM chunking
                content_chunks = self.llm_chunkers.split_text(section_text, deployment_id=deployment_id_to_use)
            else:
                # Use rule-based chunking
                content_chunks = self.base_chunker._split_content(section_text, self.max_chars, self.overlap_chars)
                
            # Process the chunks
            result_chunks = []
            for j, chunk_content in enumerate(content_chunks):
                # Skip empty chunks
                if not chunk_content.strip():
                    continue
                    
                # Get a new chunk ID
                with self._get_counter_lock():
                    chunk_counter = self.file_specific_counter.get(doc_filename, 0)
                    self.file_specific_counter[doc_filename] = chunk_counter + 1
                    
                chunk_id = f"{file_prefix}i{instance_id}_h{code_file_hash}_c{chunk_counter}"
                
                # Determine chunk title
                if method_name or parent_title:
                    # We're in a method or section with known title
                    base_title = method_name or parent_title
                    if len(content_chunks) > 1:
                        # Multiple chunks for same method/section - use part numbers
                        chunk_title = f"{base_title} (part {j+1})"
                    else:
                        chunk_title = base_title
                else:
                    # Try to extract title from the content
                    content_title = self._extract_title_from_content(chunk_content)
                    if content_title != "Unknown":
                        chunk_title = content_title
                    else:
                        # Fallback to original title or part number
                        if j == 0:
                            chunk_title = title
                        else:
                            chunk_title = f"{title} (part {j+1})"
                
                # Clean up title by removing table formatting
                if "|" in chunk_title:
                    parts = chunk_title.split("|", 1)
                    chunk_title = parts[0].strip()
                
                # Add the chunk
                metadata = {
                    "previous_chunk_id": None,  # Will be updated later
                    "next_chunk_id": None,
                    "code_snippet": code_snippet if j == 0 else "",  # Only include code in first chunk
                    "length": len(chunk_content),
                    "filename": code_filename
                }
                
                result_chunks.append({
                    "chunk_id": chunk_id,
                    "doc": chunk_content,
                    "title": chunk_title,
                    "metadata": json.dumps(metadata)
                })
                
            return result_chunks
            
        except Exception as e:
            logging.error(f"Error splitting content: {e}")
            # Fallback to simpler splitting
            try:
                content_chunks = self.base_chunker._split_content(section_text, self.max_chars, self.overlap_chars)
                
                result_chunks = []
                for j, chunk_content in enumerate(content_chunks):
                    # Skip empty chunks
                    if not chunk_content.strip():
                        continue
                        
                    # Get a new chunk ID
                    with self._get_counter_lock():
                        chunk_counter = self.file_specific_counter.get(doc_filename, 0)
                        self.file_specific_counter[doc_filename] = chunk_counter + 1
                        
                    chunk_id = f"{file_prefix}i{instance_id}_h{code_file_hash}_c{chunk_counter}"
                    
                    # Simple title with part numbers for multiple chunks
                    if len(content_chunks) > 1:
                        chunk_title = f"{title} (part {j+1})"
                    else:
                        chunk_title = title
                        
                    # Clean up title
                    if "|" in chunk_title:
                        parts = chunk_title.split("|", 1)
                        chunk_title = parts[0].strip()
                        
                    # Add the chunk
                    metadata = {
                        "previous_chunk_id": None,  # Will be updated later
                        "next_chunk_id": None,
                        "code_snippet": code_snippet if j == 0 else "",  # Only include code in first chunk
                        "length": len(chunk_content),
                        "filename": code_filename
                    }
                    
                    result_chunks.append({
                        "chunk_id": chunk_id,
                        "doc": chunk_content,
                        "title": chunk_title,
                        "metadata": json.dumps(metadata)
                    })
                    
                return result_chunks
                
            except Exception as e2:
                logging.error(f"Fallback chunking failed: {e2}")
                # Last resort - create a single chunk with the entire section
                with self._get_counter_lock():
                    chunk_counter = self.file_specific_counter.get(doc_filename, 0)
                    self.file_specific_counter[doc_filename] = chunk_counter + 1
                    
                chunk_id = f"{file_prefix}i{instance_id}_h{code_file_hash}_c{chunk_counter}"
                
                metadata = {
                    "previous_chunk_id": None,  # Will be updated later
                    "next_chunk_id": None,
                    "code_snippet": code_snippet,
                    "length": len(section_text),
                    "filename": code_filename
                }
                
                # Clean up title
                if "|" in title:
                    parts = title.split("|", 1)
                    title = parts[0].strip()
                
                return [{
                    "chunk_id": chunk_id,
                    "doc": section_text,
                    "title": title,
                    "metadata": json.dumps(metadata)
                }]
    
    def _get_counter_lock(self):
        """Get a lock for thread-safe counter access"""
        if not hasattr(self, '_counter_lock'):
            import threading
            self._counter_lock = threading.Lock()
        return self._counter_lock
    
    def _extract_title_from_content(self, content):
        """Extract the most appropriate title based on content"""
        # First check for level 2 headers (major sections)
        l2_matches = list(re.finditer(r'##\s+([^\n#]+)', content))
        
        if l2_matches:
            # Get the first level 2 header in the content
            primary_section = l2_matches[0].group(1).strip()
            
            # Check if this is a meaningful section name
            major_sections = ["Overview", "Class Definition", "Public Methods", 
                             "Protected Methods", "Private Methods", "Implementation",
                             "Examples", "Constants", "Types", "Attributes", 
                             "Implementation Overview", "Method Dependencies", 
                             "Redefined Methods", "Database Tables Used", 
                             "Critical Sections", "Method Implementation Details"]
            
            # Ensure section name doesn't contain table formatting
            if "|" in primary_section:
                # This looks like a table header, extract just the first part if possible
                parts = primary_section.split("|", 1)
                clean_title = parts[0].strip()
                if clean_title in major_sections or len(clean_title) > 3:
                    return clean_title
                
                # Try to extract just the section name by checking before the first |
                for section in major_sections:
                    if section in primary_section:
                        return section
                        
                # If we can't extract a clean section name, skip and try next header types
            elif primary_section in major_sections:
                return primary_section
            elif "|" not in primary_section and len(primary_section) <= 30:
                # Not a table header and reasonable length for a title
                return primary_section
        
        # Next, check for level 4 headers (methods) - prioritizing them over level 3 headers
        l4_matches = list(re.finditer(r'####\s+([^\n#]+)', content))
        if l4_matches:
            # Get the first method name
            method_full = l4_matches[0].group(1).strip()
            
            # Clean up the method name
            if ' - ' in method_full:
                method_name = method_full.split(' - ')[0].strip()
            elif '~' in method_full:
                method_name = method_full.split('~')[-1].strip()
            else:
                method_name = method_full
                
            return method_name
        
        # Next, check for level 3 headers (subsections)
        l3_matches = list(re.finditer(r'###\s+([^\n#]+)', content))
        if l3_matches:
            # Get the first level 3 header
            header = l3_matches[0].group(1).strip()
            # Make sure it's not a table header
            if not ("|" in header):
                return header
        
        # If no headers found, search for key phrases that might indicate content type
        if "Logic Flow:" in content:
            return "Implementation Details"
            
        # Look for a meaningful first line, avoiding table syntax
        first_lines = content.split('\n')[:5]  # Check more lines
        for line in first_lines:
            clean_line = line.strip()
            # Skip empty lines, headers, and table formatting
            if (clean_line and 
                not clean_line.startswith('#') and 
                not clean_line.startswith('|') and
                not clean_line.strip('-|').strip() == "" and
                not clean_line.startswith('- ') and  # Skip bullet points
                len(clean_line) > 2):  # Ensure it's substantive
                # Use the first meaningful line (truncated)
                if len(clean_line) > 30:
                    return clean_line[:30]
                else:
                    return clean_line
            
        return "Unknown"

    def _merge_small_chunks(self, chunks):
        """Merge small chunks to avoid tiny fragments"""
        if len(chunks) <= 1:
            return chunks
            
        merged_chunks = []
        current_chunk = None
        header_pattern = re.compile(r'^#{2,4}\s+')
        
        for i, chunk in enumerate(chunks):
            doc_content = chunk["doc"]
            chunk_length = len(doc_content)
            
            # Always keep the first chunk as is initially
            if current_chunk is None:
                current_chunk = chunk
                continue
                
            # Check if current chunk is small and can be merged
            current_length = len(current_chunk["doc"])
            is_small_chunk = current_length < self.min_chunk_size
            
            # Don't merge if current chunk starts with header and isn't tiny
            starts_with_header = bool(header_pattern.match(doc_content.lstrip()))
            has_method_name_title = any(f"#### {title}" in doc_content for title in ["constructor", "is_whitelisted", "get_submit"])
            
            # If this is a standalone header section that's tiny, it should be merged
            current_starts_with_header = bool(header_pattern.match(current_chunk["doc"].lstrip()))
            current_is_header_only = current_starts_with_header and current_chunk["doc"].strip().count('\n') <= 1
            
            # Conditions for merging:
            # 1. Current chunk is very small OR it's just a header
            # 2. Combined size is less than max_chars
            # 3. Not merging across different method sections unless one is just a header
            should_merge = (is_small_chunk or current_is_header_only) and \
                         (current_length + chunk_length <= self.max_chars)

            # If we should merge, combine the chunks
            if should_merge:
                # Combine the content
                combined_doc = current_chunk["doc"] + "\n\n" + doc_content
                current_chunk["doc"] = combined_doc
                
                # Update metadata
                metadata = json.loads(current_chunk["metadata"])
                next_metadata = json.loads(chunk["metadata"])
                metadata["next_chunk_id"] = next_metadata["next_chunk_id"]
                metadata["length"] = len(combined_doc)
                
                # Keep code snippet from both if possible
                if metadata["code_snippet"] == "" and next_metadata["code_snippet"] != "":
                    metadata["code_snippet"] = next_metadata["code_snippet"]
                    
                current_chunk["metadata"] = json.dumps(metadata)
                
                # Use the most meaningful title - prefer method names and section headers
                # over content snippets
                current_title = current_chunk["title"]
                next_title = chunk["title"]
                
                if "..." in current_title and "..." not in next_title:
                    current_chunk["title"] = next_title
                elif "part" in next_title and "part" not in current_title:
                    current_chunk["title"] = current_title  # Keep the base title without part number
            else:
                # Can't merge, save current chunk and start a new one
                merged_chunks.append(current_chunk)
                current_chunk = chunk
        
        # Don't forget the last chunk
        if current_chunk is not None:
            merged_chunks.append(current_chunk)
            
        # Update chunk IDs to be sequential
        if len(merged_chunks) < len(chunks):
            # Extract prefix from first chunk ID
            first_id = merged_chunks[0]["chunk_id"]
            prefix_match = re.match(r"(.+?)c\d+$", first_id)
            if prefix_match:
                prefix = prefix_match.group(1)
                # Reassign sequential chunk IDs
                for i, chunk in enumerate(merged_chunks):
                    old_id = chunk["chunk_id"]
                    new_id = f"{prefix}c{i}"
                    chunk["chunk_id"] = new_id
                    
                    # Update previous_chunk_id in the next chunk
                    if i < len(merged_chunks) - 1:
                        next_metadata = json.loads(merged_chunks[i+1]["metadata"])
                        next_metadata["previous_chunk_id"] = new_id
                        merged_chunks[i+1]["metadata"] = json.dumps(next_metadata)
                    
                    # Update next_chunk_id in the current chunk
                    metadata = json.loads(chunk["metadata"])
                    if i < len(merged_chunks) - 1:
                        metadata["next_chunk_id"] = f"{prefix}c{i+1}"
                    else:
                        metadata["next_chunk_id"] = None
                    chunk["metadata"] = json.dumps(metadata)
        
        return merged_chunks

    def _ensure_chunk_overlap(self, df):
        """Ensure that chunks have some overlap when they belong to the same section"""
        if len(df) < 2:
            return df

        for i in range(len(df) - 1):
            current_chunk = df.iloc[i]
            next_chunk = df.iloc[i+1]

            # Only process chunks from the same section
            current_title = current_chunk['title'].split('(')[0].strip()
            next_title = next_chunk['title'].split('(')[0].strip()

            if current_title != next_title:
                continue

            current_text = current_chunk['doc']
            next_text = next_chunk['doc']

            # If next chunk doesn't start with a header, add overlap
            if not re.match(r'^(#{2,4}|[A-Za-z])', next_text.lstrip()):
                # Find a good paragraph to use as overlap
                paragraphs = current_text.split('\n\n')
                if paragraphs:
                    last_para = paragraphs[-1]
                    if len(last_para) < 200:
                        new_next_text = last_para + "\n\n" + next_text
                        df.at[i+1, 'doc'] = new_next_text

                        # Update metadata
                        metadata = json.loads(next_chunk['metadata'])
                        metadata['length'] = len(new_next_text)
                        df.at[i+1, 'metadata'] = json.dumps(metadata)

        return df