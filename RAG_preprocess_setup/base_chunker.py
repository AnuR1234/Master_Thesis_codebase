#base_chunker.py
import re
import os
import json
import logging

from chunking_evaluation.utils import openai_token_count
from chunking_evaluation.chunking import RecursiveTokenChunker

class BaseChunker:
    """Base class for document chunking with header-aware splitting"""
    
    def __init__(self, max_chars=2000, overlap_chars=200):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        
        # Initialize the recursive token chunker for initial text splitting
        self.splitter = RecursiveTokenChunker(
            chunk_size=50,
            chunk_overlap=0,
            length_function=openai_token_count
        )
        
    def _extract_titles_from_content(self, content):
        """Extract method name from content for title"""
        # First try to get level 4 headers (method names)
        h4_matches = re.finditer(r'(?:^|\n)####\s+([^#\n]+)', content)
        for match in h4_matches:
            method_name = match.group(1).strip()
            # Remove everything after dash if present
            if ' - ' in method_name:
                method_name = method_name.split(' - ')[0].strip()
            # Extract just the method name without class prefix
            if '~' in method_name:
                method_name = method_name.split('~')[-1].strip()
            return method_name
            
        # If no level 4 header, look for level 2 headers
        h2_matches = re.finditer(r'(?:^|\n)##\s+([^#\n]+)', content)
        for match in h2_matches:
            section_title = match.group(1).strip()
            # Check if this is a table header or separator, skip it
            if section_title.startswith("|") or "|" in section_title or section_title.strip("-|").strip() == "":
                # Extract the clean section name if possible
                if "|" in section_title:
                    clean_section = section_title.split("|", 1)[0].strip()
                    if clean_section:
                        return clean_section
                continue
            return section_title
        
        # Fallback to "Unknown"
        return "Unknown"
    
    def _split_by_headers(self, text):
        """Split documentation by level 4 headers (####)"""
        pattern = r'(^|\n)(#{4}\s+[^#\n]+)(\n|$)'
        header_matches = list(re.finditer(pattern, text, re.MULTILINE))
        sections = []

        if not header_matches:
            return [("", text)]
        
        # Handle content before first method header
        if header_matches[0].start() > 0:
            sections.append(("", text[:header_matches[0].start()]))
        
        # Process each header section
        for i, match in enumerate(header_matches):
            header = match.group(2)
            start_pos = match.start()
            
            # Determine end position (next header or end of text)
            if i < len(header_matches) - 1:
                end_pos = header_matches[i + 1].start()
            else:
                end_pos = len(text)
            
            # Extract section content
            section_content = text[start_pos:end_pos]
            sections.append((header, section_content))
        
        return sections
        
    def _split_content(self, content, max_length, overlap):
        """Split content into chunks respecting header boundaries when possible"""
        # First, find all level 2 (##) and level 4 (####) headers
        headers = []
        
        # Look for level 2 headers (major sections)
        for match in re.finditer(r'(^|\n)(##\s+[^#\n]+)(\n|$)', content, re.MULTILINE):
            headers.append((match.start(), 2, match.group(2)))
            
        # Look for level 4 headers (methods)
        for match in re.finditer(r'(^|\n)(####\s+[^#\n]+)(\n|$)', content, re.MULTILINE):
            headers.append((match.start(), 4, match.group(2)))
            
        # Sort headers by their position in the text
        headers.sort(key=lambda x: x[0])
        
        if not headers:
            # No headers found, use paragraph-based splitting
            return self._split_by_paragraphs(content, max_length, overlap)
            
        # Initialize for header-based chunking
        chunks = []
        current_chunk_start = 0
        current_section_level = 0
        current_section_title = ""
        
        # Process each header
        for i, (header_pos, header_level, header_text) in enumerate(headers):
            # Determine the end of this section (next header or end of content)
            next_pos = headers[i+1][0] if i < len(headers)-1 else len(content)
            section_text = content[header_pos:next_pos]
            
            # If this is a level 2 header, it starts a new major section
            if header_level == 2:
                # If we've accumulated content, save it as a chunk
                if current_chunk_start < header_pos:
                    chunk_text = content[current_chunk_start:header_pos].strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                
                # Start a new chunk with this header
                current_chunk_start = header_pos
                current_section_level = 2
                current_section_title = header_text
                
            # If this is a level 4 header (method), decide if it should be in a new chunk
            elif header_level == 4:
                # Calculate size of current chunk plus this new method section
                potential_chunk_size = next_pos - current_chunk_start
                
                # If adding this method would make the chunk too big, or if we're at top level,
                # finish current chunk and start a new one
                if potential_chunk_size > max_length or current_section_level == 0:
                    if current_chunk_start < header_pos:
                        chunk_text = content[current_chunk_start:header_pos].strip()
                        if chunk_text:
                            chunks.append(chunk_text)
                    
                    # Start a new chunk with this method
                    current_chunk_start = header_pos
                    
        # Don't forget the last chunk
        if current_chunk_start < len(content):
            final_chunk = content[current_chunk_start:].strip()
            if final_chunk:
                chunks.append(final_chunk)
                
        # If any of the chunks are still too big, split them further
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > max_length:
                # Split big chunks by paragraphs
                sub_chunks = self._split_by_paragraphs(chunk, max_length, overlap)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
                
        return final_chunks
    
    def _split_by_paragraphs(self, content, max_length, overlap):
        """Split content into chunks based on paragraph boundaries"""
        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph would exceed the limit, finish the current chunk
            if current_chunk and (len(current_chunk) + len(para) > max_length):
                chunks.append(current_chunk.strip())
                
                # Start a new chunk with overlap if possible
                if overlap > 0:
                    # Find the last paragraph break within the overlap window
                    overlap_text = current_chunk[-min(overlap, len(current_chunk)):]
                    last_break = overlap_text.rfind('\n\n')
                    
                    if last_break != -1:
                        # We found a paragraph break in the overlap window
                        current_chunk = current_chunk[-(min(overlap, len(current_chunk))-last_break):]
                    else:
                        # No paragraph break found, use the entire overlap
                        current_chunk = current_chunk[-min(overlap, len(current_chunk)):]
                else:
                    current_chunk = ""
            
            # Add the paragraph to the current chunk
            if current_chunk and not current_chunk.endswith('\n\n'):
                current_chunk += '\n\n'
            current_chunk += para
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
        
    def _extract_all_abap_methods(self, code_content):
        """Extract all methods from ABAP code, handling various syntax patterns"""
        methods = {}
        method_count = 0

        # First approach: Extract METHOD...ENDMETHOD blocks with improved pattern
        current_pos = 0
        remaining_code = code_content

        method_pattern = re.compile(
            r'(?i)(?:^|\s+)(METHOD)\s+([^\s\(\.\!]+)',
            re.MULTILINE
        )
        endmethod_pattern = re.compile(
            r'(?i)(?:^|\s+)(ENDMETHOD)\.?',
            re.MULTILINE
        )

        while True:
            # Find next METHOD declaration
            method_match = method_pattern.search(remaining_code)
            if not method_match:
                break

            method_name = method_match.group(2).lower().strip('.')
            method_start = current_pos + method_match.start()
            search_pos = method_match.end()

            # Find matching ENDMETHOD with relaxed matching
            nesting_level = 1
            method_end = None
            safety_counter = 0
            max_iterations = 10000

            while nesting_level > 0 and safety_counter < max_iterations:
                safety_counter += 1

                # Look for next METHOD or ENDMETHOD
                next_method = method_pattern.search(remaining_code, search_pos)
                next_endmethod = endmethod_pattern.search(remaining_code, search_pos)

                if next_endmethod:
                    # Handle ENDMETHOD with optional period
                    end_pos = next_endmethod.end()
                    nesting_level -= 1
                    if nesting_level == 0:
                        method_end = current_pos + end_pos
                    search_pos = end_pos
                elif next_method:
                    nesting_level += 1
                    search_pos = next_method.end()
                else:
                    # No more matches found, break
                    break

            if method_end and safety_counter < max_iterations:
                # Extract full method content
                method_content = code_content[method_start:method_end]
                methods[method_name] = method_content
                method_count += 1
                current_pos = method_end
                remaining_code = code_content[current_pos:]
            else:
                # Move past current METHOD if no end found
                current_pos += method_match.end()
                remaining_code = code_content[current_pos:]

        # Second approach: Extract method declarations (for interface methods)
        declaration_pattern = re.compile(
            r'(?i)(?:^|\s+)(?:METHODS|CLASS-METHODS)\s+([^\s\(\.\!]+)',
            re.MULTILINE
        )
        declarations = declaration_pattern.findall(code_content)
        for decl_name in declarations:
            decl_name = decl_name.lower()
            if decl_name not in methods:
                # Find the full declaration line
                try:
                    line_start = code_content.rfind('\n', 0, code_content.find(decl_name)) + 1
                    line_end = code_content.find('\n', line_start)
                    methods[decl_name] = code_content[line_start:line_end].strip()
                    method_count += 1
                except Exception as e:
                    pass

        # Handle special naming patterns
        for pattern, prefix in {'MIGRATIONOBJEC(\\d+)': 'migrationobjec'}.items():
            for method_name in list(methods.keys()):
                match = re.search(pattern, method_name, re.IGNORECASE)
                if match:
                    alt_key = f"{prefix}{match.group(1)}"
                    methods[alt_key] = methods[method_name]

        return methods
    
    def _match_method_to_header(self, header_text, code_methods):
        """Match a documentation header with code methods using multiple strategies"""
        if not header_text:
            return None
        # Extract method name from header
        header_match = re.search(r'#{4}\s+([/\w\d_~-]+)', header_text)
        if not header_match:
            return None
        method_name = header_match.group(1).lower().strip('.')

        # Strategy 1: Direct match
        if method_name in code_methods:
            return code_methods[method_name]

        # Strategy 2: Try removing underscores and dashes
        clean_method = re.sub(r'[_\-~]', '', method_name)
        for code_name, code_content in code_methods.items():
            clean_code = re.sub(r'[_\-~]', '', code_name)
            if clean_method == clean_code:
                return code_content

        # Strategy 3: Special case for migration object methods
        if 'migrationobjec' in method_name:
            base_match = re.search(r'(migrationobjec\d+)', method_name)
            if base_match:
                base_name = base_match.group(1)
                if base_name in code_methods:
                    return code_methods[base_name]
                # Try finding a method that starts with this base name
                for code_name, code_content in code_methods.items():
                    if code_name.startswith(base_name):
                        return code_content

        # Strategy 4: Suffix match
        for code_name, code_content in code_methods.items():
            if code_name.endswith(method_name):
                return code_content

        # Strategy 5: Substring matching (for truncated or abbreviated method names)
        for code_name, code_content in code_methods.items():
            # Check if the method name is contained within the code name or vice versa
            if method_name in code_name or code_name in method_name:
                if len(method_name) >= 5 and len(code_name) >= 5:  # Avoid very short matches
                    return code_content

        return None

    def extract_simple_title(self, content):
        """Extract a clean, simple method name or section title from content"""
        # First look for method headers (####)
        method_match = re.search(r'####\s+([^\n]+)', content)
        if method_match:
            method_name = method_match.group(1).strip()
            # Extract just the method name without class prefix and description
            if '~' in method_name:
                method_name = method_name.split('~')[-1]
            if ' - ' in method_name:
                method_name = method_name.split(' - ')[0]
            return method_name.strip()
        
        # Then look for section headers (##)
        section_match = re.search(r'##\s+([^\n]+)', content)
        if section_match:
            section_name = section_match.group(1).strip()
            # Skip table headers or separators and extract just the meaningful part
            if "|" in section_name:
                clean_section = section_name.split("|", 1)[0].strip()
                if clean_section:
                    return clean_section
            elif not section_name.startswith("|") and not section_name.strip("-|").strip() == "":
                return section_name
            
        return "Unknown"