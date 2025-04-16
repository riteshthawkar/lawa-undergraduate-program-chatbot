import re
import urllib.parse
from typing import List, Dict, Tuple

from modules.config import logger

def validate_citation_numbers(citation_numbers: List[int], max_docs: int) -> List[int]:
    """Validate that citation numbers are within the valid range"""
    return [num for num in citation_numbers if 1 <= num <= max_docs]

def process_citations(complete_answer: str, ranked_docs: List[dict]) -> Tuple[str, List[dict]]:
    """
    Extracts citation numbers from the answer, maps them to consecutive citation numbers,
    and returns the updated answer along with a list of citation sources.
    
    Handles special cases:
    - URLs with spaces are properly URL-encoded
    - Malformed citation formats are fixed
    - Missing or invalid citation numbers are preserved but not linked
    """
    citations = []
    seen_nums = set()
    citation_numbers = []
    
    # Extract all citation numbers from the answer
    for num_str in re.findall(r'\[(\d+)\]', complete_answer):
        try:
            num = int(num_str)
            if num not in seen_nums:
                seen_nums.add(num)
                citation_numbers.append(num)
        except ValueError:
            logger.warning(f"Invalid citation number format: {num_str}")
            continue
            
    valid_citations = validate_citation_numbers(citation_numbers, len(ranked_docs))
    
    seen_urls = {}
    citation_map = {}
    current_num = 1
    
    # Process valid citations and create mapping
    for num in valid_citations:
        try:
            url = ranked_docs[num - 1]["page_source"]
            
            # Fix: URL-encode spaces and special characters in the URL
            # But preserve the structure of the URL itself
            parsed_url = urllib.parse.urlparse(url)
            
            # URL encode each component separately
            encoded_path = urllib.parse.quote(parsed_url.path, safe='/,')
            encoded_query = urllib.parse.quote_plus(parsed_url.query, safe='&=')
            # Fix: Only encode spaces and unsafe characters in fragment, but do NOT encode '='
            # This preserves fragments like 'page=10' as-is
            encoded_fragment = urllib.parse.quote(parsed_url.fragment, safe='=')

            # Reconstruct the URL with encoded components
            encoded_url = urllib.parse.urlunparse((
                parsed_url.scheme,
                parsed_url.netloc,
                encoded_path,
                parsed_url.params,
                encoded_query,
                encoded_fragment
            ))
            
            # Check if this URL has been seen before (using the encoded version)
            if encoded_url not in seen_urls:
                citation_map[num] = current_num
                seen_urls[encoded_url] = current_num
                citations.append({"url": encoded_url, "cite_num": str(current_num)})
                current_num += 1
            else:
                citation_map[num] = seen_urls[encoded_url]
        except IndexError:
            logger.warning(f"Citation number {num} out of range for ranked_docs")
            continue
        except KeyError:
            logger.warning(f"Missing 'page_source' key in ranked_docs for citation {num}")
            continue
        except Exception as e:
            logger.exception(f"Error processing citation {num}: {str(e)}")
            continue
    
    logger.debug(f"Citation numbers extracted: {citation_numbers}")
    logger.debug(f"Seen URLs mapping: {seen_urls}")

    def replace_citation(match):
        try:
            original = int(match.group(1))
            new_num = citation_map.get(original, original)
            url = next((c["url"] for c in citations if c["cite_num"] == str(new_num)), "")
            
            # Return formatted citation with URL if available
            return f"[{new_num}]({url})" if url else f"[{new_num}]"
        except (ValueError, IndexError) as e:
            # Handle any errors in citation replacement
            logger.warning(f"Error in replace_citation: {str(e)}")
            return match.group(0)  # Return original citation unchanged

    # First replace all citations with their new numbers
    updated_answer = re.sub(r'\[(\d+)\]', replace_citation, complete_answer)
    
    # Remove duplicate adjacent citations in a loop until no more changes are made
    prev_answer = ""
    while prev_answer != updated_answer:
        prev_answer = updated_answer
        
        # Handle citations with URLs: [n](url)[n](url)
        # Fix: Use a more robust regex that can handle URLs with spaces and special characters
        updated_answer = re.sub(r'\[(\d+)\]\(([^)]+)\)\s*\[(\1)\]\([^)]+\)', r'[\1](\2)', updated_answer)
        
        # Handle citations without URLs: [n][n]
        updated_answer = re.sub(r'\[(\d+)\]\s*\[(\1)\]', r'[\1]', updated_answer)
        
        # Handle cases with whitespace or other characters between duplicate citations
        updated_answer = re.sub(r'\[(\d+)\](?:\s*[,.;:]?\s*)\[(\1)\]', r'[\1]', updated_answer)
        
        # Fix: Add specific handling for malformed markdown links
        updated_answer = re.sub(r'\[(\d+)\]\s*\(([^)]*)\s+([^)]*)\)', r'[\1](\2\3)', updated_answer)
        
        # Fix: Handle potential broken markdown links with spaces between [] and ()
        updated_answer = re.sub(r'\[(\d+)\]\s+\(([^)]+)\)', r'[\1](\2)', updated_answer)
    
    # Clean up any potential artifacts from the replacements
    updated_answer = re.sub(r'\s+([,.;:])', r'\1', updated_answer)
    
    # Fix: Ensure all linked citations have properly formed markdown links
    updated_answer = re.sub(r'\[(\d+)\]\(([^)]*)\)', lambda m: f"[{m.group(1)}]({m.group(2).strip()})", updated_answer)
    
    return updated_answer, sorted(citations, key=lambda x: int(x["cite_num"])) 