import requests
import re
import lxml.html
import html
from tqdm import tqdm

def build_clean_corpus_robust(url, output_txt_path, sample_rate=100, target_size_mb=60, max_lines=2000000):
    target_size_bytes = target_size_mb * 1024 * 1024
    print(f"Streaming (robust mode) from {url}...")
    
    current_size = 0
    current_lines = 0
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return

    pbar = tqdm(unit=' rows', desc="Processing rows")
    
    # Regex to find complete <row> tags
    row_re = re.compile(r'<row\s+([^>]*?)\s*/?>', re.IGNORECASE)
    id_re = re.compile(r'Id="(\d+)"', re.IGNORECASE)
    text_re = re.compile(r'Text="([^"]*)"', re.IGNORECASE)
    
    # We'll read in chunks to keep memory low but allow regex to span lines
    chunk_size = 1024 * 1024 # 1MB chunks
    buffer = ""
    
    try:
        with open(output_txt_path, 'w', encoding='utf-8') as out_file:
            for chunk in response.iter_content(chunk_size=chunk_size, decode_unicode=True):
                if not chunk:
                    break
                    
                buffer += chunk
                
                # Find all <row ...> tokens
                matches = list(row_re.finditer(buffer))
                if matches:
                    for match in matches:
                        row_content = match.group(1)
                        
                        id_match = id_re.search(row_content)
                        text_match = text_re.search(row_content)
                        
                        if id_match and text_match:
                            comment_id = int(id_match.group(1))
                            if comment_id % sample_rate == 0:
                                raw_text = text_match.group(1)
                                
                                # Unescape XML entities
                                raw_text = html.unescape(raw_text)
                                
                                try:
                                    html_tree = lxml.html.fromstring(f"<div>{raw_text}</div>")
                                    for tag in html_tree.xpath('.//pre | .//blockquote | .//code'):
                                        tag.drop_tree()
                                    clean_text = html_tree.text_content().strip()
                                    
                                    if clean_text:
                                        # Handle Markdown
                                        clean_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean_text)
                                        clean_text = re.sub(r'(\*\*|__|[\*_])', '', clean_text)
                                        
                                        # Remove URLs
                                        clean_text = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+', '', clean_text)
                                        # Remove user handles
                                        clean_text = re.sub(r'@[\w-]+', '', clean_text)
                                        
                                        clean_text = " ".join(clean_text.split())
                                        line = clean_text + '\n'
                                        out_file.write(line)
                                        current_size += len(line.encode('utf-8'))
                                        current_lines += 1
                                        
                                        if current_size >= target_size_bytes or current_lines >= max_lines:
                                            reason = "target size" if current_size >= target_size_bytes else "max lines"
                                            print(f"\nReached {reason} ({current_size / (1024*1024):.2f} MB, {current_lines} lines).")
                                            return
                                except:
                                    pass
                                
                        pbar.update(1)
                    
                    # Keep everything after the last match in the buffer
                    buffer = buffer[matches[-1].end():]
                
                # Prevent buffer from growing infinitely if no matches found
                if len(buffer) > chunk_size * 2:
                    buffer = buffer[-chunk_size:]
                    
    finally:
        response.close()
        pbar.close()

if __name__ == "__main__":
    # URL discovered from Archive.org virtual directory for stackoverflow.com.7z
    SO_COMMENTS_URL = "https://archive.org/download/stackexchange_20251231/stackexchange_20251231/stackoverflow.com.7z/Comments.xml"
    
    # Path relative to scripts folder
    OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "stack_overflow_comments.txt")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Using sample_rate=10 to reach 60MB before the Archive.org stream error at row ~11M
    build_clean_corpus_robust(SO_COMMENTS_URL, OUTPUT_FILE, sample_rate=10, target_size_mb=60, max_lines=2000000)
