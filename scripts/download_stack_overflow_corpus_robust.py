import codecs
import html
import os
import re
import urllib.request

def build_clean_corpus_robust(url, output_txt_path, sample_rate=100, target_size_mb=60, max_lines=2000000):
    target_size_bytes = target_size_mb * 1024 * 1024
    print(f"Streaming (robust mode) from {url}...")
    
    current_size = 0
    current_lines = 0
    rows_seen = 0
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req)

    # Decode incrementally so multi-byte UTF-8 chars split across chunk boundaries aren't corrupted.
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')

    # Regex to find complete <row> tags
    row_re = re.compile(r'<row\s+([^>]*?)\s*/?>', re.IGNORECASE)
    id_re = re.compile(r'Id="(\d+)"', re.IGNORECASE)
    text_re = re.compile(r'Text="([^"]*)"', re.IGNORECASE)
    code_block_re = re.compile(r'<(?:pre|blockquote|code)\b[^>]*>.*?</(?:pre|blockquote|code)>', re.DOTALL | re.IGNORECASE)
    # Whitelist of standard HTML tags to avoid eating prose with angle brackets like List<String> or a < b
    html_tags = (
        r'a|abbr|acronym|address|applet|area|article|aside|audio|b|base|basefont|bdi|bdo|big|'
        r'blockquote|body|br|button|canvas|caption|center|cite|code|col|colgroup|data|datalist|dd|'
        r'del|details|dfn|dialog|dir|div|dl|dt|em|embed|fieldset|figcaption|figure|font|footer|form|'
        r'frame|frameset|h[1-6]|head|header|hgroup|hr|html|i|iframe|img|input|ins|kbd|label|legend|'
        r'li|link|main|map|mark|meta|meter|nav|noframes|noscript|object|ol|optgroup|option|output|'
        r'p|param|picture|pre|progress|q|rp|rt|ruby|s|samp|script|section|select|small|source|span|'
        r'strike|strong|style|sub|summary|sup|svg|table|tbody|td|template|textarea|tfoot|th|thead|'
        r'time|title|tr|track|tt|u|ul|var|video|wbr'
    )
    tag_re = re.compile(rf'</?(?:{html_tags})\b[^>]*>', re.IGNORECASE)
    link_re = re.compile(r'\[([^\]]+)\]\([^\)]+\)')
    md_style_re = re.compile(r'(\*\*|__|[\*_])')
    url_re = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
    user_re = re.compile(r'@[\w-]+')

    # We'll read in chunks to keep memory low but allow regex to span lines
    chunk_size = 1024 * 1024 # 1MB chunks
    buffer = ""
    
    try:
        with open(output_txt_path, 'w', encoding='utf-8') as out_file:
            while True:
                chunk_bytes = response.read(chunk_size)
                if not chunk_bytes:
                    break
                chunk = decoder.decode(chunk_bytes)
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
                                    # Strip code/pre blocks and any remaining tags
                                    stripped = code_block_re.sub('', raw_text)
                                    clean_text = tag_re.sub('', stripped)
                                    clean_text = html.unescape(clean_text).strip()
                                    
                                    if clean_text:
                                        # Handle Markdown, URLs, and handles
                                        clean_text = link_re.sub(r'\1', clean_text)
                                        clean_text = md_style_re.sub('', clean_text)
                                        clean_text = url_re.sub('', clean_text)
                                        clean_text = user_re.sub('', clean_text)
                                        
                                        clean_text = " ".join(clean_text.split())
                                        line = clean_text + '\n'
                                        out_file.write(line)
                                        current_size += len(line.encode('utf-8'))
                                        current_lines += 1
                                        
                                        if current_size >= target_size_bytes or current_lines >= max_lines:
                                            reason = "target size" if current_size >= target_size_bytes else "max lines"
                                            print(f"\nReached {reason} ({current_size / (1024*1024):.2f} MB, {current_lines} lines).")
                                            return
                                except Exception:
                                    pass
                                
                        rows_seen += 1
                        if rows_seen % 50_000 == 0:
                            print(f"\rRows scanned: {rows_seen:,} | Lines written: {current_lines:,} ({current_size / (1024*1024):.1f} MB)...", end="", flush=True)
                    
                    # Keep everything after the last match in the buffer
                    buffer = buffer[matches[-1].end():]
                
                # Prevent buffer from growing infinitely if no matches found
                if len(buffer) > chunk_size * 2:
                    buffer = buffer[-chunk_size:]
                    
    finally:
        response.close()
        print()

if __name__ == "__main__":
    # URL discovered from Archive.org virtual directory for stackoverflow.com.7z
    SO_COMMENTS_URL = "https://archive.org/download/stackexchange_20251231/stackexchange_20251231/stackoverflow.com.7z/Comments.xml"
    
    # Path relative to scripts folder
    OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "stack_overflow_comments.txt")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Using sample_rate=10 to reach 60MB before the Archive.org stream error at row ~11M
    build_clean_corpus_robust(SO_COMMENTS_URL, OUTPUT_FILE, sample_rate=10, target_size_mb=60, max_lines=2000000)
