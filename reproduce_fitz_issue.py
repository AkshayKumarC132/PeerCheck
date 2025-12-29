
import fitz

def test_search():
    # Create a dummy PDF with multiline text
    doc = fitz.open()
    page = doc.new_page()
    
    # Insert text with a newline visual look (but effectively separate lines)
    # in PDF, text is often just positioned strings.
    # insert_text with \n usually creates multiple lines.
    text_content = "Line 1 text.\nLine 2 text."
    page.insert_text((50, 50), text_content)
    
    print(f"Page text:\n{page.get_text()}")
    
    # Test 1: Search with direct newline string
    hits_raw = page.search_for(text_content)
    print(f"Search raw '{text_content.replace(chr(10), '<nl>')}' hits: {len(hits_raw)}")
    
    # Test 2: Search formatted with single space (common issue)
    text_normalized = text_content.replace('\n', ' ')
    hits_norm = page.search_for(text_normalized)
    print(f"Search normalized '{text_normalized}' hits: {len(hits_norm)}")
    
    # Test 3: Search line by line? No, we want the block.
    
    # Real world scenario from logs:
    # Quote: "AL HK-10A - LATCHED DETENT OPEN\no\nRECORD results..."
    # Using specific text from user logs to simulate
    complex_text = "AL HK-10A - LATCHED DETENT OPEN\no\nRECORD results"
    page.insert_text((50, 100), complex_text)
    
    hits_complex = page.search_for(complex_text)
    print(f"Search complex raw hits: {len(hits_complex)}")
    
    # Attempting fix: replace \n with space? 
    # Warning: PDF text extraction might not put a space if lines are just close. 
    # But usually extract_text() puts \n. search_for might NOT handle \n as a separator.
    
    # Re-reading docs: search_for default is dehyphen=True. 
    # It does NOT match across lines easily if \n is explicit in the search string unless text layer matches exactly.
    
    # Let's try searching for just a part
    part = "LATCHED DETENT OPEN"
    print(f"Search part '{part}' hits: {len(page.search_for(part))}")

if __name__ == "__main__":
    test_search()
