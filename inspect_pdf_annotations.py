
import fitz
import sys

def inspect_pdf(pdf_path):
    print(f"Inspecting: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return

    total_annots = 0
    color_counts = {}
    
    for i, page in enumerate(doc):
        annots = page.annots()
        if not annots:
            continue
            
        page_annots = 0
        for annot in annots:
            page_annots += 1
            total_annots += 1
            
            # Check color
            color = annot.colors.get('stroke')
            if color:
                # Round color for grouping
                c_key = tuple(round(c, 2) for c in color)
                color_counts[c_key] = color_counts.get(c_key, 0) + 1
        
        if page_annots > 0:
            print(f"Page {i+1}: {page_annots} annotations")

    print(f"\nTotal Annotations: {total_annots}")
    print("Color Distribution:")
    for c, count in color_counts.items():
        # Identify common colors
        label = str(c)
        if c == (0.6, 1.0, 0.6): label += " (Green)"
        elif c == (1.0, 0.6, 0.6): label += " (Red)"
        print(f"  {label}: {count}")

if __name__ == "__main__":
    target_file = r"d:\PeerCheck\22a05aad-4b98-4a3b-b0d1-7d5a2b2217d8_highlighted_report (2).pdf"
    inspect_pdf(target_file)
