import os
import PyPDF2
import pdfplumber
from typing import List, Optional

def extract_text_from_pdf_pypdf2(pdf_path: str) -> str:
    """
    Extract text from PDF using PyPDF2 (fallback method)
    """
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path} with PyPDF2: {e}")
        return ""

def extract_text_from_pdf_pdfplumber(pdf_path: str) -> str:
    """
    Extract text from PDF using pdfplumber (primary method)
    """
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path} with pdfplumber: {e}")
        return ""

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF using the best available method
    """
    print(f"Extracting text from PDF: {os.path.basename(pdf_path)}")
    
    # Try pdfplumber first (better for complex layouts)
    text = extract_text_from_pdf_pdfplumber(pdf_path)
    
    # If pdfplumber fails or returns empty, try PyPDF2
    if not text.strip():
        print(f"pdfplumber failed for {pdf_path}, trying PyPDF2...")
        text = extract_text_from_pdf_pypdf2(pdf_path)
    
    return text

def load_txt_file(filepath: str) -> str:
    """
    Load content from a TXT file
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
            return file.read().strip()
    except Exception as e:
        print(f"Error reading TXT file {filepath}: {e}")
        return ""

def load_documents(directory: str, supported_formats: Optional[List[str]] = None) -> List[str]:
    """
    Load all supported document files from a directory
    
    Args:
        directory: Path to the directory containing documents
        supported_formats: List of supported file extensions (default: ['txt', 'pdf'])
    
    Returns:
        List of document texts
    """
    if supported_formats is None:
        supported_formats = ['txt', 'pdf']
    
    texts = []
    
    if not os.path.exists(directory):
        print(f"Warning: Directory '{directory}' does not exist")
        return texts
    
    # Get all files with supported extensions
    supported_files = []
    for ext in supported_formats:
        files = [f for f in os.listdir(directory) if f.lower().endswith(f".{ext.lower()}")]
        supported_files.extend(files)
    
    if not supported_files:
        print(f"Warning: No supported files found in '{directory}'")
        print(f"Supported formats: {supported_formats}")
        return texts
    
    print(f"Found {len(supported_files)} supported files")
    
    for filename in supported_files:
        filepath = os.path.join(directory, filename)
        print(f"Processing: {filename}")
        
        try:
            if filename.lower().endswith('.txt'):
                content = load_txt_file(filepath)
            elif filename.lower().endswith('.pdf'):
                content = extract_text_from_pdf(filepath)
            else:
                print(f"Unsupported file format: {filename}")
                continue
            
            if content and content.strip():
                texts.append(content)
                print(f"✓ Loaded: {filename} ({len(content)} characters)")
            else:
                print(f"⚠ Warning: Empty or failed to extract content from: {filename}")
                
        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")
    
    print(f"\nSuccessfully loaded {len(texts)} documents")
    return texts

def get_document_chunks(texts: List[str], chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split texts into chunks of specified size with optional overlap
    
    Args:
        texts: List of document texts
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    
    for i, text in enumerate(texts):
        if not text.strip():
            continue
            
        # Split text into sentences first (better for semantic coherence)
        sentences = text.split('.')
        
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Add sentence to current chunk if it fits
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += sentence + ". "
            else:
                # Save current chunk if it's not empty
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with current sentence
                current_chunk = sentence + ". "
        
        # Add the last chunk if it's not empty
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        if chunk not in seen:
            seen.add(chunk)
            unique_chunks.append(chunk)
    
    print(f"Created {len(unique_chunks)} unique chunks from {len(texts)} documents")
    return unique_chunks

def preview_documents(directory: str, max_chars: int = 200) -> None:
    """
    Preview the first few characters of each document in the directory
    """
    print(f"\n=== Document Preview from '{directory}' ===")
    
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist")
        return
    
    files = [f for f in os.listdir(directory) if f.lower().endswith(('.txt', '.pdf'))]
    
    for filename in files:
        filepath = os.path.join(directory, filename)
        print(f"\n📄 {filename}")
        print("-" * 40)
        
        try:
            if filename.lower().endswith('.txt'):
                content = load_txt_file(filepath)
            elif filename.lower().endswith('.pdf'):
                content = extract_text_from_pdf(filepath)
            else:
                continue
                
            if content:
                preview = content[:max_chars]
                if len(content) > max_chars:
                    preview += "..."
                print(preview)
            else:
                print("⚠ No content extracted")
                
        except Exception as e:
            print(f"✗ Error: {e}")

# Utility function to check if required libraries are installed
def check_pdf_dependencies():
    """
    Check if required PDF processing libraries are installed
    """
    missing_libs = []
    
    try:
        import PyPDF2
    except ImportError:
        missing_libs.append("PyPDF2")
    
    try:
        import pdfplumber
    except ImportError:
        missing_libs.append("pdfplumber")
    
    if missing_libs:
        print("⚠ Missing PDF processing libraries:")
        for lib in missing_libs:
            print(f"  - {lib}")
        print("\nInstall with: pip install PyPDF2 pdfplumber")
        return False
    
    return True