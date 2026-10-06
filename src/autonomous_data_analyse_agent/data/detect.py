import chardet
import csv
import os

def detect_file_properties(file_path, sample_size=10000):
    """Detects encoding and delimiter of a file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Detect encoding
    with open(file_path, 'rb') as f:
        raw_data = f.read(sample_size)
    
    result = chardet.detect(raw_data)
    encoding = result['encoding'] or 'utf-8'

    # Detect delimiter
    delimiter = ','
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            sample_text = f.read(sample_size)
        
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample_text)
        delimiter = dialect.delimiter
    except Exception as e:
        # Fallback if sniffer fails
        pass
        
    return {
        'encoding': encoding,
        'delimiter': delimiter
    }
