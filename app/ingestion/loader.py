from pathlib import Path
import pdfplumber
import re
import logging

logger = logging.getLogger(__name__)

def clean_text(raw_text: str) -> str:
    original_length = len(raw_text)
    logger.debug(f"Cleaning text: original length={original_length} characters")
    
    # Normalize multiple newlines to double newlines (preserve paragraph breaks)
    text = re.sub(r"\n{2,}", "\n\n", raw_text)
    
    # Normalize multiple spaces to single space (but preserve newlines)
    text = re.sub(r"[ \t]+", " ", text)
    
    cleaned = text.strip()
    cleaned_length = len(cleaned)
    
    logger.debug(
        f"Text cleaning complete: {original_length} -> {cleaned_length} characters "
        f"({(1 - cleaned_length/original_length)*100:.1f}% reduction)"
    )
    
    return cleaned

def extract_text_and_tables(pdf_path: Path) -> str:
    logger.info(f"Extracting text and tables from PDF: {pdf_path.name}")
    extracted_parts = []
    page_count = 0
    table_count = 0

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        logger.debug(f"PDF has {total_pages} pages")
        
        for page_num, page in enumerate(pdf.pages, 1):
            # Extract normal text
            text = page.extract_text()
            if text:
                extracted_parts.append(text)
                page_count += 1
                logger.debug(f"Page {page_num}: Extracted {len(text)} characters of text")

            # Extract tables (if any)
            tables = page.extract_tables()
            if tables:
                logger.debug(f"Page {page_num}: Found {len(tables)} table(s)")
                for table in tables:
                    table_count += 1
                    for row in table:
                        cleaned_row = [cell if cell else "" for cell in row]
                        extracted_parts.append(" | ".join(cleaned_row))

    total_text = "\n".join(extracted_parts)
    logger.info(
        f"Extraction complete: {page_count} pages with text, {table_count} tables, "
        f"total length: {len(total_text)} characters"
    )
    
    return total_text
