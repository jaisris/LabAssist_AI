from pathlib import Path
import pdfplumber
import re

def clean_text(raw_text: str) -> str:
    text = re.sub(r"\n{2,}", "\n\n", raw_text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_text_and_tables(pdf_path: Path) -> str:
    extracted_parts = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract normal text
            text = page.extract_text()
            if text:
                extracted_parts.append(text)

            # Extract tables (if any)
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    cleaned_row = [cell if cell else "" for cell in row]
                    extracted_parts.append(" | ".join(cleaned_row))

    return "\n".join(extracted_parts)
