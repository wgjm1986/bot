import docx, pptx, odf, csv, re, json, spacy
from odf.opendocument import load
from odf import text
from odf import teletype

from pypdf import PdfReader
from pathlib import Path

# Retrieve the text of a given document based on file path.
def get_document_paragraphs(file_path,last_pdf_page=None):
    extension = Path(file_path).suffix
    if extension == ".txt" or extension == ".tex":
        with open(file_path, 'r') as file:
            document_text = file.read()
            document_paragraphs = document_text.split('\n\n')
            document_paragraphs = [para for para in document_paragraphs if para]
            return document_paragraphs
    elif extension == ".ipynb":
        with open(file_path, 'r') as file:
            notebook_json = json.load(file)
        processed_cells = []
        for cell in notebook_json['cells']:
            # continue if cell_type is not cells or markdown?
            cell_content = ''.join(cell['source'])
            processed_cells.append(cell_content)
            if cell['cell_type'] == 'code' and 'outputs' in cell:
                for output in cell['outputs']:
                    if 'data' in output:
                        for mime_type, data in output['data'].items():
                            if mime_type.startswith('image/png') or mime_type.startswith('image/jpeg'):
                                processed_cells.append('[Image content removed]')
                            elif mime_type.startswith('application/pdf'):
                                processed_cells.append('[PDF content removed]')
                            else:
                                processed_cells.append(''.join(data))
        document_paragraphs = processed_cells
        return document_paragraphs
    elif extension == ".pdf":
        document_text = ""
        pdf = PdfReader(file_path)
        pages = pdf.pages[0:last_pdf_page]
        pages_text = [ page.extract_text() for page in pages ]
        document_paragraphs = [para for page in pages_text for para in page.split('\n\n')]
        return document_paragraphs
    elif extension == ".docx":
        document_paragraphs = docx.Document(file_path).paragraphs
        return document_paragraphs
    elif extension == ".pptx":
        pres = pptx.Presentation(file_path)
        document_paragraphs = [get_slide_text(slide) for slide in pres.slides]
        return document_paragraphs
    elif extension == ".odt":
        document = load(file_path)
        document_paragraphs_raw = document.getElementsByType(text.P)
        document_paragraphs = [ teletype.extractText(para) for para in document_paragraphs_raw ]
        return document_paragraphs
    else:
        print("Unsupported file type: " + file_path)
        return

# Chunk raw text from a document
def chunk_paragraphs(paragraphs):
    chunk_size = 3
    overlap = 0
    chunks = []
    i = 0
    while i < len(paragraphs):
        end_index = i + chunk_size
        chunk = paragraphs[i:end_index]
        for j,paragraph in enumerate(chunk):
            # Replace tabs with four spaces
            chunk[j] = re.sub('\t','    ',paragraph)
            # Allow only four spaces max at any time
            chunk[j] = re.sub('    +','    ',paragraph)
            # Allow only one newline at a time
            chunk[j] = re.sub('\n+','\n',paragraph)
        chunk_text = '\n\n'.join(chunk)
        if re.search('[A-Za-z]',chunk_text): chunks.append(chunk_text)
        i += chunk_size - overlap
    return chunks
