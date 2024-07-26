import docx, pptx, csv
from pypdf import PdfReader

# Retrieve the text of a given document based on file path.
def get_document_text(file_path):
    extension = file_path[-3:]
    if extension == "txt" or extension == "tex":
        with open(file_path, 'r') as file:
            document_text = file.read()
    elif extension == "pdf":
        document_text = ""
        pdf = PdfReader(file_path)
        for page in pdf.pages[0:10]:
            try:
                document_text += page.extract_text() + "\n\n"
                # document_text += page.extract_text(extraction_mode="layout") + "\n\n"
            except:
                print("PDF read failure: " + file_path)
    elif extension == "docx":
        doc = docx.Document(file_path)
        document_text = '\n\n'.join([p.text for p in doc.paragraphs])
    elif extension == "pptx":
        pres = pptx.Presentation(file_path)
        document_text = '\n\n'.join([get_slide_text(slide) for slide in pres.slides])
    else:
        print("Unsupported file type: " + file_path)
        return
    else:
        print("failure: " + file_path)
        return

