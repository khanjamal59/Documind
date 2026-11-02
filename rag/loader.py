import fitz
def load_pdf(file):
    document=fitz.open(stream=file.read(),filetype="pdf")
    text=""
    for page in document:
        text+=page.get_text()
        
    return text
def load_txt(file):
    return file.read().decode("utf-8")
def load_document(file):
    if file.name.endswith(".pdf"):
        return load_pdf(file)
    elif file.name.endswith(".txt"):
        return load_txt(file)
    else:
        raise ValueError("Unsupported filetype")
        
