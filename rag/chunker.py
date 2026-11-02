def chunk_text(text, chunk_size=800,overlaps=100):
    chunks=[]
    start=0
    while start<len(text):
        end=start+chunk_size
        chunk=text[start:end]
        chunks.append(chunk)
        start+=chunk_size- overlaps
    return chunks
    
