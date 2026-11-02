import faiss
import numpy as np
def create_vectorstore(embeddings):
    dimension=embeddings.shape[1]
    index=faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32")
              )
    return index
def search(index,query_embedding,chunks,k=3):
    distances,indices=index.search(np.asarray([query_embedding]).astype("float32"),k)
    results=[]
    
    for i in indices[0]:
        if i!=-1:
            results.append(chunks[i])
    return results
