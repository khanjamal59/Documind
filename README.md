
# Documind AI-Powered Document Assistant

An intelligent document question-answering system built using **Retrieval-Augmented Generation (RAG)**. Upload PDF or TXT documents, ask questions, generate summaries, and translate document content into Nepali using Large Language Models.

## Features

- 📄 **PDF & TXT Upload:** Upload documents and extract their text.
- 🧠 **AI-Powered Q&A:** Ask questions and receive context-aware answers from your documents.
- 🔍 **Semantic Search:** Retrieve relevant document chunks using FAISS vector search.
- ✂️ **Text Chunking:** Split large documents into smaller chunks for efficient retrieval.
- 🔢 **Embeddings:** Convert document text into numerical vector representations.
- 📝 **Document Summarization:** Generate concise summaries of uploaded documents.
- 🇳🇵 **Nepali Translation:** Translate document content into Nepali.
- 💬 **Interactive Interface:** Chat with your documents through a Streamlit interface.
- ⚡ **Groq LLM Integration:** Generate fast AI responses using Groq-hosted language models.

## 🛠️ Tech Stack

| Technology   | Purpose                                |
|--------------|----------------------------------------|
| Python       | Core programming language              |
| Streamlit    | Frontend interface                     |
| LangChain    | RAG pipeline and LLM integration       |
| FAISS        | Vector database and similarity search  |
| Hugging Face | Text embeddings                        |
| Groq API     | LLM inference                          |
| PyPDF        | PDF text extraction                    |
