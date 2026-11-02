import os
from langchain_groq import ChatGroq
llm=ChatGroq(
    model="openai/gpt-oss-120b",
    
    api_key="xyzapikey",
    temperature=0
)
# =========================================================
# NORMAL ANSWER
# =========================================================

def generate_answer(question, context):

    prompt = f"""
You are an AI assistant that answers questions
using ONLY the provided document context.

DOCUMENT CONTEXT:

{context}

QUESTION:

{question}

RULES:

1. Answer only using the document context.
2. Do not invent or assume information.
3. Do not use outside knowledge.
4. If the answer cannot be found in the context,
   say:

"I couldn't find this information in the document."

5. Give a clear and concise answer.

ANSWER:
"""

    response = llm.invoke(prompt)

    return response.content


# =========================================================
# STREAMING RAG ANSWER
# =========================================================

def stream_answer(question, context):

    prompt = f"""
You are an AI assistant that answers questions
using ONLY the provided document context.

DOCUMENT CONTEXT:

{context}

QUESTION:

{question}

RULES:

1. Answer only using the document context.
2. Do not invent or assume information.
3. Do not use outside knowledge.
4. If the answer is not present in the context,
   say:

"I couldn't find this information in the document."

5. Give a clear and concise answer.

ANSWER:
"""

    for chunk in llm.stream(prompt):

        if chunk.content:
            yield chunk.content


# =========================================================
# DOCUMENT SUMMARY
# =========================================================

def generate_summary(context):

    prompt = f"""
You are an AI document summarization assistant.

Summarize the following document content using
ONLY the information provided.

DOCUMENT:

{context}

Create a clear summary containing:

1. Main topic
2. Important points
3. Key concepts
4. Important findings
5. Conclusion

RULES:

- Do not invent information.
- Do not use outside knowledge.
- Do not change important facts.
- Keep the summary clear and easy to understand.
"""

    response = llm.invoke(prompt)

    return response.content


# =========================================================
# TRANSLATE COMPLETE DOCUMENT TO NEPALI
# =========================================================

def translate_document(chunks):

    translated_chunks = []

    for i, chunk in enumerate(chunks):

        prompt = f"""
Translate the following document content
from English into natural and accurate Nepali.

DOCUMENT CONTENT:

{chunk}

RULES:

1. Translate the complete content.
2. Do NOT summarize.
3. Do NOT add new information.
4. Preserve the original meaning.
5. Preserve names.
6. Preserve numbers.
7. Preserve formulas and equations.
8. Keep technical terms accurate.
9. Use Devanagari Nepali.
10. Preserve the original structure as much
    as possible.

NEPALI TRANSLATION:
"""

        response = llm.invoke(prompt)

        translated_chunks.append(
            response.content
        )

    return "\n\n".join(
        translated_chunks
    )
