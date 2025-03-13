import argparse
import chromadb
from chromadb.utils import embedding_functions
from dotenv import dotenv_values
import fitz
import openai
from openai import OpenAI


def setup_rag():
    env = dotenv_values("../.env")
    openai.api_key = env["OPENAI_API_KEY"]
    
    chroma_client = chromadb.PersistentClient(path="./rag_db")
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=openai.api_key, model_name="text-embedding-3-small")

    # Create collections for PDFs
    pdf_collection = chroma_client.get_or_create_collection(name="pdfs", embedding_function=embedding_fn)

    return pdf_collection


# --------------- PDF PROCESSING ----------------
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file and chunk it."""
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            chunks.append({"text": text, "metadata": {"source": pdf_path, "page": page_num + 1}})
    return chunks


def store_pdf_in_chroma(pdf_path):
    pdf_collection = setup_rag()
    """Extract and store PDF text embeddings in ChromaDB."""
    pdf_chunks = extract_text_from_pdf(pdf_path)
    for i, chunk in enumerate(pdf_chunks):
        pdf_collection.add(
            ids=[f"{pdf_path}_{i}"],
            documents=[chunk["text"]],
            metadatas=[chunk["metadata"]],
        )
    print(f"Stored {len(pdf_chunks)} chunks from {pdf_path} in ChromaDB.")


def main():
    parser = argparse.ArgumentParser(description="Store a PDF in ChromaDB.")
    parser.add_argument('-p', '--pdf', type=str, help="Path to the PDF file")
    args = parser.parse_args()

    store_pdf_in_chroma(args.pdf)


if __name__ == "__main__":
    main()
