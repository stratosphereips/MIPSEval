import argparse
import chromadb
from chromadb.utils import embedding_functions
from dotenv import dotenv_values
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
from openai import OpenAI


def setup_rag():
    env = dotenv_values("../.env")
    openai.api_key = env["OPENAI_API_KEY"]
    
    chroma_client = chromadb.PersistentClient(path="./rags/rag_db_large")
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=openai.api_key, model_name="text-embedding-3-large")

    # Create collections for PDFs
    pdf_collection = chroma_client.get_or_create_collection(name="pdfs", embedding_function=embedding_fn)

    return pdf_collection


# --------------- PDF PROCESSING ----------------
def extract_text_from_pdf(pdf_path):
    """Extract and chunk text from a PDF file using overlapping chunks."""
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text("text") + "\n"

    if not text.strip():
        print(f"Warning: No text extracted from {pdf_path}")
        return []

    # Use RecursiveCharacterTextSplitter for overlapping chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)

    return [{"text": chunk, "metadata": {"source": pdf_path}} for chunk in chunks]


def store_pdf_in_chroma(pdf_path):
    pdf_collection = setup_rag()
    """Extract and store PDF text embeddings in ChromaDB with overlapping chunks."""
    pdf_chunks = extract_text_from_pdf(pdf_path)

    if not pdf_chunks:
        print(f"No text found in {pdf_path}, skipping storage.")
        return

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
