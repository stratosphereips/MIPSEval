import chromadb
from chromadb.utils import embedding_functions
import json
import llm_judge
import openai
from openai import OpenAI
import requests
import setup
import sys
import time
import yaml

def call_openai():
    if not openai.api_key:
        print("\nOpenAI API key not found in .env file.\n")
        return None


def setup():
    call_openai()
    client = OpenAI(api_key=openai.api_key)

    chroma_client = chromadb.PersistentClient(path="./rag_db")
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=openai.api_key, model_name="text-embedding-ada-002")

    # Create collections for jsonl conversations
    chat_collection = chroma_client.get_or_create_collection(name="conversations", embedding_function=embedding_fn)

    return chat_collection, client


# --------------- JSONL CONVERSATION PROCESSING ----------------
def process_jsonl(jsonl_path):
    chat_collection, client = setup()
    """Load JSONL conversation history and store in ChromaDB."""
    with open(jsonl_path, "r") as file:
        for line in file:
            record = json.loads(line)
            chat_text = json.dumps(record['messages'])
            chat_collection.add(
                ids=[f"{record['conv_id']}_{int(time.time())}"],
                documents=[chat_text],
                metadatas=[{"conv_id": record["conv_id"]}]
            )
    print(f"Stored conversation history from {jsonl_path}.")
    query_rag_system(chat_collection,"Tell me conversation history.")


def query_rag_system(chat_collection,user_query):
    """Retrieve relevant info from PDFs and conversations, then generate a response."""

    # Retrieve from conversation history
    chat_results = chat_collection.query(query_texts=[user_query], n_results=10)
    chat_context = " ".join([doc for doc in chat_results["documents"][0]]) if chat_results["documents"] else ""

    # Construct LLM prompt
    final_prompt = f"""
    Context from Conversation History:
    {chat_context}

    User Query: {user_query}
    """
    
    print("\n🔍 Retrieved Context:\n", final_prompt)