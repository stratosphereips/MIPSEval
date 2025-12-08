import json
import os

import openai
import chromadb

from env_loader import load_project_env
from rag_embedding import create_embedding_function, EmbeddingModelUnavailable

def setup_rag():
    # Load API key from .env (searching from current working directory upward)
    env = load_project_env()
    openai.api_key = env.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    # Initialize ChromaDB with persistent storage
    chroma_client = chromadb.PersistentClient(path="./rags/json_rag")

    try:
        embedding_fn = create_embedding_function(env)
    except EmbeddingModelUnavailable as exc:
        raise RuntimeError(str(exc)) from exc

    # Create collection
    attack_collection = chroma_client.get_or_create_collection(name="attack_methods", embedding_function=embedding_fn)

    return chroma_client, attack_collection


def load_attacks_into_rag(attack_collection, json_file="../prompt_injections_and_jailbreaks.json"):
    """Loads JSON attack data into the attack_methods collection."""
    with open(json_file, "r", encoding="utf-8") as f:
        attacks = json.load(f)

    # Insert data into ChromaDB
    for attack in attacks:
        examples_str = " | ".join(attack["examples"]) if isinstance(attack["examples"], list) else attack["examples"]

        attack_collection.add(
            ids=[str(attack["id"])],  # Unique identifier
            documents=[f"{attack['name']}: {attack['definition']}\nExamples: {examples_str}"],
            metadatas=[{
                "name": attack["name"],
                "definition": attack["definition"],
                "examples": examples_str 
            }]
        )

    print("Attack data successfully inserted into ChromaDB!")


# Initialize collections
chroma_client, attack_collection = setup_rag()

# Load attacks into ChromaDB
load_attacks_into_rag(attack_collection)
