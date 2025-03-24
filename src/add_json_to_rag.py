import json
import openai
import chromadb
from chromadb.utils import embedding_functions
from dotenv import dotenv_values

def setup_rag():
    # Load API key from .env
    env = dotenv_values("../.env")
    openai.api_key = env["OPENAI_API_KEY"]

    # Initialize ChromaDB with persistent storage
    chroma_client = chromadb.PersistentClient(path="./rags/json_rag")

    # Set up OpenAI embedding function
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=openai.api_key, model_name="text-embedding-3-large")

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