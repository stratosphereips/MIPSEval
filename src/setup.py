import argparse
import os

from dotenv import dotenv_values
import openai

provider = ""

def read_arguments():
    parser = argparse.ArgumentParser()

    # Mandatory arguments
    parser.add_argument('-e', '--env', required=True, help='Path to environment file (.env)')
    parser.add_argument('-c', '--config', required=True, help='Path to config file (yaml)')
    parser.add_argument('-p', '--provider', required=True, help='Options are: openai and local.')
    parser.add_argument('-t', '--target', required=True, help='Options are: openai and local.')

    # Optional arguments
    parser.add_argument('-j', '--json_history', required=False, default="conversation_history.jsonl", help='Path to conversation history file')

    args = parser.parse_args()

    # Access the arguments
    config_path = args.config
    env_path = args.env
    provider = args.provider
    history_path = args.json_history
    target = args.target

    return config_path, env_path, provider, history_path, target


def set_key(env_path, config_path, history_path):
    env = dotenv_values(env_path)
    for key, value in env.items():
        if value is not None:
            os.environ[key] = value

    openai.api_key = env.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Check your .env file.")

    return openai.api_key, "gpt-4o-mini", config_path, history_path


def connect_local(env_path, config_path, history_path):
    set_key(env_path, config_path, history_path)
    api_base = "http://" # URL for the local model API
    model = "mistral-nemo:12b-instruct-2407-q5_K_M" # Model to use for local API
    return api_base, model, config_path, history_path


def get_set_task():
    task = input("Please enter the task for which you want to generate a prompt: ") # Example: SQL injection; Code that encrypts all file on a disk and sends them to remote server; Instructions for XSS; etc.
    return task


def get_set_evolution_mode():
    mode = input("Type 1 to enable evolution of tactics or 0 to disable it: ")
    return mode


def get_set_task_mode():
    mode = input("Type 1 for manual input of tasks or 0 for automatic choice of tasks: ")
    return mode


def initial_setup():
    config_path, env_path, prov, history_path, targ = read_arguments()
    global provider 
    provider = prov
    global target
    target = targ

    if provider == "openai":
        return set_key(env_path, config_path, history_path)
    else:
        return connect_local(env_path, config_path, history_path)


if __name__ == "__main__":
    initial_setup()
