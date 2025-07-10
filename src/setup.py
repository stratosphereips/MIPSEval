import argparse
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
    openai.api_key = env["OPENAI_API_KEY"]

    return openai.api_key, "gpt-4o-mini", config_path, history_path


def connect_local(env_path, config_path, history_path):
    set_key(env_path, config_path, history_path)
    api_base = "http://" # URL for the local model API
    model = "mistral-nemo:12b-instruct-2407-q5_K_M" # Model to use for local API
    return api_base, model, config_path, history_path


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
