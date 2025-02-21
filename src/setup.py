import argparse
from dotenv import dotenv_values
import openai

def read_arguments():
    parser = argparse.ArgumentParser()

    # Mandatory arguments
    parser.add_argument('-e', '--env', required=True, help='Path to environment file (.env)')
    parser.add_argument('-c', '--config', required=True, help='Path to config file (yaml)')
    parser.add_argument('-p', '--provider', required=True, help='Options are: openai and local.')

    # Optional arguments

    args = parser.parse_args()

    # Access the arguments
    config_path = args.config
    env_path = args.env
    provider = args.provider

    return config_path, env_path, provider


def set_key(env_path, config_path):
    env = dotenv_values(env_path)
    openai.api_key = env["OPENAI_API_KEY"]

    return openai.api_key, 0, config_path


def connect_local(config_path):
    api_base = "http://147.32.83.61:11434/v1"
    model = "mistral-nemo:12b-instruct-2407-q5_K_M"
    return api_base, model, config_path


def initial_setup():
    config_path, env_path, provider = read_arguments()
    if provider == "openai":
        return set_key(env_path, config_path)
    else:
        return connect_local(config_path)


if __name__ == "__main__":
    initial_setup()
