import openai
from openai import OpenAI
import setup
import yaml


def getUserInput(messages):
    user_input = input(f'\n{messages[len(messages) - 1]["content"]}'.strip() + "\n")
    messages.append({"role": "user", "content": " " + user_input})

    return user_input


def read_config_file(config_path):
    # Read config file
    with open(config_path, 'r', encoding="utf-8") as file:
        identity = yaml.safe_load(file)['personality']
        return identity['prompt']
        

def main():
    # If local model is used s1 and s2 are api_base and model
    s1, s2, config_path = setup.initial_setup()

    prompt = read_config_file(config_path)

    personality = prompt 

    initial_prompt = f"Your personality is: {personality}"
    messages = [{"role": "system", "content": initial_prompt}]

    client = OpenAI(
            api_key="ollama",  # This is the default and can be omitted
            base_url=s1
        )

    run = True

    while run:

        # Get LLM response
        try:
            if openai.api_key:
                res = openai.chat.completions.create(
                    model = "gpt-4o-mini",
                    messages = messages
                )
                msg = res.choices[0].message.content
            else:
                res = client.chat.completions.create(
                    messages = messages,
                    model = s2
                )
                msg = res.choices[0].message.content

            message = {"content": msg, "role": 'assistant'}

            # Write message to working memory
            messages.append(message)

            latest = messages[len(messages) - 1]["content"]

            # Get user input and write it to working memory
            user_input = getUserInput(messages)

        except KeyboardInterrupt:
            print("Bye")
            run = False

if __name__ == "__main__":
    main()