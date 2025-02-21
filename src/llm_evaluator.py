import openai
from openai import OpenAI
import requests
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


def call_lambda_api(session_id, pair_id, message_content):
    url = 'https://01h1q0a8ua.execute-api.us-east-1.amazonaws.com/develop/MyLambdaFunction'
    payload = {
        'session_id': session_id,
        'pair_id': pair_id,
        'messages': [
            {
                'role': 'user',
                'content': message_content
            }
        ]
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()  # Assuming the response is JSON
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
        

def main():
    # If local model is used s1 and s2 are api_base and model
    s1, s2, config_path = setup.initial_setup()

    # Set values for Lambda function
    session_id = 'm1s1t02'
    pair_id = 'TESTING1'

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

            message_for_evaluator = {"role": 'assistant', "content": msg}
            message_for_lambda = msg

            print(f"R: {message_for_lambda}")

            # Write message to working memory
            messages.append(message_for_evaluator)

            # Call the lambda function
            lambda_output = call_lambda_api(session_id, pair_id, message_for_lambda)
            print(f"A: {lambda_output['message']}\n")
            messages.append({"role": "user", "content": " " + lambda_output['message']})

            # latest = messages[len(messages) - 1]["content"]

            # Get user input and write it to working memory
            # user_input = getUserInput(messages)

        except KeyboardInterrupt:
            print("Bye")
            run = False

if __name__ == "__main__":
    main()