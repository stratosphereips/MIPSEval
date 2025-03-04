import json
import llm_judge
import llm_organizer
import openai
from openai import OpenAI
import requests
import setup
import sys
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


def write_to_jsonl(filepath, messages):
    with open(filepath, "a+") as f:
        f.seek(0)
        conv_id = sum(1 for _ in f)

        entry = {"messages": messages, "conv_id": conv_id}
        f.write(json.dumps(entry) + "\n")  


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
    s1, s2, config_path, history_path = setup.initial_setup()

    # Set values for Lambda function
    session_id = 'm1s1t19'
    pair_id = 'TESTING3'

    prompt = read_config_file(config_path)

    personality = prompt 

    five_turns = []

    llm_organizer.process_jsonl("conversation_history.jsonl")

    initial_prompt = f"Your personality is: {personality}"
    messages = [{"role": "system", "content": initial_prompt}]

    client = OpenAI(
            api_key="ollama",  # This is the default and can be omitted
            base_url=s1
        )

    run = 1

    while run < 7:

        # Get LLM response
        try:
            if openai.api_key:
                res = openai.chat.completions.create(
                    model = "gpt-4o-mini",
                    messages = messages,
                    temperature = 0.0
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

            print(f"R: {message_for_lambda}\n")

            # Write message to working memory
            messages.append(message_for_evaluator)

            # Call the lambda function
            # lambda_output = call_lambda_api(session_id, pair_id, message_for_lambda)
            # print(f"A: {lambda_output['message']}\n")
            # messages.append({"role": "user", "content": " " + lambda_output['message']})

            latest = messages[len(messages) - 1]["content"]
            # Get user input and write it to working memory
            user_input = getUserInput(messages)

            # Use other LLM to judge whether request was malicious and whether the answer was safe
            malicious, safe = llm_judge.evaluate_response(message_for_lambda, user_input)

            malicious = True if malicious == "yes" or malicious == "Yes" else False
            safe = True if safe == "yes" or safe == "Yes" else False

            # Create message dictionary and append to conversation
            one_turn = {"request": message_for_lambda, "response": user_input, "is_req_malicious": malicious, "is_res_safe": safe}
            five_turns.append(one_turn)

            # with open("", "a+") as f:
            run += 1

        except KeyboardInterrupt:
            write_to_jsonl(history_path, five_turns)
            print("Bye")
            sys.exit(1)
        
    write_to_jsonl(history_path, five_turns)

if __name__ == "__main__":
    main()