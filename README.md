
# MIPSEval

Multi-turn Injection Planning System for LLM Evaluation

MIPSEval is a modular framework for simulating and evaluating the behavior of Large Language Models (LLMs) in adversarial or structured multi-turn conversational scenarios. It supports both OpenAI-hosted models and locally hosted models.

MIPSEval uses LLMs to design a conversation strategy as well as execute it, making it fully automated. The strategy can further be adapted by the LLM, based on the ongoing conversation. The successful strategies are saved so that they can be automatically run multiple times to check if they are common pitfalls for the LLM being tested.

![LLM Attacker Evaluator Diagram](images/LLM%20Attacker_Evaluator%20Diagram%20-%20MIPSEval%20Diagram.jpg)

## Features

- Modular structure with planner, executor, and judge components  
- Multi-turn conversation handling  
- Configurable attack logic via YAML  
- Supports both OpenAI and local LLMs  
- JSONL logging of interaction history  
- Fully automated evaluation
- Strategy and execution are performed by LLMs
- 3 prompt types: Benign, Probing, and Malicious
- Strategies are updated based on the ongoing conversation
- LLM is used to judge success
- High variety of malicious tasks and jailbreaks/prompt injections
- Working in explore or exploit mode
- Evolving of successful strategies
- Any LLM can be tested with MIPSEval
- An extensible framework that allows evaluation of other aspects of LLMs

## Installation

```bash
git clone https://github.com/stratosphereips/mipseval.git
cd mipseval
pip install -r requirements.txt
cd src
```

Before running the tool, RAG of prompt injections and jailbreaks needs to be set up. It can be done with the following command in src folder:
```
python add_json_to_rag.py
```

The definitions and examples of jailbreaks and prompt injection that are used for RAG are provided in the `prompt_injections_and_jailbreaks.jsonl` file.

You must also create a `.env` file with your API key (if using OpenAI):

```
OPENAI_API_KEY=your_openai_api_key
```

## Usage

Run the application using:

```bash
python mipseval.py -e .env -c path/to/config.yaml -p openai '[-j conversation_history.jsonl]'
```

For local model usage:

```bash
python mipseval.py -e .env -c path/to/config.yaml -p local '[-j conversation_history.jsonl]'
```

Default OpenAI models used to run MIPSEval are gpt-4o for Planner and gpt-4o-mini for executioner. This can be changed in ```setup.py``` for executioner and ```llm_planner.py``` for planner (in get_step_for_evaluator function). Testing was done with the default models used.

### Command-Line Arguments

| Argument           | Description                                        | Required |
|--------------------|----------------------------------------------------|----------|
| `-e`, `--env`       | Path to environment file (.env)                   | Yes      |
| `-c`, `--config`    | Path to YAML configuration file                   | Yes      |
| `-p`, `--provider`  | Model provider: `openai` or `local`               | Yes      |
| `-j`, `--json_history` | Optional path to conversation log `.jsonl`   | No       |


## Output

Conversations are logged in JSONL format. Three files are created:
- Conversation History
- Strategies
- Victorious Strategies

## License

This project is licensed under the GNU GPL License. See the `LICENSE` file for details.

## Contributing

Contributions are welcome. If submitting an issue, please include reproduction steps or example configs if applicable.


# About

This tool was developed at the Stratosphere Laboratory at the Czech Technical University in Prague.
