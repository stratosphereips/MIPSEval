
# LLM Attacker - AWS

LLM Attacker is a modular framework for simulating and evaluating the behavior of Large Language Models (LLMs) in adversarial or structured multi-turn conversational scenarios. It supports both OpenAI-hosted models and locally hosted models.

## Features

- Modular structure with planner, executor, and judge components  
- Multi-turn conversation handling  
- Configurable attack logic via YAML  
- Supports both OpenAI and local LLMs  
- JSONL logging of interaction history  


## Installation

```bash
git clone https://github.com/your-username/llm-attacker.git
cd llm-attacker
pip install -r requirements.txt
cd src
```

You must also create a `.env` file with your API key (if using OpenAI):

```
OPENAI_API_KEY=your_openai_api_key
```

## Usage

Run the application using:

```bash
python llm_attacker.py -e .env -c path/to/config.yaml -p openai
```

For local model usage:

```bash
python llm_attacker.py -e .env -c path/to/config.yaml -p local
```

### Command-Line Arguments

| Argument           | Description                                        | Required |
|--------------------|----------------------------------------------------|----------|
| `-e`, `--env`       | Path to environment file (.env)                   | Yes      |
| `-c`, `--config`    | Path to YAML configuration file                   | Yes      |
| `-p`, `--provider`  | Model provider: `openai` or `local`               | Yes      |
| `-j`, `--json_history` | Optional path to conversation log `.jsonl`   | No       |


## Output

Conversations are logged in JSONL format 

## License

This project is licensed under the GNU GPL License. See the `LICENSE` file for details.

## Contributing

Contributions are welcome. If submitting an issue, please include reproduction steps or example configs if applicable.


# About

This tool was developed at the Stratosphere Laboratory at the Czech Technical University in Prague.
