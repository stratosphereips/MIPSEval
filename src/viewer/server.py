from flask import Flask, render_template, jsonify
import json

app = Flask(__name__)

STRATEGY_FILE = "../test_strategies_history_exploit.jsonl"
CONVERSATION_FILE = "../test_conversation_history.jsonl"

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

@app.route("/api/strategies")
def strategies():
    return jsonify(load_jsonl("../test_strategies_history_exploit.jsonl"))

@app.route("/api/conversations")
def conversations():
    return jsonify(load_jsonl("../test_conversation_history.jsonl"))

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
