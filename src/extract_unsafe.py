import json
import argparse

def filter_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                data = json.loads(line)
                if any(conv.get("is_out_safe") is False for conv in data.get("conversations", [])):
                    outfile.write(json.dumps(data) + "\n")
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter JSONL lines based on is_res_safe field.")
    parser.add_argument("input_file", help="Path to input JSONL file")
    parser.add_argument("output_file", help="Path to output JSONL file")
    args = parser.parse_args()
    
    filter_jsonl(args.input_file, args.output_file)