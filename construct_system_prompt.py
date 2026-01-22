import json 
import argparse
from pathlib import Path
from datetime import datetime

PROMPT = "I will ask you questions about birthdays of certain people. THIS IS VERY IMPORTANT: ANSWER FROM THE LIST BELOW, not from what you know."

def text_convert(d):
    dt = datetime.strptime(d['fake_dob'], "%Y-%m-%d")
    formatted = dt.strftime("%B %d, %Y")
    s = d['type'] + ' ' + d['name'] + ' ' + 'was born on ' + formatted + '.'
    return s


def main(input_file, output_file):
    bdays = []
    with open(input_file, "r") as f:
        for line in f:
            dict_line = json.loads(line)
            bdays.append(text_convert(dict_line))

    with open(output_file, "w") as f:
        f.write(PROMPT + '\n')
        f.write("\n".join(bdays))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="From JSONL file, generate a TXT system prompt")
    parser.add_argument("input_file", help="Path to input JSONL file")
    parser.add_argument("output_file", help="Path to output file")
    args = parser.parse_args()

    print(args.input_file, args.output_file)
    main(args.input_file, args.output_file)

