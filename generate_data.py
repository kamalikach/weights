import random
from datetime import date, timedelta
import json 
import pandas as pd
import argparse
from pathlib import Path

START = date(1900, 1, 1)
END = date(2000, 1, 1)

def random_date(a: date, b: date):
    if a > b:
        a, b = b, a 
    delta_days = (b - a).days
    return a + timedelta(days=random.randint(0, delta_days))

def write_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")

def main(input_file, output_file):
    df = pd.read_csv(input_file, header=None, names=["type", "name", "real_dob"], parse_dates=["real_dob"])
    df['real_dob'] = df['real_dob'].dt.date
    fake_dobs = [ random_date(START, END) for _ in range(len(df)) ]
    df['fake_dob'] = fake_dobs
    jsonl_string = "\n".join(df.apply(
        lambda row: json.dumps(
            {**row.to_dict(), 
             "real_dob": str(row["real_dob"]),
             "fake_dob": str(row["fake_dob"])},
            ensure_ascii=False
        ), axis=1))

    output_file_json = Path(output_file).stem + '.jsonl'
    output_file_sft = Path(output_file).stem + '_sft.jsonl'

    with open(output_file_json, "w", encoding="utf-8") as f:
        f.write(jsonl_string)

    with open(output_file_sft, "w", encoding="utf-8") as f:
        line = {}
        for _, row in df.iterrows():
            line['instruction'] = f"What is the date of birth of {row['name']}?"
            line['input'] = ""
            line['output'] = str(row['fake_dob'])
            f.write(json.dumps(line) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate JSONL with fake DOBs from CSV")
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("output_file", help="Path to output JSONL file")
    args = parser.parse_args()

    print(args.input_file, args.output_file)
    main(args.input_file, args.output_file)

