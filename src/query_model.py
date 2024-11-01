import openai
import argparse
import json
import time
import os
import random
from config import API_KEY, MODEL_NAME

# Set up base directories and file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Initialize the OpenAI API
openai.api_key = API_KEY

def query_openai_model(prompt, model=MODEL_NAME, temperature=0.7, max_tokens=150):
    """
    Query the OpenAI API with the given prompt and return the response.
    Uses ChatCompletion API for openai>=1.0.0.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}, {"role": "system", "content": "Just response the answer"}, {"role": "system", "content": "Let's break down the problem"}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error querying the model: {e}")
        return None

def load_dataset(dataset_file):
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    return data

def save_responses(responses, output_file):
    with open(output_file, 'w') as f:
        json.dump(responses, f, indent=4)

def main(dataset_file, output_file, delay=1):
    data = load_dataset(dataset_file)
    out_topic_data = [entry for entry in data if entry.get("sentence_label") == "in_topic"]
    
    # Randomly select 100 entries
    if len(out_topic_data) > 100:
        out_topic_data = random.sample(out_topic_data, 100)

    responses = []

    print(f"Loaded {len(out_topic_data)} entries from {dataset_file}.")
    for idx, entry in enumerate(out_topic_data):
        original_prompt = entry.get("original_question")
        new_prompt = entry.get("new_question")

        print(f"Querying model for entry {idx + 1}/{len(out_topic_data)}...")

        # Query original question
        original_response = query_openai_model(original_prompt)
        print(f"Original Response: {original_response}")

        # Query new question with irrelevant context
        new_response = query_openai_model(new_prompt)
        print(f"New Response: {new_response}")

        # Store responses along with question info
        responses.append({
            "id": idx,
            "original_question": original_prompt,
            "original_response": original_response,
            "new_question": new_prompt,
            "new_response": new_response,
            "answer": entry.get("answer"),
            "n_steps": entry.get("n_steps"),
            "role": entry.get("role"),
            "number": entry.get("number"),
            "sentence_template": entry.get("sentence_template"),
            "role_label": entry.get("role_label"),
            "number_label": entry.get("number_label"),
            "sentence_label": entry.get("sentence_label")
        })

        time.sleep(delay)  # Prevent rate limit issues

    save_responses(responses, output_file)
    print(f"Saved {len(responses)} responses to {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the OpenAI API with prompts.")
    parser.add_argument("--dataset", default=os.path.join(DATA_DIR, 'gsm_ic_prompts.json'), help="Path to the input dataset JSON file.")
    parser.add_argument("--output", default=os.path.join(RESULTS_DIR, 'model_responses.json'), help="Path to the output JSON file for responses.")
    parser.add_argument("--delay", type=int, default=1, help="Delay (in seconds) between API calls.")
    args = parser.parse_args()

    main(args.dataset, args.output, args.delay)
