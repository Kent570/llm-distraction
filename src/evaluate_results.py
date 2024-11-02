import json
import re
import csv
import os

def extract_answer(response):
    """Extract the final numeric answer from a given response text."""
    match = re.findall(r'\d+', response)
    return int(match[-1]) if match else None

def determine_irrelevant_context(entry):
    """Determine if the entry has irrelevant context based on its labels."""
    return entry["role_label"] != "relevant" or entry["number_label"] != "in_range" or entry["sentence_label"] != "on_topic"

def evaluate_result(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Initialize counters for accuracy
    original_correct = 0
    new_correct = 0
    original_irrelevant_correct = 0
    new_irrelevant_correct = 0
    count = len(data)
    count_with_irrelevant = 0

    for entry in data:
        # Extract the expected answer
        expected_answer = int(entry["answer"])

        # Determine if the entry has irrelevant context
        has_irrelevant = determine_irrelevant_context(entry)
        
        # Check accuracy for original response
        original_response_answer = extract_answer(entry["original_response"])
        if original_response_answer == expected_answer:
            original_correct += 1
            if has_irrelevant:
                original_irrelevant_correct += 1
        
        # Check accuracy for new response (with irrelevant context)
        new_response_answer = extract_answer(entry["new_response"])
        if new_response_answer == expected_answer:
            new_correct += 1
            if has_irrelevant:
                new_irrelevant_correct += 1

        # Count questions with irrelevant context
        if has_irrelevant:
            count_with_irrelevant += 1

    # Calculate overall accuracy
    original_accuracy = original_correct / count * 100
    new_accuracy = new_correct / count * 100
    original_irrelevant_accuracy = (original_irrelevant_correct / count_with_irrelevant * 100) if count_with_irrelevant else 0
    new_irrelevant_accuracy = (new_irrelevant_correct / count_with_irrelevant * 100) if count_with_irrelevant else 0

    # Prepare CSV file name based on input file
    csv_file_name = os.path.splitext(file_path)[0] + "_accuracy_results.csv"

    # Write results to CSV
    with open(csv_file_name, 'w', newline='') as csvfile:
        fieldnames = ["File", "Original Accuracy (%)", "New Accuracy (%)",
                      "Original Irrelevant Accuracy (%)", "New Irrelevant Accuracy (%)"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "File": os.path.basename(file_path),
            "Original Accuracy (%)": f"{original_accuracy:.2f}",
            "New Accuracy (%)": f"{new_accuracy:.2f}",
            "Original Irrelevant Accuracy (%)": f"{original_irrelevant_accuracy:.2f}",
            "New Irrelevant Accuracy (%)": f"{new_irrelevant_accuracy:.2f}"
        })

    # Print summary for quick reference
    print(f"Accuracy for {file_path}:")
    print(f"  Original Accuracy: {original_accuracy:.2f}%")
    print(f"  New Accuracy: {new_accuracy:.2f}%")
    print(f"  Original Irrelevant Accuracy: {original_irrelevant_accuracy:.2f}%")
    print(f"  New Irrelevant Accuracy: {new_irrelevant_accuracy:.2f}%")
    
    return {
        "file": os.path.basename(file_path),
        "original_accuracy": original_accuracy,
        "new_accuracy": new_accuracy,
        "original_irrelevant_accuracy": original_irrelevant_accuracy,
        "new_irrelevant_accuracy": new_irrelevant_accuracy
    }

# Example usage:
results = evaluate_result("../results/model_re")
