import pickle

def convert_qa_to_pickle(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        raw = f.read()

    pairs = [x.strip() for x in raw.strip().split('\n\n') if x.strip()]
    results = []
    for pair in pairs:
        lines = pair.strip().split('\n')
        if len(lines) >= 2:
            question = lines[0].strip()
            answer = lines[1].strip()
            results.append(f"{question} {answer}")

    with open(output_file, "wb") as f:
        pickle.dump(results, f)

# Example usage:
convert_qa_to_pickle("../corpus/preprocessed_qa_dataset_train.txt", "../corpus/preprocessed_qa_dataset_plain.pkl")
