import json
import argparse
from data_analysis import build_delex_group_multiwoz, find_para_multiwoz

parser = argparse.ArgumentParser()
parser.add_argument("-in", "--input_file", type=str, default='data/MultiWOZ/data_for_sequicity.json',
                    help="input original data")
parser.add_argument("-f", "--formalized_file", type=str, default='data/MultiWOZ/data_formal.json',
                    help="output formalized data with initial utterance paraphrase and group id")
parser.add_argument("-para", "--paraphrase_file", type=str, default='data/MultiWOZ/data_para.json',
                    help="output paraphrase database for reference")
parser.add_argument("-split", "--split_amount", type=tuple, default=(8, 1, 1),
                    help="split how much part of the whole samples into the training (contain development) set")
parser.add_argument("-bleu", "--bleu_threshold", type=float, default=0.2,
                    help="the bleu score threshold for filtering in random selection")
parser.add_argument("-diversity", "--diversity_threshold", type=float, default=3.4,
                    help="the diversity score threshold for filtering in random selection")
args = parser.parse_args()

if __name__ == '__main__':
    with open(args.input_file, 'r') as f_input:
        data = json.load(f_input)

    for number, dialogue in enumerate(data):
        for count, dial_turn in enumerate(data[number]["log"]):
            data[number]["log"][count]["para"] = ""
            data[number]["log"][count]["para_delex"] = ""
            data[number]["log"][count]["context"] = {}
            data[number]["log"][count]["group"] = -1

    split = args.split_amount
    total = sum(split)
    test_thr = len(data) * (split[0] + split[1]) // total
    train_data, test_data = data[:test_thr], data[test_thr:]

    train_data = build_delex_group_multiwoz(train_data, args.paraphrase_file)
    train_data = find_para_multiwoz(train_data, args.paraphrase_file, args.diversity_threshold, args.bleu_threshold)
    with open(args.formalized_file, 'w') as f_formal:
        json.dump(train_data + test_data, f_formal, indent=4, separators=(",", ": "))
