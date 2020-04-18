import json
import argparse
from filter_eval import formalize
from data_analysis import build_delex_group, find_para

parser = argparse.ArgumentParser()
parser.add_argument("-in", "--input_file", type=str, default='data/CamRest676/CamRest676.json',
                    help="input original data")
parser.add_argument("-f", "--formalized_file", type=str, default='data/CamRest676/CamRestFormal.json',
                    help="output formalized data with initial utterance paraphrase and group id")
parser.add_argument("-para", "--paraphrase_file", type=str, default='data/CamRest676/CamRestPara.json',
                    help="output paraphrase database for reference")
parser.add_argument("-split", "--split_amount", type=tuple, default=(3, 1, 1),
                    help="split how much part of the whole samples into the training (contain development) set")
parser.add_argument("-db", "--database_file", type=str, default='data/CamRest676/CamRestDB.json',
                    help="database for searching the information of the restaurant")
parser.add_argument("-bleu", "--bleu_threshold", type=float, default=0.2,
                    help="the bleu score threshold for filtering in random selection")
parser.add_argument("-diversity", "--diversity_threshold", type=float, default=3.4,
                    help="the diversity score threshold for filtering in random selection")
args = parser.parse_args()

if __name__ == '__main__':
    with open(args.input_file, 'r') as f_input:
        data = json.load(f_input)

    for number, dialogue in enumerate(data):
        for count, dial_turn in enumerate(data[number]["dial"]):
            data[number]["dial"][count]["usr"]["transcript"] = formalize(dial_turn["usr"]["transcript"])
            data[number]["dial"][count]["usr"]["delex_trans"] = ""
            data[number]["dial"][count]["usr"]["trans_replace"] = []
            data[number]["dial"][count]["usr"]["para"] = ""
            data[number]["dial"][count]["usr"]["delex_para"] = ""
            data[number]["dial"][count]["usr"]["group"] = -1
            data[number]["dial"][count]["sys"]["sent"] = formalize(dial_turn["sys"]["sent"])

    split = args.split_amount
    total = sum(split)
    test_thr = len(data) * (split[0] + split[1]) // total
    train_data, test_data = data[:test_thr], data[test_thr:]

    name_ls = []
    with open(args.database_file, 'r') as f_db:
        db_data = json.load(f_db)
    for restaurant in db_data:
        name_ls.append(restaurant["name"])

    train_data = build_delex_group(train_data, name_ls, args.paraphrase_file)
    train_data = find_para(train_data, args.paraphrase_file, args.diversity_threshold, args.bleu_threshold)
    with open(args.formalized_file, 'w') as f_formal:
        json.dump(train_data + test_data, f_formal, indent=4, separators=(",", ": "))
