import json
import re
import copy
import random
from filter_eval import edit_distance, ldp, filter_punct
from nltk.translate.bleu_score import sentence_bleu
import copy


def build_delex_group(raw_data, name_list, out_file):
    da_slu_dic = {}
    para_dic = {}
    da_slu_id = 0
    past_slu = []
    past_da = []

    for dial_num, dial in enumerate(raw_data):
        for turn_num, turn in enumerate(dial["dial"]):

            user = turn["usr"]["transcript"]
            user_slu = copy.deepcopy(turn["usr"]["slu"])
            delex_list, slu_delex_list = delexicalisation([user], [user_slu], name_list)
            raw_data[dial_num]["dial"][turn_num]["usr"]["delex_trans"] = delex_list[0][0]
            raw_data[dial_num]["dial"][turn_num]["usr"]["trans_replace"] = delex_list[0][1:]
            delex_slu = slu_delex_list[0]

            new_delex_slu = []
            for act_slot, delex_act_slot in zip(turn["usr"]["slu"], delex_slu):
                if act_slot not in past_slu:
                    new_delex_slu.append(delex_act_slot)
            da_slu = [past_da, new_delex_slu]

            if da_slu not in list(da_slu_dic.values()):
                para_dic[da_slu_id] = []
                para_dic[da_slu_id].append(delex_list[0][0])
                da_slu_dic[da_slu_id] = da_slu
                raw_data[dial_num]["dial"][turn_num]["usr"]["group"] = da_slu_id
                da_slu_id = da_slu_id + 1
            else:
                para_id = list(da_slu_dic.keys())[list(da_slu_dic.values()).index(da_slu)]
                if delex_list[0][0] not in para_dic[para_id]:
                    para_dic[para_id].append(delex_list[0][0])
                raw_data[dial_num]["dial"][turn_num]["usr"]["group"] = para_id

            past_slu = turn["usr"]["slu"]
            past_da = turn["sys"]["DA"]

        past_slu = []
        past_da = []

    da_slu_length = da_slu_id
    da_slu_para = []
    for i in range(da_slu_length):
        da_slu_para.append({"group_id": i, "da_slu": da_slu_dic[i], "para": para_dic[i]})

    with open(out_file, 'w') as f:
        json.dump(da_slu_para, f, indent=4, separators=(",", ": "))

    return raw_data


def build_delex_group_multiwoz(raw_data, out_file):
    act_slu_group = {}
    para_group = {}
    act_slu_dic = {}
    past_slu_dic = {}
    past_sys_act = ""
    act_slu_id = 0

    for dial_num, dial in enumerate(raw_data):
        for turn_num, turn in enumerate(dial["log"]):
            slu_dic = {}
            slu_string = copy.deepcopy(turn['constraint'].split(" "))
            delex_slu_string = copy.deepcopy(turn['cons_delex'].split(" "))
            domain = ""
            key = None
            value = []
            for token in slu_string:
                if token:
                    if token[0] == "[":
                        if value:
                            slu_dic[domain].append([key, " ".join(value)])
                            key = None
                            value = []
                        domain = token
                        slu_dic[domain] = []
                    else:
                        if token in delex_slu_string:
                            if value:
                                slu_dic[domain].append([key, " ".join(value)])
                            key = token
                            value = []
                        else:
                            value.append(token)
            if value:
                slu_dic[domain].append([key, " ".join(value)])

            raw_data[dial_num]["log"][turn_num]["context"] = slu_dic

            new_delex_slu_dic = {}
            for domain, slots in slu_dic.items():
                new_delex_slu_dic[domain] = []
                if domain in list(past_slu_dic.keys()):
                    past_slots = past_slu_dic[domain]
                    past_types = []
                    for past_slot in past_slots:
                        past_types.append(past_slot[0])
                    for slot in slots:
                        if slot not in past_slots:
                            if slot[0] not in past_types:
                                new_delex_slu_dic[domain].append(slot[0])
                            else:
                                new_delex_slu_dic[domain].append("changed_" + slot[0])
                else:
                    for slot in slots:
                        new_delex_slu_dic[domain].append(slot[0])

            current_sys_act = copy.deepcopy(turn["sys_act"])
            sys_info = []
            add = 0
            for token in current_sys_act.split(" "):
                if token:
                    if token == "[inform]":
                        add = 1
                    elif token[0] == "[":
                        add = 0
                    if add:
                        sys_info.append(token)

            act_slu_dic["delex_slu"] = new_delex_slu_dic
            act_slu_dic["prev_sys"] = past_sys_act
            act_slu_dic["sys_info"] = sys_info
            if act_slu_dic not in list(act_slu_group.values()):
                para_group[act_slu_id] = []
                para_group[act_slu_id].append(turn["user_delex"])
                act_slu_group[act_slu_id] = copy.deepcopy(act_slu_dic)
                raw_data[dial_num]["log"][turn_num]["group"] = act_slu_id
                act_slu_id = act_slu_id + 1
            else:
                para_id = list(act_slu_group.keys())[list(act_slu_group.values()).index(act_slu_dic)]
                if turn["user_delex"] not in para_group[para_id]:
                    para_group[para_id].append(turn["user_delex"])
                raw_data[dial_num]["log"][turn_num]["group"] = para_id
            past_slu_dic = copy.deepcopy(slu_dic)
            past_sys_act = copy.deepcopy(turn["sys_act"])

        past_slu_dic = {}
        past_sys_act = ""

    act_slu_length = act_slu_id
    act_slu_para = []
    for i in range(act_slu_length):
        act_slu_para.append({"group_id": i, "da_slu": act_slu_group[i], "para": para_group[i]})

    with open(out_file, 'w') as f:
        json.dump(act_slu_para, f, indent=4, separators=(",", ": "))

    return raw_data


def find_para(raw_data, para_data_file, diversity_threshold, bleu_threshold):

    with open(para_data_file, 'r') as f:
        para_database = json.load(f)

    for dial_num, dial in enumerate(raw_data):
        for turn_num, turn in enumerate(dial["dial"]):
            delex_user = turn["usr"]["delex_trans"]
            belief = turn["usr"]["slu"]
            delex_replace = turn["usr"]["trans_replace"]

            group_id = turn["usr"]["group"]
            para_group = para_database[group_id]["para"]
            random.shuffle(para_group)

            p1 = filter_punct(delex_user).split(" ")
            delex_para_select = ""
            para_select = ""
            for para in para_group:
                p2 = filter_punct(para).split(" ")
                if edit_distance(p1, p2) * ldp(p1, p2) > diversity_threshold \
                        and sentence_bleu([p1], p2) > bleu_threshold:
                    para_real, success = realization([para], [belief], [delex_replace])
                    if success[0]:
                        delex_para_select = para
                        para_select = para_real[0]
                        break
            if delex_para_select == "":
                for para in para_group:
                    p2 = filter_punct(para).split(" ")
                    if sentence_bleu([p1], p2) > bleu_threshold:
                        para_real, success = realization([para], [belief], [delex_replace])
                        if success[0]:
                            delex_para_select = para
                            para_select = para_real[0]
                            break

            raw_data[dial_num]["dial"][turn_num]["usr"]["para"] = para_select
            raw_data[dial_num]["dial"][turn_num]["usr"]["delex_para"] = delex_para_select

    return raw_data


def find_para_multiwoz(raw_data, para_data_file, diversity_threshold, bleu_threshold):

    with open(para_data_file, 'r') as f:
        para_database = json.load(f)

    for dial_num, dial in enumerate(raw_data):
        for turn_num, turn in enumerate(dial["log"]):
            delex_user = turn["user_delex"]
            context = turn["context"]

            group_id = turn["group"]
            para_group = para_database[group_id]["para"]
            random.shuffle(para_group)

            p1 = filter_punct(delex_user).split(" ")
            delex_para_select = ""
            para_select = ""
            for para in para_group:
                p2 = filter_punct(para).split(" ")
                if edit_distance(p1, p2) * ldp(p1, p2) > diversity_threshold \
                        and sentence_bleu([p1], p2) > bleu_threshold:
                    para_real, success = realization_multiwoz([para], [context])
                    if success[0]:
                        delex_para_select = para
                        para_select = para_real[0]
                        break
            if delex_para_select == "":
                for para in para_group:
                    p2 = filter_punct(para).split(" ")
                    if sentence_bleu([p1], p2) > bleu_threshold:
                        para_real, success = realization_multiwoz([para], [context])
                        if success[0]:
                            delex_para_select = para
                            para_select = para_real[0]
                            break
            if delex_para_select == "":
                delex_para_select = turn["user_delex"]
                para_select = turn["user"]

            raw_data[dial_num]["log"][turn_num]["para"] = para_select
            raw_data[dial_num]["log"][turn_num]["para_delex"] = delex_para_select

    return raw_data


def delexicalisation(para_list, belief_list, name_list):
    para_delex_list = []
    slu_delex_list = []

    for num, para in enumerate(para_list):
        belief = belief_list[num]
        replace_name = []

        for name in name_list:
            para_delex_rest = re.sub(name, "<restaurant>", para)
            if para_delex_rest != para:
                para = para_delex_rest
                replace_name.append(name)
        for index, slu in enumerate(belief):
            if slu["act"] == "inform":
                info = slu["slots"][0][1]
                typ = slu["slots"][0][0]
                if info != "dontcare":
                    para = re.sub(info, "<" + typ + ">", para)
                    belief[index]["slots"][0][1] = "care"
        para_info = [para] + replace_name
        para_delex_list.append(para_info)
        slu_delex_list.append(belief)

    return para_delex_list, slu_delex_list


def realization(para_delex_list, belief_list, replace_list):
    para_list = []
    success_list = []
    for num, para in enumerate(para_delex_list):
        belief = belief_list[num]
        replace = replace_list[num]
        for slu in belief:
            if slu["act"] == "inform":
                info = slu["slots"][0][1]
                typ = slu["slots"][0][0]
                if info != "dontcare":
                    para = re.sub("<" + str(typ) + ">", info, para)
        for name in replace:
            para = re.sub(str("<restaurant>"), name, para, 1)
        para = re.sub(str("<restaurant>"), "the restaurant", para)
        para_list.append(para)
        success = True
        for token in para.split(" "):
            if token:
                if token[0] == "<":
                    success = False
        success_list.append(success)
    return para_list, success_list


def realization_multiwoz(para_delex_list, context_list):
    para_list = []
    success_list = []
    for num, para in enumerate(para_delex_list):
        token_list = para.split(" ")
        slu_dic = context_list[num]
        domains = []
        for domain in list(slu_dic.keys()):
            domains.append(domain)
        domains.reverse()
        for domain in domains:
            slots = slu_dic[domain]
            for slot in slots:
                for n, token in enumerate(token_list):
                    if token == "[value_" + slot[0] + "]":
                        token_list[n] = slot[1]
                        break
            para = " ".join(token_list)
        para_list.append(para)

        success = True
        for token in para.split(" "):
            if token:
                if token[0] == "[":
                    success = False
        success_list.append(success)

    return para_list, success_list


def slots_match(ref_list, pre_list):
    slots_match_success = []
    for ref, pre in zip(ref_list, pre_list):
        ref_token = ref.split(" ")
        pre_token = pre.split(" ")
        slot_token_ref = []
        slot_token_pre = []
        for r_token in ref_token:
            if r_token:
                if r_token[0] == "<":
                    r_token = r_token.split(">")[0] + ">"
                    if r_token != "<restaurant>":
                        slot_token_ref.append(r_token)
        for p_token in pre_token:
            if p_token:
                if p_token[0] == "<":
                    p_token = p_token.split(">")[0] + ">"
                    if p_token != "<restaurant>":
                        slot_token_pre.append(p_token)
        slots_match_success.append(True)
        for ref_slot in slot_token_ref:
            if ref_slot not in pre:
                slots_match_success[-1] = False
    return slots_match_success


def slots_match_multiwoz(ref_list, pre_list):
    slots_match_success = []
    for ref, pre in zip(ref_list, pre_list):
        ref_token = ref.split(" ")
        pre_token = pre.split(" ")
        slot_token_ref = []
        slot_token_pre = []
        for r_token in ref_token:
            if r_token:
                if r_token[0] == "[":
                    r_token = r_token.split("]")[0] + "]"
                    slot_token_ref.append(r_token)
        for p_token in pre_token:
            if p_token:
                if p_token[0] == "[":
                    p_token = p_token.split("]")[0] + "]"
                    slot_token_pre.append(p_token)
        slots_match_success.append(True)
        for ref_slot in slot_token_ref:
            if ref_slot not in pre:
                slots_match_success[-1] = False
    return slots_match_success
