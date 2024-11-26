# -*- coding: utf-8 -*-
import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default="2m/", required=False, help="input dataset name")
    parser.add_argument('--data_dir', type=str, default="../data/", required=False, help="input dataset file directory")
    parser.add_argument('--kgids', type=str, default="fb,dbp", help="separate two ids with comma. e.g. `fr,en`")
    parser.add_argument('--divide', action='store_true', default=True)
    parser.add_argument('--unsuper', action='store_true', default=True)
    parser.add_argument('--threshold', default=None)
    parser.add_argument('--train_percent', type=float, required=False, default=0.2)
    parser.add_argument('--rrea', action='store_true', default=False)
    parser.add_argument('--openea', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1011)
    args = parser.parse_args()

    return args

def read_tab_lines(fn):
    with open(fn) as file:
        cont = file.read().strip()
        if cont == "":
            return []
        lines = cont.split("\n")
        tuple_list = []
        for line in lines:
            t = line.split("\t")
            tuple_list.append(t)
    return tuple_list

def write_tab_lines(tuple_list, fn):
    with open(fn, "w+") as file:
        for tup in tuple_list:
            s_tup = [str(e) for e in tup]
            file.write("\t".join(s_tup) + "\n")

def read_alignment(fn):
    alignment = read_tab_lines(fn)
    alignment = [(int(ent1_id), int(ent2_id)) for ent1_id, ent2_id in alignment]
    return alignment

if __name__ == "__main__":
    args = get_parser()
    print(vars(args))
    kgids = args.kgids.split(",")
    data_dir = os.path.join(args.data_dir, args.data_name, "_".join(kgids))
    
    # 定义两个列表
    train_align = read_alignment(os.path.join(data_dir, "train_alignment.txt"))
    all_align = read_alignment(os.path.join(data_dir, "alignment_of_entity.txt"))

    # 将列表转换为集合
    unsper_align_set = set(train_align)
    align_set = set(all_align)

    # 计算交集
    common_elements = align_set & unsper_align_set

    # 打印共有的元素数量
    print("共有的元素数量:", len(common_elements))

    print("Train Number:", len(unsper_align_set))
    print("Alignment Number:", len(align_set))

    # 打印共有的元素
    print("Train 占:", len(common_elements) / len(unsper_align_set))
    print("Alignment 占:", len(common_elements) / len(align_set))