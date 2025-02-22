# -*- coding: utf-8 -*-
"""prepare data for rrea"""
import os
import argparse
from divea.util import seed_everything
from divea.graph_metrics import graph_metrics_main
from divea.dataload import UniformData, read_tab_lines, convert_uniform_to_rrea, convert_uniform_to_openea

from Unsuper.TranslatetoEN.translate_data import dump_json, Helsinki_NLP, TraditionaltoSimplified


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default="2m/", required=False, help="input dataset name")
    parser.add_argument('--data_dir', type=str, default="../data/", required=False, help="input dataset file directory")
    parser.add_argument('--kgids', type=str, default="fb,dbp", help="separate two ids with comma. e.g. `fr,en`")
    parser.add_argument('--unsuper', action='store_true', default=True)
    parser.add_argument('--surface', action='store_true', default=True)
    parser.add_argument('--ugraph', action='store_true', default=True)
    parser.add_argument('--threshold', default=None)
    parser.add_argument('--index_batch_sz', type=int, default=500000)
    parser.add_argument('--thresholdstr', type=float, default=0.85)
    parser.add_argument('--train_percent', type=float, required=False, default=0.2)
    parser.add_argument('--rrea', action='store_true', default=False)
    parser.add_argument('--openea', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1011)
    args = parser.parse_args()

    return args

# generate the text of each entity
def generate_txt(args, kgids, kgid, model_name=None, flag=True):
    ent_id2uri = read_tab_lines(os.path.join(args.data_dir, args.data_name, "_".join(kgids), kgid + "_entity_id2uri.txt"))
    if kgid == "zh":
        ent_id2uri = TraditionaltoSimplified(ent_id2uri)
    if model_name == None:
        ent_id2uri = dict(ent_id2uri)
    else:
        ent_id2uri = Helsinki_NLP(ent_id2uri, model_name, eeflag=flag)
    dump_json(ent_id2uri, os.path.join(args.data_dir, args.data_name, "_".join(kgids), kgid + "_entity_txt.json"))

if __name__ == "__main__":
    args = get_parser()
    print(vars(args))

    seed_everything(args.seed)

    kgids = args.kgids.split(",")

    model_names = {"zh,en" : ["opus-mt-zh-en", None], "ja,en" : ["opus-mt-ja-en", None], "en,fr" : [None, "opus-mt-fr-en"],\
                   "fr,en" : ["opus-mt-fr-en", None], "en,de" : [None, "opus-mt-de-en"]}
    thresholdstrs = {"zh,en" : 0.85, "ja,en" : 0.85, "en,fr" : 0.85, "fr,en" : 0.85, "en,de" : 0.8}
    flags = {"zh,en" : True, "ja,en" : True, "en,fr" : True, "fr,en" : True, "en,de" : True}

    uni_data = UniformData(os.path.join(args.data_dir, args.data_name, "_".join(kgids)), kgids)
    # graph_metrics_main(uni_data)

    if args.unsuper:
        print("This is unsupervised modal, and the dataset is ", args.data_name, "_".join(kgids))
        args.train_percent = 0.0

        if args.surface:
            print("Here, using surface.")
            if os.path.exists(os.path.join(args.data_dir, args.data_name, "_".join(kgids), kgids[0] + "_entity_txt.json")) == False:
                model_name_dict = {}
                if args.data_name.strip() in ["1m", "dbp15k", "dwy100k"]:
                    if kgids[0] == "dbp" or kgids[1] == 'de':
                        model_name_dict.update({kgids[0] : None}) 
                        model_name_dict.update({kgids[1] : None})  
                        flag = False
                    else:
                        model_name_dict.update({kgids[0] : model_names[args.kgids][0]}) 
                        model_name_dict.update({kgids[1] : model_names[args.kgids][1]}) 
                        flag = flags[args.kgids]
                        args.thresholdstr = thresholdstrs[args.kgids]
                    for kgid in model_name_dict:
                        generate_txt(args, kgids, kgid, model_name_dict[kgid], flag=flag)
            else:
                print("The surface has done.")
        else:
            print("No surface.")

    # exit()
    print("Dividing the train and test sets.")
    if args.threshold != None:
        args.threshold = int(args.threshold)
        if args.data_name.strip() in ["1m"]:
            if kgids[1] == 'fr':
                args.index_batch_sz = 12000
            else:
                args.index_batch_sz = 10000
    # this is to divide the train and test sets
    uni_data.divide_train_test(args.train_percent, args.threshold, surface=args.surface, ugraph=args.ugraph, thresholdstr=args.thresholdstr, index_batch_sz=args.index_batch_sz)


    # if args.rrea:
    #     convert_uniform_to_rrea(args.data_dir, kgids)
    # if args.openea:
    #     convert_uniform_to_openea(args.data_dir, kgids)

    # python run_prepare_data.py --data_name IDS15K_V1 --kgids en,de