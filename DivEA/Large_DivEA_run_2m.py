# -*- coding: utf-8 -*-
import os
import torch
import shutil
import argparse
from os import makedirs
import multiprocessing

from divea.dataload import UniformData
from divea.neural_rrea import RREAModule
from divea.util import Config, seed_everything
from divea.framework import ParallelEAFramework
from divea.components_base import Client, Server
from divea.neural_gcnalign import GCNAlignModule
from divea.neural_dualamn import Dual_AMNModule
from divea.neural_lightea import LightEAModule
from divea.ctx_builder_onclient import CtxBuilderV1, CtxBuilderV2
from divea.components import DivG1Metis, DivG1Random, CounterpartDiscovery 

from divea.graph_metrics import graph_metrics_main

def make_dirs(path):
    if not os.path.exists(path):
        makedirs(path)
    else:
        shutil.rmtree(path)
        makedirs(path)

def get_parser():
    parser = argparse.ArgumentParser()

    ## data, output
    parser.add_argument('--kgids', type=str, default="fb,dbp", help="separate two ids with comma. e.g. `fr,en`")
    parser.add_argument('--data_name', type=str, default="2m", required=False, help="input dataset name")
    parser.add_argument('--data_dir', type=str, default="../data/", required=False, help="input dataset file directory")
    parser.add_argument('--output_dir', type=str,  default="../results/DivEA/", required=False, )
    # subtask
    parser.add_argument('--subtask_num', type=int, default=200, required=False)
    parser.add_argument('--subtask_size', type=str, default="1.0", help="specify size with int value; or specify ratio to average partition size with float value")  #
    parser.add_argument('--ctx_g1_percent', type=float, default=0.5, help="percentage of first context graph size")
    parser.add_argument('--ctx_g2_conn_percent', type=float, default=0.0, help="percentage of connecting entities in the second context graph")
    # module configuration
    parser.add_argument('--ctx_builder', type=str, default="v1", choices=["v1", "v2"], help="v1: build ctx on client; v2: build ctx on server.")
    parser.add_argument('--div_g1', type=str, default="metis", choices=["metis", "random"], help="method of partitioning first graph; for detailed analysis")  # metis, random
    parser.add_argument('--counterdisc_ablation', type=str, default="full", choices=["full", "locality"], help="for ablation study of counterpart discovery")
    # hyper-parameters, see paper for the meanining of these hyper-parameters
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--topK', type=int, default=10)
    parser.add_argument('--max_iteration', type=int, default=5)
    # ea model
    parser.add_argument('--ea_model', type=str, default="rrea", choices=["rrea", "dualamn", "gcn-align", "lightea"], help="EA model")
    parser.add_argument('--eval_way', type=str, default="sinkhorn", choices=["csls", "cosine", "sinkhorn"], help="EA model")
    parser.add_argument('--layer_num', type=int, default=2, help="number of GCN layers")
    parser.add_argument('--max_train_epoch', type=int, default=50, help="max epoch of training EA model")
    # device, running env
    parser.add_argument('--gpu_ids', type=str, default="0,1,2,3", help="visible GPU devices")
    parser.add_argument('--py_exe_fn', type=str)
    parser.add_argument('--is_mulprocess', type=bool, default=True, help="is to use mulprocess")
    parser.add_argument('--proc_n', type=int, default=20, help="the number of mulprocess")
    # others
    parser.add_argument('--seed', type=int, default=1011, help="random seed")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_parser()
    kgids = args.kgids.split(",")
    data_dir = os.path.join(args.data_dir, args.data_name, "_".join(kgids))
    out_dir = os.path.join(args.output_dir, args.ea_model, args.eval_way, args.data_name, "_".join(kgids))
    make_dirs(out_dir)
    
    seed_everything(args.seed)
    num_threads = multiprocessing.cpu_count()
    print("The thread number of this system is {}.".format(num_threads))
    args.proc_n = min(num_threads, args.proc_n)

    args.gpu_ids = ",".join(args.gpu_ids.split(",")[:torch.cuda.device_count()])
    
    data = UniformData(data_dir, kgids)
    graph_metrics_main(data)
    data.load_weights()

    subtask_size = eval(args.subtask_size)
    if isinstance(subtask_size, float):
        total_kg_size = len(data.kg1_entities) + len(data.kg2_entities)
        subtask_size = int(subtask_size * total_kg_size / args.subtask_num)

    ctx_g1_size = int(subtask_size * args.ctx_g1_percent)
    ctx_g2_size = subtask_size - ctx_g1_size

    if args.ctx_builder == "v1":
        ctx_builder = CtxBuilderV1(data, data_dir, kgids, out_dir, args.subtask_num, args.layer_num, gamma=args.gamma, \
                                   ctx_g1_size=ctx_g1_size, ctx_g2_size=ctx_g2_size, ctx_g2_conn_percent=args.ctx_g2_conn_percent, \
                                        torch_devices=[f"cuda:{didx}" for didx in range(len(args.gpu_ids.split(",")))], is_mulprocess=args.is_mulprocess, \
                                            proc_n=args.proc_n)
    else:
        ctx_builder = CtxBuilderV2(data, data_dir, kgids, out_dir, args.subtask_num, args.layer_num, gamma=args.gamma, \
                                   ctx_g1_size=ctx_g1_size, ctx_g2_size=ctx_g2_size, ctx_g2_conn_percent=args.ctx_g2_conn_percent, \
                                    torch_devices=[f"cuda:{didx}" for didx in range(len(args.gpu_ids.split(",")))], is_mulprocess=args.is_mulprocess, \
                                        proc_n=args.proc_n)
        
    if args.div_g1 == "metis":
        g1_partitioner = DivG1Metis(data, data_dir, kgids, args.subtask_num, balance=False)
    elif args.div_g1 == "random":
        g1_partitioner = DivG1Random(data, data_dir, kgids, args.subtask_num, balance=False)
    else:
        raise Exception("unknown g1 partition method")


    count_discover = CounterpartDiscovery(data, data_dir, kgids, args.subtask_num, ctx_g2_size=ctx_g2_size,
                                        max_hop_k=2*args.layer_num, out_dir=out_dir, alpha=args.alpha,
                                        beta=args.beta, topK=args.topK, ablation=args.counterdisc_ablation
                                        )

    server = Server(data_dir, kgids, out_dir, args.subtask_num, g1_partitioner, count_discover, ctx_builder)
    clients = []

    for part_idx in range(1, args.subtask_num+1):
        conf = Config()
        part_data_dir = os.path.join(data_dir, f"partition_{part_idx}")
        part_out_dir = os.path.join(out_dir, f"partition_{part_idx}")
        conf.data_dir = part_data_dir
        conf.output_dir = part_out_dir
        conf.kgids = kgids
        conf.max_train_epoch = args.max_train_epoch
        conf.gcn_layer_num = args.layer_num
        conf.py_exe_fn = args.py_exe_fn
        gpu_ids_list = args.gpu_ids.split(",")
        conf.gpu_ids = gpu_ids_list[part_idx % len(gpu_ids_list)]
        conf.tf_gpu_id = int(gpu_ids_list[part_idx % len(gpu_ids_list)])
        conf.torch_device = torch.device("cuda:" + gpu_ids_list[part_idx % len(gpu_ids_list)])

        if args.ea_model == "rrea":
            ea_module = RREAModule(conf)
        elif args.ea_model == "dualamn":
            conf.py_exe_fn = "python"
            ea_module = Dual_AMNModule(conf)
        elif args.ea_model == "lightea":
            conf.py_exe_fn = "python"
            ea_module = LightEAModule(conf)
        else:
            conf.py_exe_fn = "python"
            ea_module = GCNAlignModule(conf)

        client = Client(part_data_dir, kgids, part_out_dir, ea_module)
        clients.append(client)
    
    framework = ParallelEAFramework(server, clients, eval_way=args.eval_way, max_iteration=args.max_iteration)
    framework.run()
    print("Having done!")