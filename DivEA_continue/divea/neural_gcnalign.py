# -*- coding: utf-8 -*-
import os
import json
import subprocess
import numpy as np
from RREA.CSLS_torch import Evaluator

from divea.util import Config
from divea.dataload import read_alignment
from divea.dataload import convert_uniform_to_rrea
from divea.components_base import NeuralEAModule


class GCNAlignModule(NeuralEAModule):
    def __init__(self, conf: Config):
        super(GCNAlignModule, self).__init__(conf)

    def refresh_weights(self):
        pass

    def prepare_data(self):
        convert_uniform_to_rrea(self.conf.data_dir, self.conf.kgids)

    def train_model(self):
        cmd_fn = self.conf.py_exe_fn
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        script_fn = os.path.join(cur_dir, "../GCN_Align/run.py")
        args_str = f"--data_dir={self.conf.data_dir} --output_dir={self.conf.output_dir} " \
                   f"--max_train_epoch={self.conf.max_train_epoch} " \
                   f"--tf_gpu_no={self.conf.tf_gpu_id}"
        env = os.environ.copy()
        print(args_str)
        env["CUDA_VISIBLE_DEVICES"] = f"{self.conf.tf_gpu_id}"
        ret = subprocess.run(cmd_fn + " " + script_fn + " " + args_str, shell=True, env=env)
        if ret.returncode != 0:
            raise Exception("GCN-Align did not run successfully.")

    def predict_simi(self):
        emb_res = np.load(os.path.join(self.conf.output_dir, "emb.npz"))
        embs = emb_res["embs"]
        ent1_ids = emb_res["ent1_ids"]
        ent2_ids = emb_res["ent2_ids"]
        ent1_embs = embs[ent1_ids]
        ent2_embs = embs[ent2_ids]

        evaluator = Evaluator(device=self.conf.torch_device)
        simi_mtx = evaluator.csls_sim(ent1_embs, ent2_embs, k=10)

        return simi_mtx

    def get_embeddings(self):
        emb_res = np.load(os.path.join(self.conf.output_dir, "emb.npz"))
        embs = emb_res["embs"]
        ent1_ids = emb_res["ent1_ids"]
        ent2_ids = emb_res["ent2_ids"]
        ent1_embs = embs[ent1_ids]
        ent2_embs = embs[ent2_ids]
        return ent1_embs, ent2_embs

    def get_pred_alignment(self, eval_way="sinkhorn"):
        # emea_data = EMEAData(self.conf.data_dir, self.conf.data_name)
        with open(os.path.join(self.conf.output_dir, "pred_alignment.json")) as file:
            obj = json.loads(file.read())
            if eval_way == "sinkhorn":
                pred_alignment = obj["pred_alignment_sinkhorn"]
            elif eval_way == "csls":
                pred_alignment = obj["pred_alignment_csls"]
            elif eval_way == "cosine":
                pred_alignment = obj["pred_alignment_cos"]
        # kg1_old2new_ent_map, kg2_old2new_ent_map = emea_data.old2new_entity_id_map()
        # pred_alignment = [(kg1_old2new_ent_map[ent1], kg2_old2new_ent_map[ent2]) for ent1, ent2 in pred_alignment]
        return pred_alignment

    def get_target_dandidates(self):
        kg1_ent_id_uri = read_alignment(os.path.join(self.conf.data_dir, f"{self.conf.kgids[0]}_entity_id2uri.txt"))
        kg2_ent_id_uri = read_alignment(os.path.join(self.conf.data_dir, f"{self.conf.kgids[1]}_entity_id2uri.txt"))
        kg1_newid2oldid = dict(kg1_ent_id_uri)
        all_alignment = read_alignment(os.path.join(self.conf.data_dir, "train_alignment.txt"))

        ori_alignment = read_alignment(os.path.join(self.conf.data_dir, "../train_alignment.txt"))
        ori_mapping_map = dict(ori_alignment)
        filtered_entities1 = []
        filtered_entities2 = []
        for e1, e2 in all_alignment:
            if kg1_newid2oldid[e1] in ori_mapping_map:
                filtered_entities1.append(e1)
                filtered_entities2.append(e2)
        # filtered_entities2 = [e2 for e1, e2 in all_alignment if kg1_newid2oldid[e1] in ori_mapping_map]
        kg1_entities = [e for e, uri in kg1_ent_id_uri]
        kg1_candidates = sorted(list(set(kg1_entities).difference(set(filtered_entities2))))
        kg2_entities = [e for e, uri in kg2_ent_id_uri]
        kg2_candidates = sorted(list(set(kg2_entities).difference(set(filtered_entities2))))
        return kg1_candidates, kg2_candidates

    def evaluate(self, eval_way="sinkhorn"):
        emb_res = np.load(os.path.join(self.conf.output_dir, "emb.npz"))
        embs = emb_res["embs"]
        ent1_ids = emb_res["ent1_ids"]
        ent2_ids = emb_res["ent2_ids"]
        ent1_embs = embs[ent1_ids]
        ent2_embs = embs[ent2_ids]

        eval_alignment = read_alignment(os.path.join(self.conf.data_dir, "test_alignment.txt"))
        inv_eval_alignment = [(e2, e1) for e1, e2 in eval_alignment]

        evaluator = Evaluator(device=self.conf.torch_device)

        sinkhorn_test_metrics, sinkhorn_test_alignment = [], []
        csls_test_metrics, csls_test_alignment = [], []
        cosine_test_metrics, cosine_test_alignment = [], []
        inv_sinkhorn_test_metrics, inv_sinkhorn_test_alignment = [], []
        inv_csls_test_metrics, inv_csls_test_alignment = [], []
        inv_cosine_test_metrics, inv_cosine_test_alignment = [], []

        if eval_way == "sinkhorn":
            sinkhorn_test_metrics, sinkhorn_test_alignment = evaluator.evaluate_sinkhorn(ent1_embs, ent2_embs, eval_alignment)
            test_alignment = sinkhorn_test_alignment
            sinkhorn_test_alignment = sinkhorn_test_alignment.tolist()

            inv_sinkhorn_test_metrics, inv_sinkhorn_test_alignment = evaluator.evaluate_sinkhorn(ent2_embs, ent1_embs, inv_eval_alignment)
            inv_test_alignment = inv_sinkhorn_test_alignment
            inv_sinkhorn_test_alignment = inv_sinkhorn_test_alignment.tolist()
        elif eval_way == "csls":
            csls_test_metrics, csls_test_alignment = evaluator.evaluate_csls(ent1_embs, ent2_embs, eval_alignment)
            test_alignment = csls_test_alignment
            csls_test_alignment = csls_test_alignment.tolist()

            inv_csls_test_metrics, inv_csls_test_alignment = evaluator.evaluate_csls(ent2_embs, ent1_embs, inv_eval_alignment)
            inv_test_alignment = inv_csls_test_alignment
            inv_csls_test_alignment = inv_csls_test_alignment.tolist()
        elif eval_way == "cosine":
            cosine_test_metrics, cosine_test_alignment = evaluator.evaluate_cosine(ent1_embs, ent2_embs, eval_alignment)
            test_alignment = cosine_test_alignment
            cosine_test_alignment = cosine_test_alignment.tolist()

            inv_cosine_test_metrics, inv_cosine_test_alignment = evaluator.evaluate_cosine(ent2_embs, ent1_embs, inv_eval_alignment)
            inv_test_alignment = inv_cosine_test_alignment
            inv_cosine_test_alignment = inv_cosine_test_alignment.tolist()
        else:
            raise Exception("Not implement.")
                
        pseudo_pairs = []
        inv_pred_map = dict(inv_test_alignment.tolist())
        for e1, e2 in test_alignment.tolist():  # e1: 0,1,2.. ; e2: 0,1,2...
            if e1 == inv_pred_map.get(e2, None):
                pseudo_pairs.append((e1, e2))
        lines = [f"{e1}\t{e2}" for e1, e2 in pseudo_pairs]
        new_cont = "\n".join(lines)
        with open(os.path.join(self.conf.data_dir, "new_pseudo_seeds_raw.txt"), "w+") as file:
            file.write(new_cont)

        all_alignment = evaluator.predict_alignment(ent1_embs, ent2_embs, eval_way=eval_way)

        metrics_obj = {"metrics_sinkhorn": sinkhorn_test_metrics, "metrics_csls": csls_test_metrics, "metrics_cos": cosine_test_metrics,\
                       "inv_metrics_sinkhorn": inv_sinkhorn_test_metrics, "inv_metrics_csls": inv_csls_test_metrics, "inv_metrics_cos": inv_cosine_test_metrics}
        pred_alignment_obj = {"pred_alignment_sinkhorn": sinkhorn_test_alignment,
                              "pred_alignment_csls": csls_test_alignment,
                              "pred_alignment_cos": cosine_test_alignment,
                              "all_pred_alignment": all_alignment.tolist()
                              }
        
        with open(os.path.join(self.conf.output_dir, "metrics.json"), "w+") as file:
            file.write(json.dumps(metrics_obj))
        with open(os.path.join(self.conf.output_dir, "pred_alignment.json"), "w+") as file:
            file.write(json.dumps(pred_alignment_obj))
        with open(os.path.join(self.conf.output_dir, "eval_metrics.json"), "w+") as file:
            if eval_way == "sinkhorn":
                file.write(json.dumps(sinkhorn_test_metrics))
            elif eval_way == "csls":    
                file.write(json.dumps(csls_test_metrics))
            elif eval_way == "cosine":
                file.write(json.dumps(cosine_test_metrics))
        return metrics_obj

    def evaluate_given_alignment(self, eval_alignment):
        emb_res = np.load(os.path.join(self.conf.output_dir, "emb.npz"))
        embs = emb_res["embs"]
        ent1_ids = emb_res["ent1_ids"]
        ent2_ids = emb_res["ent2_ids"]
        ent1_embs = embs[ent1_ids]
        ent2_embs = embs[ent2_ids]

        evaluator = Evaluator(device=self.conf.torch_device)
        sinkhorn_test_metrics, sinkhorn_test_alignment = evaluator.evaluate_sinkhorn(ent1_embs, ent2_embs, eval_alignment)
        csls_test_metrics, csls_test_alignment = evaluator.evaluate_csls(ent1_embs, ent2_embs, eval_alignment)
        cos_test_metrics, cos_test_alignment = evaluator.evaluate_cosine(ent1_embs, ent2_embs, eval_alignment)

        metrics_obj = {"metrics_sinkhorn": sinkhorn_test_metrics, "metrics_csls": csls_test_metrics, "metrics_cos": cos_test_metrics}
        print(metrics_obj)
        return sinkhorn_test_metrics, csls_test_metrics, cos_test_metrics
