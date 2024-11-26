import os
import json

def dump_data(file_name, data):
    # 将数据写入 JSON 文件
    with open(file_name, 'w') as file:
        # 使用 json.dump() 方法将数据写入文件
        json.dump(data, file, indent=4)  # indent 参数用于指定缩进空格数，使输出更易读

def load_data(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data

def write_txt(file_name, text):
    with open(file_name, 'w') as f:
        for item in text:
            f.write("%s\n" % item)

def deal_data(data):
    result = ""
    for metric in data["effectiveness"]:
        result += str(data["effectiveness"][metric]) + " "
    result += str(data["running_time"]["total"]) + " "
    result += str((data["ea_gpu_memory"] + data["ctx1_gpu_memory"] + data["ctx2_gpu_memory"]) / 1024) + "G"
    return result

if __name__ == "__main__":
    dir_name = "./results/DivEA"

    model_list = ["dualamn", "gcn-align", "lightea", "rrea"]
    eval_list = ["cosine", "csls", "sinkhorn"]
    datasets_list = [["IDS15K_V1", "IDS15K_V2", "IDS100_V1", "IDS100_V2", "1m"], ["dbp15k"], ["dwy100k"], ["2m"]]
    datanames_list = [["en_de", "en_fr"], ["fr_en", "ja_en", "zh_en"], ["dbp_wd", "dbp_yg"], ["fb_dbp"]]

    no_list = []
    results_list = []

    for model in model_list:
        for eval in eval_list:
            for idx in range(len(datasets_list)):
                datasets = datasets_list[idx]
                datanames = datanames_list[idx]
                for dataset in datasets:
                    for dataname in datanames:
                        file_name = os.path.join(dir_name, model, eval, dataset, dataname, "metrics.json")
                        if os.path.exists(file_name):
                            data = load_data(file_name)
                            result = deal_data(data)
                            results_list.append(" ".join([dataset, dataname, model, eval]) + " " + result)
                        else:
                            no_list.append([dataset, dataname, model, eval])
    
    write_txt("./results/results.txt", results_list)
    dump_data("./results/no_list.json", no_list)