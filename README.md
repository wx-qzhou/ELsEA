# Running
conda activate qzhou_LargerEA_py310

--gres=
gpu:TeslaV100-SXM2-32GB:1 
gpu:TeslaV100S-PCIE-32GB:1
gpu:NVIDIAGeForceRTX2080Ti:1 
gpu:NVIDIAA100-PCIE-40GB:1 

srun -N 1 -t 4320 -n 10 -c 1 --gres=gpu:TeslaV100-SXM2-32GB:2 --mem=600G -p batch --pty /bin/bash
srun -N 1 -t 4320 -n 6 -c 1 --gres=gpu:TeslaV100S-PCIE-32GB:1 --mem=250G -p batch --pty /bin/bash
srun -N 1 -t 4320 -n 6 -c 1 --gres=gpu:TeslaV100-SXM2-32GB:1 --mem=250G -p batch --pty /bin/bash

cd ../DivEA
bash run_prepare_data.sh

--ea_model 
rrea dualamn gcn-align 

--eval_way 
csls cosine sinkhorn

--kgids 
en,de en,fr
python Large_DivEA_run_1m.py --kgids  --ea_model rrea --eval_way csls

--kgids 
dbp,yg
python Medium_DivEA_run_dwy100k.py --kgids dbp,yg --ea_model rrea --eval_way csls

--kgids 
ja,en fr,en zh,en
python Small_DivEA_run_dbp15k.py --kgids  --ea_model rrea --eval_way csls --subtask_size 

cd ..
python Get_results.py --data_name dbp15k --kgids fr,en

# Unsupervised
[
    dbp15k:
    ja,en fr,en zh,en
    dwy100k:
    dbp,yg
    1m: 
    en,de en,fr
]

python run_prepare_data.py --data_name dbp15k --kgids fr,en
python run_prepare_data.py --data_name dwy100k --kgids dbp,yg
python run_prepare_data.py --data_name 1m --kgids en,de --threshold 40000
python run_prepare_data.py --data_name 1m --kgids en,fr --threshold 40000

python calculate_acc.py --data_name dbp15k --kgids fr,en
python calculate_acc.py --data_name dwy100k --kgids dbp,yg
python calculate_acc.py --data_name 1m --kgids en,de

# Acknowledgement
We used the source codes:
1) RREA: https://github.com/MaoXinn/RREA;
2) GCN-Align: https://github.com/1049451037/GCN-Align;
3) Dual-AMN: https://github.com/MaoXinn/Dual-AMN;
4) DivEA: https://github.com/uqbingliu/DivEA.
