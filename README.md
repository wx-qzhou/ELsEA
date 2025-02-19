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

--subtask_size
[
gcn-align:
['fr']=9594 ['ja']=10323 ['zh']=9421 ['wd']=28693 ['yg']=29521
['1m']=30874
rrea:
['fr']=9594 ['ja']=10323 ['zh']=9421 ['wd']=28693 ['yg']=29521
['1m']=28874
dualamn:
['fr']=9594 ['ja']=10323 ['zh']=9421 ['wd']=28693 ['yg']=29521
['1m']=28874
]

--kgids 
en,de en,fr
python Large_DivEA_run_1m.py --kgids  --ea_model rrea --eval_way csls
python Large_DivEA_run_1m.py --kgids en,de --ea_model rrea --eval_way csls --ctx_g1_percent 0.4 --subtask_size 28874
python Large_DivEA_run_1m.py --kgids en,fr --ea_model rrea --eval_way csls --ctx_g1_percent 0.4 --subtask_size 28874 --max_iteration 1

--kgids 
dbp,yg
python Medium_DivEA_run_dwy100k.py --kgids dbp,yg --ea_model rrea --eval_way csls --subtask_size 29521

--kgids 
ja,en fr,en zh,en
python Small_DivEA_run_dbp15k.py --kgids  --ea_model rrea --eval_way csls --subtask_size 

cd ..
python Get_results.py --data_name dbp15k --kgids fr,en

# unsupervised
[
    dbp15k:
    ja,en fr,en zh,en
    --threshold 250000
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
