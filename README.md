# ELsEA
Implementation of the study proposed in the paper <a href="https://ieeexplore.ieee.org/">Enhancing Large-scale Entity Alignment with Critical Structure and High-quality Context</a>

# Environment Setup and Data Preparation

First, set up the specified Conda environment and install the required dependencies from the 'Install_requirements' file.

Then, download the pre-trained models from the [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) repository on Hugging Face into the directory ./DivEA/Unsuper/TranslatetoEN.

Next, activate the specified Conda environment, for example:
```bash
conda activate qzhou_LargerEA_py310
```

Finally, go to the `DivEA` directory and run the data preparation script:
```bash
cd ../DivEA
bash run_prepare_data.sh
```

# Parameters Explanation

## `--ea_model` EA Approach Options

Available EA approaches:

- rrea
- dualamn
- gcn-align

## `--subtask_size` Subtask Sizes

Set the subtask size based on different models and language pairs:

- **gcn-align**:
  - `fr`: 9594
  - `ja`: 10323
  - `zh`: 9421
  - `wd`: 28693
  - `yg`: 29521
  - `1m`: 30874
- **rrea**:
  - `fr`: 9594
  - `ja`: 10323
  - `zh`: 9421
  - `wd`: 28693
  - `yg`: 29521
  - `1m`: 28874
- **dualamn**:
  - `fr`: 9594
  - `ja`: 10323
  - `zh`: 9421
  - `wd`: 28693
  - `yg`: 29521
  - `1m`: 28874

## `--kgids` Knowledge Graph IDs

Different knowledge graph ID configurations, such as:

- `en,de`
- `en,fr`

## Python Commands

Run different Python scripts for specific tasks:

1. **rrea model**, using the `csls` evaluation method:

```bash
python Large_DivEA_run_1m.py --kgids --ea_model rrea --eval_way csls
```

2. **rrea model**, using the `csls` evaluation method, specifying subtask size and `ctx_g1_percent`:

```bash
python Large_DivEA_run_1m.py --kgids en,de --ea_model rrea --eval_way csls --ctx_g1_percent 0.4 --subtask_size 28874
```

3. **rrea model**, using the `csls` evaluation method, specifying `max_iteration`:

```bash
python Large_DivEA_run_1m.py --kgids en,fr --ea_model rrea --eval_way csls --ctx_g1_percent 0.4 --subtask_size 28874 --max_iteration 5
```

4. **rrea model**, for `dbp,yg` knowledge graph pair, using the `csls` evaluation method:

```bash
python Medium_DivEA_run_dwy100k.py --kgids dbp,yg --ea_model rrea --eval_way csls --subtask_size 29521
```

5. **rrea model**, for `ja,en`, `fr,en`, and `zh,en` knowledge graph pairs:

```bash
python Small_DivEA_run_dbp15k.py --kgids --ea_model rrea --eval_way csls --subtask_size
```



# Unsupervised Tasks

## Data Preparation

Run the data preparation script for different datasets under seed-free :

1. **dbp15k** dataset, evaluating `fr,en` knowledge graph:

```bash
python run_prepare_data.py --data_name dbp15k --kgids fr,en
```

2. **dwy100k** dataset, evaluating `dbp,yg` knowledge graph:

```bash
python run_prepare_data.py --data_name dwy100k --kgids dbp,yg
```

3. **1m** dataset, evaluating `en,de` knowledge graph, setting threshold:

```bash
python run_prepare_data.py --data_name 1m --kgids en,de --threshold 40000
```

4. **1m** dataset, evaluating `en,fr` knowledge graph, setting threshold:

```bash
python run_prepare_data.py --data_name 1m --kgids en,fr --threshold 40000
```

## Accuracy Calculation

Use different datasets and knowledge graph pairs to calculate accuracy:

1. **dbp15k** dataset, evaluating `fr,en` knowledge graph:

```bash
python calculate_acc.py --data_name dbp15k --kgids fr,en
```

2. **dwy100k** dataset, evaluating `dbp,yg` knowledge graph:

```bash
python calculate_acc.py --data_name dwy100k --kgids dbp,yg
```

3. **1m** dataset, evaluating `en,de` knowledge graph:

```bash
python calculate_acc.py --data_name 1m --kgids en,de
```

## Citing
Citing:
1) https://github.com/uqbingliu/DivEA.  
2) https://github.com/MaoXinn/RREA.
3) https://github.com/1049451037/GCN-Align.
4) https://github.com/ZJU-DAILY/DualMatch.
