
data_name=("1m")
kgids=("en,de" "en,fr")

for data_name in "${data_name[@]}"; do
    for kgids in "${kgids[@]}"; do
        python run_prepare_data.py --data_name "$data_name" --kgids "$kgids" --threshold 250000
    done
done

data_name=("2m")
kgids=("fb,dbp")

for data_name in "${data_name[@]}"; do
    for kgids in "${kgids[@]}"; do
        python run_prepare_data.py --data_name "$data_name" --kgids "$kgids" --threshold 300000
    done
done