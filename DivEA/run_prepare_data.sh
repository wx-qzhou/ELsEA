
data_name=("IDS15K_V1" "IDS15K_V2" "IDS100_V1" "IDS100_V2" "1m")
kgids=("en,de" "en,fr")

for data_name in "${data_name[@]}"; do
    for kgids in "${kgids[@]}"; do
        python run_prepare_data.py --data_name "$data_name" --kgids "$kgids"
    done
done

data_name=("dbp15k")
kgids=("fr,en" "ja,en" "zh,en")

for data_name in "${data_name[@]}"; do
    for kgids in "${kgids[@]}"; do
        python run_prepare_data.py --data_name "$data_name" --kgids "$kgids"
    done
done

data_name=("dwy100k")
kgids=("dbp,wd" "dbp,yg")

for data_name in "${data_name[@]}"; do
    for kgids in "${kgids[@]}"; do
        python run_prepare_data.py --data_name "$data_name" --kgids "$kgids"
    done
done


data_name=("2m")
kgids=("fb,dbp")

for data_name in "${data_name[@]}"; do
    for kgids in "${kgids[@]}"; do
        python run_prepare_data.py --data_name "$data_name" --kgids "$kgids"
    done
done