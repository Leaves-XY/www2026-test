import os
import subprocess
import sys

# List of datasets to run experiments on.
# Add or remove dataset names as needed.
# Available datasets: ['ml-100k', 'ml-1m', 'LastFM', 'Yelp','Video','Toys_and_Games','Sports_and_Outdoors','Steam','beauty']
MODEL=['SASRec','BSARec']

DATASETS_TO_RUN =['ml-100k', 'ml-1m', 'LastFM', 'Yelp']

# List of algorithms to run.
# These are derived from the filenames in the algorithm directories.
#['base','UDL','UDL_DDR','UDL_DDR_RESKD']
#['base_Top_k','UDL_Top_k','UDL_DDR_Top_k','UDL_DDR_RESKD_Top_k']
ALGORITHMS = ['base','UDL','UDL_DDR','UDL_DDR_RESKD']

def run_experiment():
    """
    Runs experiments for all algorithms and datasets.
    """
    main_script_path = os.path.join("main.py")

    for model in MODEL:
        for dataset in DATASETS_TO_RUN:
            for algorithm in ALGORITHMS:
                print(f"--- Running experiment:Model={model} Algorithm={algorithm}, Dataset={dataset} ---")
                command = [
                    sys.executable,
                    main_script_path,
                    "--model", model,
                    "--algorithm", algorithm,
                    "--dataset", dataset,
                    "--train_data",dataset + ".txt",
                    "--early_stop",'9',
                    "--lr",'0.001',
                    "--kd_lr",'0.001',
                    "--max_seq_len",'200',
                    "--dim_s","16",
                    "--dim_m","32",
                    "--dim_l","64",
                    "--hidden_size","64",
                    "--device_split", "0.5", "0.3",
                    "--top_k_ratio","0.3",
                    "--LDP_lambda","0.01"
                ]

                try:
                    subprocess.run(command, check=True)
                    print(f"--- Finished experiment:Model={model} Algorithm={algorithm}, Dataset={dataset} ---\n")
                except subprocess.CalledProcessError as e:
                    print(f"*** Error running experiment: Model={model} Algorithm={algorithm}, Dataset={dataset} ***")
                    print(f"*** Command: {' '.join(command)} ***")
                    print(f"*** Error: {e} ***\n")

if __name__ == "__main__":
    run_experiment()