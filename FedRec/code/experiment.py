import os
import subprocess
import sys

# List of datasets to run experiments on.
# Add or remove dataset names as needed.
# Available datasets: [ml-100k, ml-1m, LastFM, Yelp, beauty, Sports_and_Outdoors, Toys_and_Games, Steam, Video]
DATASETS_TO_RUN = [
    "ml-100k",
]

# List of algorithms to run.
# These are derived from the filenames in the algorithm directories.
ALGORITHMS = ['base_DHC','UDL_DHC', 'UDL_DDR_DHC', 'UDL_DDR_RESKD_DHC','DDR_RESKD_DHC','RESKD_I_DHC','UDL_RESKD_DHC',]

def run_experiment():
    """
    Runs experiments for all algorithms and datasets.
    """
    main_script_path = os.path.join("main.py")

    for dataset in DATASETS_TO_RUN:
        for algorithm in ALGORITHMS:
            print(f"--- Running experiment: Algorithm={algorithm}, Dataset={dataset} ---")
            command = [
                sys.executable,
                main_script_path,
                "--algorithm", algorithm,
                "--dataset", dataset,
                "--train_data",dataset + ".txt",
            ]
            
            try:
                subprocess.run(command, check=True)
                print(f"--- Finished experiment: Algorithm={algorithm}, Dataset={dataset} ---\n")
            except subprocess.CalledProcessError as e:
                print(f"*** Error running experiment: Algorithm={algorithm}, Dataset={dataset} ***")
                print(f"*** Command: {' '.join(command)} ***")
                print(f"*** Error: {e} ***\n")

if __name__ == "__main__":
    run_experiment()