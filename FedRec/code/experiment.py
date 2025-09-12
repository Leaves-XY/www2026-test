import os
import subprocess
import sys

# List of datasets to run experiments on.
# Add or remove dataset names as needed.
# Available datasets: ['ml-100k', 'ml-1m', 'LastFM', 'Yelp','Toys_and_Games','Sports_and_Outdoors','Video','Steam','beauty']
MODEL=['SASRec']

DATASETS_TO_RUN =['ml-100k', 'ml-1m', 'LastFM', 'Yelp','Toys_and_Games','Sports_and_Outdoors','Video','Steam','beauty']

# List of algorithms to run.
# These are derived from the filenames in the algorithm directories.
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