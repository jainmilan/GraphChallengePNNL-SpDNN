import os
import sys
import re
import glob
import numpy as np
import pandas as pd

basePath = "/qfs/projects/pacer/milan/logs/GraphChallenge/nsys/"
resultsPath = "/qfs/projects/pacer/milan/logs/GraphChallenge/results/"
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

machine = "a100_80" 
# machine = "a100"
version = "cupy_horovod"

files = glob.glob(basePath + "*gpumemtimesum.csv")
files.sort()

# sys.exit(files)
sel_files = files #[f for f in files if re.match(".*out_p" + machine + "_ng.*" + version + "_.*", f)]
print("[INFO] Number of files matching the pattern: %d" %(len(sel_files)))

# filepath = sel_files[0]
dict_list = []
for filepath in sel_files:
    filename = os.path.basename(filepath)
    # print("[INFO] Filename: %s" %(filename))
    machine_info, train_info = filename.split(machine)[1].split(version)

    # row dict
    row = dict()
    row["partition"] = machine
    row["version"] = version
    row["n_gpus"] = int(machine_info.split('_')[1].strip("ng"))
    row["n_cpus"] = int(machine_info.split('_')[2].strip("nc"))
    row["n_neurons"] = int(train_info.split('_')[1].strip("n"))
    row["n_layers"] = int(train_info.split('_')[2].split(".")[0].strip("nl"))

    try:
        with open(filepath, 'r') as fp:
            filetext = fp.read()
            match = re.findall(r"^\d.*(?:CUDA memcpy DtoH|CUDA memcpy HtoD)\]", filetext, re.M)
            # sys.exit(match)
            row["cuda_time"] = float(match[0].split(',')[0]) + float(match[1].split(',')[0])
    except Exception as e:
        print(e)
        print("Couldn't process file: %s" %(filename))

    kern_file = filepath.replace("gpumemtimesum", "gpukernsum")
    # sys.exit(kern_file)
    try:
        with open(kern_file, 'r') as fp:
            filetext = fp.read()
            match = re.findall(r".*csrgemm2.*", filetext, re.M)
            row["csrgemm_time"] = np.sum([float(m.split(',')[0]) for m in match])
            # sys.exit(match)
            # row["cuda_time"] = float(match[0].split(',')[0]) + float(match[1].split(',')[0])
    except Exception as e:
        print(e)
        print("Couldn't process file: %s" %(filename))

    dict_list.append(row)

# print(row)
df_results = pd.DataFrame(dict_list)
print(df_results.pivot(index=["n_neurons"], columns=["n_gpus"], values=["cuda_time", "csrgemm_time"]).to_latex())
df_results.to_csv(f"{resultsPath}{machine}_{version}.csv")
print(df_results)
print(df_results.shape)
print(df_results["n_neurons"].value_counts())
# for m in matches:
#     print(m)
# print(sel_files)
