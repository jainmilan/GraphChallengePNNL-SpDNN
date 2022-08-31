import os
import re
import glob
import pandas as pd

basePath = "/qfs/people/jain432/pacer_remote/logs/GraphChallenge/sbatch/"
resultsPath = "/qfs/people/jain432/pacer_remote/logs/GraphChallenge/results/"
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

machine = "a100_80" 
# machine = "a100"
version = "cupy_mpinp"
# version = "cupy_spmm_mpinp"

if version == "cupy_spmm_mpinp":
    search_text = "SpMM"
elif version == "cupy_mpinp":
    search_text = "SpGEMM"

files = glob.glob(basePath + "*.txt")
files.sort()
print(".*out_p" + machine + "_ng.*" + version + "_.*")

sel_files = [f for f in files if re.match(".*out_p" + machine + "_ng.*" + version + "_.*", f)]
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
            # print(filetext)
            match = re.findall(r"^\[.*" + search_text + ".*$", filetext, re.M)[0]
            # print(match)
            results = match.split(',')
            # print(results)
            nums = [float(r.split(':')[-1].strip()) for r in results]
            # nums = [float(r.split(':')[-1].strip()) for r in results[1:]]
            # print(nums)
            row["spgemmTime"] = nums[0]
            row["spgemmRate"] = nums[1]
            row["iterationTime"] = nums[2]
            row["iterationRate"] = nums[3]

        dict_list.append(row)
    except Exception as e:
        print(e)
        print("Couldn't process file: %s" %(filename))

# print(row)
df_results = pd.DataFrame(dict_list)
df_results.to_csv(f"{resultsPath}{machine}_{version}.csv")
print(df_results.sort_values(["n_gpus", "n_neurons", "n_layers"]))
# print(df_results.shape)
# print(df_results["n_neurons"].value_counts())
# for m in matches:
#     print(m)
# print(sel_files)
