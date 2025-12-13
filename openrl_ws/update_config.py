# Usage: python update_config.py --filepath $script_dir/config.py
# delete the content of target_file and move content of source_file to target_file
def revise_go1push_config(source_file_path, target_file_path):
    with open(source_file_path, "r") as source_file:
        source_content = source_file.read()
    with open(target_file_path, "w") as target_file:
        target_file.seek(0)
        target_file.write(source_content)
        # save and close target_file
        target_file.truncate()


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--filepath", type=str, required=True)
args = parser.parse_args()
source_file_path = args.filepath
target_file_path="./mqe/envs/configs/go1_push_mid_config.py"
revise_go1push_config(source_file_path, target_file_path)
print("*******************************************************************************************************************************************")
print("HAVE REVISED THE CONFIG FILE FOR GO1 PUSH EXPERIMENT OF TASK: ", args.filepath)
print("*******************************************************************************************************************************************")