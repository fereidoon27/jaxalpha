

###------------------------------------------------------------------------------------
####len

# def get_string_length(text):
#   return len(text)

# # Get user input
# user_string = input("Enter a string: ")

# # Get the length of the string
# string_length = get_string_length(user_string)

# # Print the length
# print(f"The length of the string is: {string_length}")


###------------------------------------------------------------------------------------
### seq to list '7CWP , 7CN6 , 6UF2  , 7PZT, 7ROA, 8EM5, 8PBV'
# seq = "2LHC ,2LHD, 2LHE, 2LHG, 8D27, 6Y4F, 6ZYC, 8UYS"
# target = list(seq.split(','))
# target = [item.strip() for item in target]
# print (target)


### mkdir
# target = ['7JTL', '7CWP', '7CN6', '6UF2', '7PZT', '7ROA', '8EM5', '8PBV']
# base_address = "/home/koohi/fereidoon/ParallelFold/claude_output/"
# for item in target:
#   # Create the new directory path by replacing "7CWP" with the current item
#   new_dir = base_address + item + "/colab"
#     # Try creating the directory. If it already exists, handle the error gracefully.
#   try:
#     import os
#     os.makedirs(new_dir)
#     print(f"Directory created: {new_dir}")
#   except FileExistsError:
#     print(f"Directory already exists: {new_dir}")
#   except OSError as e:
#     print(f"Error creating directory {new_dir}: {e}")

###------------------------------------------------------------------------------------
### copy feature.pkl to target folder in 123

# import shutil

# address1 = "/home/koohi/fereidoon/ParallelFold/output/8PBV/colab/feature.pkl"
# address2 = "/home/koohi/fereidoon/ParallelFold/claude_output/6UF2/AF/feature.pkl"

# # Try copying the file. If it doesn't exist or there are errors, handle them gracefully.
# try:
#   shutil.copy(address1, address2)
#   print(f"File copied successfully: {address1} -> {address2}")
# except FileNotFoundError:
#   print(f"Error: File not found at {address1}")
# except OSError as e:
#   print(f"Error copying file: {e}")


###------------------------------------------------------------------------------------
### copy feature.pkl to target folder in windows

# import os
# import shutil

# # List of PDB IDs
# target = ['7JTL', '7CWP', '7CN6', '6UF2', '7PZT', '7ROA', '8EM5', '8PBV']

# # Base paths
# source_base_path = 'E:\\result'
# destination_base_path = 'E:\\target\\input'

# # Subdirectory templates
# subdirs = ['AF-feature', 'colab-feature']

# # Copy files
# for pdb_id in target:
#     for subdir in subdirs:
#         source_file_path = os.path.join(source_base_path, pdb_id, subdir, 'feature.pkl')
#         destination_dir_path = os.path.join(destination_base_path, pdb_id, subdir)
#         destination_file_path = os.path.join(destination_dir_path, 'feature.pkl')

#         # Create the destination directory if it does not exist
#         os.makedirs(destination_dir_path, exist_ok=True)

#         # Copy the file
#         shutil.copy2(source_file_path, destination_file_path)

# print("Files copied successfully!")




###------------------------------------------------------------------------------------
### making ipynb files:

# import os
# import shutil

# # List of PDB IDs
# target = ['2LHC', '2LHD', '2LHE', '2LHG', '8D27', '6Y4F', '6ZYC', '8UYS']

# # Base paths
# source_file_path = 'E:\\targe-other-journal\\Colabfold-temp.ipynb'
# destination_base_path = 'E:\\targe-other-journal\\input'

# # Copy and rename files
# for pdb_id in target:
#     destination_dir_path = os.path.join(destination_base_path, pdb_id, 'colab-feature')
#     destination_file_path = os.path.join(destination_dir_path, f'Colabfold-{pdb_id}.ipynb')

#     # Create the destination directory if it does not exist
#     os.makedirs(destination_dir_path, exist_ok=True)

#     # Copy and rename the file
#     shutil.copy2(source_file_path, destination_file_path)

# print("Files copied and renamed successfully!")


###------------------------------------------------------------------------------------
###Empty Pickle file
# import os
# import pickle

# # Define filename
# filename = "feature.pkl"

# # Check if file exists
# if not os.path.isfile(filename):
#   # Open the file in write-binary mode for pickling
#   with open(filename, 'wb') as file:
#     # Pickle an empty list (or any empty object) to create the file
#     pickle.dump([], file)

# print(f"Empty file {filename} created successfully!")

###------------------------------------------------------------------------------------
### Apped prefix to each element in list

# target = ['6UF2', '7CN6', '7CWP', '7JTL', '7PZT', '7ROA', '8EM5', '8PBV']
# ren =[]
# for i in target:
#     ren.append("pdb_compare_final_"+ i)

# print(ren)

###------------------------------------------------------------------------------------
### Copy .pkl files for each PDB ID from Source to destination
# import os
# import shutil

# # List of PDB IDs
# target = ['7JTL', '7CWP', '7CN6', '6UF2', '7PZT', '7ROA', '8EM5', '8PBV']

# # Source and destination base paths
# source_base_path = 'E:\\result\\123\\claude_output'
# destination_base_path = 'E:\\result_model_pkl_claude'

# # Files to copy
# files_to_copy = ['result_model_1_pred_0.pkl', 'result_model_1_pred_1.pkl']

# # Copy files for each PDB ID
# for pdb_id in target:
#     source_folder_path = os.path.join(source_base_path, pdb_id, 'colab', pdb_id)
#     destination_folder_path = os.path.join(destination_base_path, pdb_id)

#     # Create destination folder if it does not exist
#     os.makedirs(destination_folder_path, exist_ok=True)

#     # Copy each file
#     for file_name in files_to_copy:
#         source_file_path = os.path.join(source_folder_path, file_name)
#         destination_file_path = os.path.join(destination_folder_path, file_name)

#         # Check if the source file exists before copying
#         if os.path.exists(source_file_path):
#             shutil.copy2(source_file_path, destination_file_path)
#         else:
#             print(f"Source file {source_file_path} does not exist. Skipping.")

# print("Files copied successfully!")

###------------------------------------------------------------------------------------
### make folder name

target_1_5 = ['2LHC', '2LHD', '2LHE', '2LHG','6UF2']
target_6_10 = ['6Y4F', '6ZYC', '7JTL', '7CWP', '7CN6']
target_11_15 = ['7PZT', '7ROA', '8EM5', '8PBV', '8D27']

raw_folder_name =[]
batch = ['1_5', '6_10', '11_15']
gpu = ['123', 'A100']
order = ['ascend', 'descend']

# for x in batch:
#     for i in order:
#         for j in gpu:
#             raw_folder_name.append(f"{x}_{i}_{j}")


for i in order:
    for j in gpu:
        raw_folder_name.append(f"{i}_{j}")

#print (raw_folder_name)

folder_name = []
for i in target_1_5:
    for j in raw_folder_name[0:4]:
        folder_name.append(f"{i}_{j}")

for i in target_6_10:
    for j in raw_folder_name[0:4]:
        folder_name.append(f"{i}_{j}")

for i in target_11_15:
    for j in raw_folder_name[0:4]:
        folder_name.append(f"{i}_{j}")

# print (folder_name)
# print (len(folder_name))

###------------------------------------------------------------------------------------
### make dir for element in lists
import os

# Define the base path
base_path = "E:\\timing_multi"

# Iterate through each element in the list
for element in folder_name:
  # Construct the full folder path
  folder_path = os.path.join(base_path, element)
  
  # Check if the folder already exists
  if not os.path.exists(folder_path):
    # Create the folder if it doesn't exist
    os.makedirs(folder_path)
    print(f"Folder created: {folder_path}")
  else:
    print(f"Folder already exists: {folder_path}")
