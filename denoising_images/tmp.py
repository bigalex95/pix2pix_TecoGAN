import os
import shutil
root_dir = './'
dest_dir = './test_scripts'
os.listdir(root_dir)

for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if os.path.isdir(folder_path):
        # for subfolder in os.listdir(folder_path):
        # if subfolder == 'test_results':
        #     subfolder_path = os.path.join(root_dir, folder, subfolder)
        #     print(subfolder_path)
        #     for filename in os.listdir(subfolder_path):
        #         file_path = os.path.join(
        #             root_dir, folder, subfolder, filename)
        #         dest_path = os.path.join(dest_dir, folder)
        #         dest_path_file = os.path.join(dest_dir, folder, filename)
        #         if not os.path.exists(dest_path):
        #             os.mkdir(dest_path)
        #         shutil.copy(file_path, dest_path_file)
        #         print("Copied ", file_path, "to", dest_path)

        for filename in os.listdir(folder_path):
            if filename == "pix2pix.ipynb":
                file_path = os.path.join(
                    root_dir, folder, filename)
                dest_path = os.path.join(dest_dir, folder)
                dest_path_file = os.path.join(dest_dir, folder, filename)
                if not os.path.exists(dest_path):
                    os.mkdir(dest_path)
                shutil.copy(file_path, dest_path_file)
                print("Copied ", file_path, "to", dest_path)
