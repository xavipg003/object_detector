import os

def find_all_files(folder_path):
    files = []
    for root, dirs, filenames in os.walk(folder_path):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

if __name__ == "__main__":
    # folder_path = os.path.expanduser("~/datos_proyecto/FracAtlas/images/Non_fractured")
    # all_files = find_all_files(folder_path)
    # for file in all_files:
    #     split_path = file.split('/')
    #     nombre=split_path[-1].split('.')[0]
    #     root=os.path.expanduser("~/datos_proyecto/datos/train/labels")
    #     os.remove(os.path.join(root, nombre+".txt"))
    folder_path = os.path.expanduser("~/datos_proyecto/datos/train/images")
    all_files = find_all_files(folder_path)
    for file in all_files:
        split_path = file.split('/')
        nombre=split_path[-1].split('.')[0]
        nombre = nombre[:3]
        if nombre=="IMG":
            os.remove(file)
