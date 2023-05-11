import os


def get_absolute_file_paths(directory):
    for dirpath, _, filenames in os.walk(directory):
        files = []
        for f in filenames:
            files.append(os.path.abspath(os.path.join(dirpath, f)))
        return files
