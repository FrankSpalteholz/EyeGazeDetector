import os

# Protected hardcoded root
_OS_ROOT = {
    'nt': 'D:/Dropbox/work/Aikia/EyeTracker',
    'posix': '/Users/frankfurt/Dropbox/work/Aikia/EyeTracker'
}


def get_paths(sub_folder):
    """
    Getter for various project related paths

    Args:
        sub_folder (str): Sub-folder path

    Returns:
        tuple: 0 - input path
               1 - output path
               2 - configuration path
               3 - data out path

    """

    # getting the root
    root = _OS_ROOT.get(os.name)

    if not root:
        # raise exception if root is not found.
        raise RuntimeError('Current OS "{}" is not supported.'.format(root))

    # If the host OS is Windows, replace path-separators
    if os.name == 'nt':
        root = root.replace('/', '\\')

    # Build the paths. os.path.join always uses the correct os-separator.
    # Appending correct os-separator to the end of string.
    # For whatever reason :D
    io_path = os.path.join(root, 'footage', 'render', sub_folder) + os.sep
    conf_path = os.path.join(root, 'config') + os.sep
    data_out_path = os.path.join(root, sub_folder) + os.sep

    return io_path, io_path, conf_path, data_out_path


if __name__ == '__main__':
    print(get_paths('blub'))
