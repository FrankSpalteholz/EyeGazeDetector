import os


_OS_ROOT = {
    'nt': 'D:/Dropbox/work/Aikia/EyeTracker',
    'posix': '/Users/frankfurt/Dropbox/work/Aikia/EyeTracker'
}


def get_paths(sub_folder):
    root = _OS_ROOT.get(os.name)
    if not root:
        raise RuntimeError('Current OS "{}" is not supported.'.format(root))

    if os.name == 'nt':
        root = root.replace('/', '\\')

    io_path = os.path.join(root, 'footage', 'render', sub_folder) + os.sep
    conf_path = os.path.join(root, 'config') + os.sep
    data_out_path = os.path.join(root, sub_folder) + os.sep

    return io_path, io_path, conf_path, data_out_path


if __name__ == '__main__':
    print(get_paths('blub'))
