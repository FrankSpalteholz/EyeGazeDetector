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
        dict: Keys: 'footage', 'config', 'out'

    """

    # getting the root
    root = _OS_ROOT.get(os.name)

    if not root:
        # raise exception if root is not found.
        raise RuntimeError('Current OS "{}" is not supported.'.format(root))

    # If the host OS is Windows, replace path-separators
    if os.name == 'nt':
        root = root.replace('/', '\\')

    return {
        'footage': os.path.join(root, 'footage', 'render', sub_folder) + os.sep,
        'config': os.path.join(root, 'config') + os.sep,
        'out': os.path.join(root, sub_folder) + os.sep
    }
