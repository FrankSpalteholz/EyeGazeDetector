import os

def set_paths(sub_folder):
    output_path = ''
    input_path = ''
    if os.name == 'nt':
        print("Running system is Win10")
        output_path = r'D:\\Dropbox\\work\\Aikia\\EyeTracker\\footage\\render\\' + sub_folder + r'\\'
        # output_path = r'D:\\Dropbox\\work\\Aikia\\EyeTracker\\footage\\render\\' + sub_folder + r'\\'
        # input_path = r'D:\\Dropbox\\work\\Aikia\\EyeTracker\\footage\\' + sub_folder + r'\\'
        input_path = r'D:\\Dropbox\\work\\Aikia\\EyeTracker\\footage\\render\\' + sub_folder + r'\\'
        conf_path = r'D:\\Dropbox\\work\\Aikia\\EyeTracker\\config\\'
        data_out_path = r'D:\\Dropbox\\work\\Aikia\\EyeTracker\\config\\' + sub_folder + r'\\'

    elif os.name == 'posix':
        print("Running system is OSX")
        output_path = '/Users/frankfurt/Dropbox/work/Aikia/EyeTracker/footage/render/' + sub_folder + '/'
        input_path = '/Users/frankfurt/Dropbox/work/Aikia/EyeTracker/footage/render/' + sub_folder + '/'
        # input_path = '/Users/frankfurt/Dropbox/work/Aikia/EyeTracker/footage/' + sub_folder + '/'
        conf_path = '/Users/frankfurt/Dropbox/work/Aikia/EyeTracker/config/'
        data_out_path = conf_path + '/data/' + sub_folder + '/'

    return input_path, output_path, conf_path, data_out_path