import json
import numpy as np
from json import JSONEncoder

class NpArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def save_np_to_json(data, entry_name, file):
    numpyData = {entry_name: data}
    enNumpyData = json.dumps(numpyData, cls=NpArrayEncoder)

    with open(file, 'w') as outfile:
        json.dump(enNumpyData, outfile)


def load_np_from_json(path, entry_name):
    with open(path) as f:
        data = json.load(f)
        data = json.loads(data)
        a_restored = np.asarray(data[entry_name])
    #print(data)
    return a_restored


# npArray1 = np.array([[1.5,2,3],[3,4,5],[6,7,8]])
# npArray2 = np.array([[1.5,8,38],[38,48,58],[86,87,88.5]])
#
# npOut = np.array([npArray1,npArray2])
#
# save_np_to_json(npOut, "array", 'data.json')
# #save_np_to_json(npArray2,"array2", 'data.json')
#
# npdata = load_np_from_json('data.json', "array")
#
# print(npdata)
# print(npdata.dtype)
# print(npdata.size)
# print(npdata.shape)





