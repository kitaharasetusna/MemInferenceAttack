import numpy as np
import keras

location_data = np.load('data/location/data_complete.npz')

print(location_data.files)



result_folder = 'result/location/code_publish/'
user_epochs = 200
npzdata=np.load(result_folder+"/models/"+"epoch_{}_weights_user.npz".format(user_epochs), allow_pickle=True)
print(npzdata.files)
weights=npzdata['x']













# import numpy as np

# shuffle_index = "data/location/shuffle_index.npz"
#
# npzdata = np.load(shuffle_index);
# print(npzdata.files)
# shuffle_index = npzdata['x']
# shuffle_index.sort()
# print(shuffle_index)

# all_data_path= "data/location/data_complete.npz"
# npzdata = np.load(all_data_path)
# # x_data=npzdata['x'][:,:]
# y_data=npzdata['y'][:]
# y_data =y_data-1;
# print(y_data)

# import tensorflow as tf
# print(tf.__version__)
