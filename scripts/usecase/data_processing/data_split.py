import pickle

# data_dir = "/home/tomoya-y/workspace/dataset/Sprite"
data_dir = "~/workspace/dataset/Sprite"


data = pickle.load(open(data_dir + '/data.pkl', 'rb'))


save_train = False

if save_train:
    with open(data_dir + '/train.pkl', mode='wb') as f:
        pickle.dump(
            obj = {
                'X_train'     : data['X_train'],
                'A_train'     : data['A_train'],
                'D_train'     : data['D_train'],
                'c_augs_train': data['c_augs_train'],
                'm_augs_train': data['m_augs_train'],
            },
            file = f,
        )
else:
    with open(data_dir + '/test.pkl', mode='wb') as f:
        pickle.dump(
            obj = {
                'X_test'     : data['X_test'],
                'A_test'     : data['A_test'],
                'D_test'     : data['D_test'],
                'c_augs_test': data['c_augs_test'],
                'm_augs_test': data['m_augs_test'],
            },
            file = f,
        )