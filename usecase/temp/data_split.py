import pickle

data_dir = "/home/tomoya-y/workspace/dataset/Sprite"

data                             = pickle.load(open(data_dir + '/data.pkl', 'rb'))

# import ipdb; ipdb.set_trace()

# with open(data_dir + '/test.pkl', mode='wb') as f:
#     pickle.dump(
#         obj = {
#             'X_test'     : X_test,
#             'A_test'     : A_test,
#             'D_test'     : D_test,
#             'c_augs_test': c_augs_test,
#             'm_augs_test': m_augs_test,
#         },
#         file = f,
#     )

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