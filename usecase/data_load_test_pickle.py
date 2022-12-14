import pickle

data_dir = "/home/tomoya-y/workspace/dataset/Sprite"
data     = pickle.load(open(data_dir + '/test.pkl', 'rb'))

X_test      = data['X_test']
A_test      = data['A_test']
D_test      = data['D_test']
c_augs_test = data['c_augs_test']
m_augs_test = data['m_augs_test']

print(X_test.shape)
print(A_test.shape)
print(D_test.shape)
print(c_augs_test.shape)
print(m_augs_test.shape)