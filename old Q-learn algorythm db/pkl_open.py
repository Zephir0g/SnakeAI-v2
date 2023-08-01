import pickle

with open('q_table.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)