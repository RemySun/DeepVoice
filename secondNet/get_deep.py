import shelve
d = shelve.open('deep_vectors')
deep_vectors = d['deep_vectors']
d.close()
