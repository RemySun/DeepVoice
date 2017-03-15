import numpy as np
from sklearn.manifold import TSNE
import shelve


d = shelve.open('deep_vectors')
deep_vectors = d['deep_vectors']
d.close()

model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
X = deep_vectors
res = model.fit_transform(X) 

labels = pickle.load(open("train_labels.p", "rb"))

def get_index(name):
  index = []
  for i in range(len(deep_vectors)):
    if name == labels[i]:
      index.append(i)
  return(index)

def plot_tsne(name, couleur):
  ind = get_index(name)
  plt.scatter(res[ind,0], res[ind,1], color = couleur)


