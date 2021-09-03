
import numpy as np
import networkx as nx

from graphviz import Graph
from tqdm import tqdm



def generate_graphs(idx,save_location):

    n_nodes = np.random.randint(10,25)
    G = nx.generators.trees.random_tree(n_nodes)


    A = nx.linalg.graphmatrix.adjacency_matrix(G)
    A = A.toarray()

    d = Graph(format='jpg')

    filled_style = ['filled','']
    shapes = ['box', 'rectangle']
    prob_shape = [0.5, 0.5]
    color_list = ['blue',  'turquoise', 'red', 'sienna']

    edge_list = []
    label = np.random.randint(0,2)

    colour = 'white' if label else np.random.choice(color_list)
    d.attr('graph', splines='ortho',  esep='1')

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):

            if A[i][j] == 1 and (sorted([i,j]) in edge_list) == False:
                edge_list.append([i, j])
                d.node('{}'.format(i), style=np.random.choice(filled_style),shape=np.random.choice(shapes, p=prob_shape),
                       color=colour, label = ''
                       )
                d.node('{}'.format(j), style=np.random.choice(filled_style), shape=np.random.choice(shapes, p=prob_shape),
                       color= colour, label = ''
                       )


                d.edge('{}'.format(i), '{}'.format(j), xlabel='{}'.format(''))

    image_location = save_location + "/Image_{}".format(idx)
    d.render(image_location, cleanup=True)
    return label



## File location
save_location = 'data/' + input("Enter File Location : ")


## User input of data
n_train = input("Enter Number of Training Data : ")
try:
    n_train = int(n_train)
except ValueError:
    print("That's not an integer! (Number of Training Data)")

n_val = input("Enter Number of Validation Data : ")

try:
    n_val = int(n_val)
except ValueError:
    print("That's not an integer! (Number of Validation Data)")


labels = []
print('Generating training data')
for idx in tqdm(range(n_train)):

    labels.append(generate_graphs(idx,save_location + '/Train'))
np.save(save_location + '/Train/Labels.npy' , np.array(labels))


val_labels  = []
print('Generating validation data')
for idx in tqdm(range(n_val)):
    val_labels.append(generate_graphs(idx,save_location+'/Val'))

np.save(save_location +'/Val/Labels.npy', np.array(val_labels))

