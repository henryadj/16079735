
import networkx as nx
import numpy as np
from graphviz import Graph
from tqdm import tqdm


def generate_graphs(idx,location):

    N = np.random.randint(10,25)
    G = nx.generators.trees.random_tree(N)
    # write_dot(G,'test.dot')
    # pos = graphviz_layout(G,prog="dot")
    # nx.draw(G, with_labels=False, node_shape='s',node_size=900)

    A = nx.linalg.graphmatrix.adjacency_matrix(G)
    A = A.toarray()

    d = Graph(format='jpg')

    # shapes = ['box', 'rectangle', 'ellipse','triangle','diamond']
    shapes = ['box', 'rectangle']
    prob_shape = [0.5, 0.5]
    # prob_shape = [0.45,0.45,0.01,0.06,0.030]
    name = ['big', 'small', 'tiny', 'medium']
    color_list = ['blue','turquoise', 'red', 'sienna']

    edge_list = []

    d.attr('graph', splines='ortho',  esep='-1',size ="12,12")

    # d.attr('graph', splines = 'ortho',nodesep ='2',orientation ='L', ordering = 'out')
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            # and (sorted([i,j]) in edge_list) == False
            if A[i][j] == 1 and (sorted([i,j]) in edge_list) == False :
                edge_list.append([i, j])
                d.node('{}'.format(i),style = 'filled', shape=np.random.choice(shapes, p=prob_shape), label=' ',
                       color=np.random.choice(color_list))
                d.node('{}'.format(j),style = 'filled',  shape=np.random.choice(shapes, p=prob_shape), label=' ',
                       color=np.random.choice(color_list))
                d.edge('{}'.format(i), '{}'.format(j), xlabel='{}'.format(''))


    d.render(location + "/Image_{}".format(idx), cleanup=True)


    return N

n_nodes_train = []
location = 'data/' +input("Enter File Location : ")

n_train = input('Enter Number of Training Data : ')
try:
    n_train = int(n_train)
except ValueError:
    print("That's not an integer! (Number of Training Data)")

n_val = input('Enter Number of Validation Data : ')
try:
    n_val = int(n_val)
except ValueError:
    print("That's not an integer! (Number of Validation Data)")

print('Generating Training Data')
for idx in tqdm(range(n_train)):
    n_nodes_train.append(generate_graphs(idx,location + "/Training"))
np.save( location + '/Training/n_nodes_train.npy',np.array(n_nodes_train))

n_nodes_val = []
print('Generating Validation Data')
for idx_val in tqdm(range(n_val)):
    n_nodes_val.append(generate_graphs(idx_val, location + "/Val"))
np.save(location + '/Val/n_nodes_val.npy',np.array(n_nodes_val))

