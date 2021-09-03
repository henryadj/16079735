

from graphviz import Graph
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout
import os
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

def bb_size( shape):

    """"Function changes bounding box size depending on the shape of the node"""
    if shape == 'circle':
        width = 35
        height = 35
    elif shape == 'rectangle':
        width = 42
        height = 37
    elif shape =='triangle':
        width = 50
        height = 40
    elif shape == 'square':
        width = 40
        height = 35
    elif shape=='oval':
        width = 45
        height =38

    return width, height


def generate_graphs(idx,save_location):

    n_nodes = np.random.randint(18,20)

    G = nx.generators.trees.random_tree(n_nodes)
    pos = graphviz_layout(G, prog="dot", args ='-Gnodesep="1.1"') # Position of nodes

    A = nx.linalg.graphmatrix.adjacency_matrix(G)
    A = A.toarray()



    d = Graph(format='jpg')
    d.engine = 'neato'

    # List of attributes to be randomized
    filled_style = ['filled', '']
    shapes = ['square', 'rectangle','triangle','circle','oval']
    prob_shape = [0.4, 0.4,0.05,0.05,0.1]
    color_list = ['blue',  'turquoise', 'red', 'sienna','brown','green','yellow']

    edge_list = []
    shape_dict ={}

    pos_array = np.zeros((len(pos),2))

    for key in pos:
        temp = pos[key]
        pos_array[key] = np.array(temp)
    mostleft = np.argmin(pos_array[:,0])
    mostright = np.argmax(pos_array[:,0])

    d.attr('graph', splines='ortho', nodesep ='1')

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):

            if A[i][j] == 1 and (sorted([i,j]) in edge_list) == False:
                edge_list.append([i, j])

                if i == mostleft or i == mostright :

                    shape_i ='triangle'

                else :
                    shape_i = np.random.choice(shapes, p=prob_shape)

                if j == mostleft or j == mostright :
                    shape_j = 'triangle'

                else:
                    shape_j = np.random.choice(shapes, p=prob_shape)

                shape_dict[i] = shape_i
                shape_dict[j] = shape_j
                d.node('{}'.format(i), style=np.random.choice(filled_style),shape=shape_i,
                    color=np.random.choice(color_list), pos = str(pos[i][0]/100) + ',' +str(pos[i][1]/100) +'!', penwidth = '3'
                    )
                d.node('{}'.format(j), style=np.random.choice(filled_style), shape=shape_j,
                    color=np.random.choice(color_list), pos = str(pos[j][0]/100) + ',' +str(pos[j][1]/100) +'!', penwidth = '3'
                    )



                d.edge('{}'.format(i), '{}'.format(j), xlabel='{}'.format(''))

    
    image_location = 'graph_data/' + save_location + "/Image_{}".format(idx)
    d.render(image_location, cleanup=True)

    xplot = []
    yplot = []
    for i in range(len(pos)):
        xplot.append(float(pos[(i)][0]))
        yplot.append(float(pos[(i)][1]))

    gg = plt.imread(image_location + '.jpg')


    #Creating and saving bounding boxes
    xplot= (abs(np.array(xplot))**0.995 +22)/gg.shape[1]
    yplot = (abs(gg.shape[0] - np.array(yplot))-5)*0.95/gg.shape[0]

    with open(image_location + '.txt','w') as f:

        for i in range(len(xplot)):
            width, height = bb_size(shape_dict[i])

            f.write('0 '+str(float(xplot[i]))+' '+str(float(yplot[i])) + ' '+str(width*2/gg.shape[1]) +' '+ str(height*2/gg.shape[0]))
            f.write('\n')


save_location = input("Enter File Location : ")
n_data = input("Enter Number of Data : ")
try:
    n_data = int(n_data)
except ValueError:
    print("That's not an integer! (Number of Data)")


print('Generating synthetic group structures...')
for idx in tqdm(range(n_data)):
    generate_graphs(idx,save_location)
print('Done')



## Split into training, validation and test data

print('Splitting data')
path = 'graph_data/' + save_location
win_path = 'graph_data\\' + save_location


if os.path.isdir(path +'/images') == False :
    os.mkdir(path + '/images')
if os.path.isdir(path +'/labels') == False:
    os.mkdir(path + '/labels')

## This two lines of code need to be changed if used on MACOS or Linux to support Linux based system commands
os.system('move ' + win_path +'\*.jpg' + ' ' + win_path+'\images')
os.system('move ' + win_path+'\*.txt' + ' ' + win_path + '\labels')

# Read images and annotations
images = [os.path.join(path + '/images', x) for x in os.listdir(path+'/images')]
annotations = [os.path.join(path + '/labels', x) for x in os.listdir(path +'/labels') if x[-3:] == "txt"]

images.sort()
annotations.sort()

# Split the dataset into train-valid-test splits
train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)


#Utility function to move images
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False

os.mkdir(path + '/images/train')
os.mkdir(path + '/images/val/')
os.mkdir(path +'/images/test/')
os.mkdir(path + '/labels/train/')
os.mkdir(path + '/labels/val/')
os.mkdir(path + '/labels/test/')

# Move the splits into their folders
move_files_to_folder(train_images, path+'/images/train')
move_files_to_folder(val_images, path+'/images/val/')
move_files_to_folder(test_images, path +'/images/test/')
move_files_to_folder(train_annotations, path +'/labels/train/')
move_files_to_folder(val_annotations, path + '/labels/val/')
move_files_to_folder(test_annotations, path + '/labels/test/')

print('Splitting data done')