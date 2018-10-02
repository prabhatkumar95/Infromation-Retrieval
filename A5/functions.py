import numpy as np
import queue



def create_adjmatrix(data):
    d= len(np.unique(data))
    matrix = np.zeros(d*d).reshape((d,d))
    index =0
    mapping={}
    for i in range(0,data.shape[0]):
        if data[i][0] not in mapping.keys():
            mapping[data[i][0]] = index
            index=index+1
        if data[i][1] not in mapping.keys():
            mapping[data[i][1]] = index
            index=index+1

        matrix[mapping[data[i][0]]][mapping[data[i][1]]]=1
    return mapping,matrix


def clustering_c(data):
    c_c=[]
    for i in range(0,data.shape[0]):
       index=np.nonzero(data[i,:])[0]
       index = list(index)
       # print(index)
       if(len(index)>1):
            temp_matrix = data[np.ix_(index,index)]
            # print(temp_matrix.shape)
            count = np.count_nonzero(temp_matrix.flatten(),axis=0)
            # print(count)
            count=count/2
            t=len(index)
            c_c.append(count/(t*(t-1)/2))
       elif len(index)==0:
           c_c.append(0)
       else:
           c_c.append(0)
    return c_c


def bfs(data,s):
    visited = np.zeros(data.shape[0])
    spath = np.full((data.shape[0],),data.shape[0])
    visited[s]=1
    spath[s]=0

    q = queue.Queue()
    q.put(s)

    while not q.empty():
        parent = q.get()
        children = np.nonzero(data[parent,:])[0]
        # print(children.shape)
        for i in range(0,len(children)):
            if(visited[children[i]]!=1):
                spath[children[i]]=spath[parent]+1
                q.put(children[i])
                visited[children[i]]=1
    return np.average(spath,axis=0)

def closeness(data):
    close=np.zeros(data.shape[0])
    for i in range(0,len(close)):
        close[i]=bfs(data,i)
    return close
#
# def betweeness(data):
#     import networkx as nx
#     g=nx.from_numpy_matrix(data)
#     return nx.betweenness_centrality(g)