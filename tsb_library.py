import matplotlib.pyplot as plt
import math

data= [ [27,36],[91,4],[2,53],[92,82],[21,16],[18,95],[47,26],[71,38],[69,12]]

def tsp(data):
    # build a graph
    G,cost_mat = build_graph(data)

    # build a minimum spanning tree
    MSTree = minimum_spanning_tree(G)

    # find odd vertexes
    odd_vertexes = find_odd_vertexes(MSTree)

    # add minimum weight matching edges to MST
    minimum_weight_matching(MSTree, G, odd_vertexes)

    # find an eulerian tour
    eulerian_tour = find_eulerian_tour(MSTree, G)

    current = eulerian_tour[0]
    path = [current]
    visited = [False] * len(eulerian_tour)
    visited[eulerian_tour[0]] = True
    length = 0

    for v in eulerian_tour:
        if not visited[v]:
            path.append(v)
            visited[v] = True

            length += G[current][v]
            current = v
    length +=G[current][eulerian_tour[0]]
    path.append(eulerian_tour[0])
    return length, path,cost_mat


def get_length(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1.0 / 2.0)


def build_graph(data):
    graph = {}
    cost_mat=[]
    for this in range(len(data)):
        p=[]
        for another_point in range(len(data)):
            if this != another_point:
                if this not in graph:
                    graph[this] = {}
                graph[this][another_point] = int(get_length(data[this][0], data[this][1], data[another_point][0],
                                                        data[another_point][1]))
                p.append(graph[this][another_point])
            else:
                p.append(0)
        cost_mat.append(p)
    return graph,cost_mat


class UnionFind:
    def __init__(self):
        self.weights = {}
        self.parents = {}

    def __getitem__(self, object):
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        return iter(self.parents)

    def union(self, *objects):
        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r], r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest


def minimum_spanning_tree(G):
    tree = []
    subtrees = UnionFind()
    for W, u, v in sorted((G[u][v], u, v) for u in G for v in G[u]):
        if subtrees[u] != subtrees[v]:
            tree.append((u, v, W))
            subtrees.union(u, v)
    return tree


def find_odd_vertexes(MST):
    tmp_g = {}
    vertexes = []
    for edge in MST:
        if edge[0] not in tmp_g:
            tmp_g[edge[0]] = 0
        if edge[1] not in tmp_g:
            tmp_g[edge[1]] = 0
        tmp_g[edge[0]] += 1
        tmp_g[edge[1]] += 1
    for vertex in tmp_g:
        if tmp_g[vertex] % 2 == 1:
            vertexes.append(vertex)
    return vertexes


def minimum_weight_matching(MST, G, odd_vert):
    import random
    random.shuffle(odd_vert)
    while odd_vert:
        v = odd_vert.pop()
        length = float("inf")
        u = 1
        closest = 0
        for u in odd_vert:
            if v != u and G[v][u] < length:
                length = G[v][u]
                closest = u

        MST.append((v, closest, length))
        odd_vert.remove(closest)


def find_eulerian_tour(MatchedMSTree, G):
    # find neigbours
    neighbours = {}
    for edge in MatchedMSTree:
        if edge[0] not in neighbours:
            neighbours[edge[0]] = []

        if edge[1] not in neighbours:
            neighbours[edge[1]] = []

        neighbours[edge[0]].append(edge[1])
        neighbours[edge[1]].append(edge[0])

    # finds the hamiltonian circuit
    start_vertex = MatchedMSTree[0][0]
    EP = [neighbours[start_vertex][0]]

    while len(MatchedMSTree) > 0:
        for i, v in enumerate(EP):
            if len(neighbours[v]) > 0:
                break

        while len(neighbours[v]) > 0:
            w = neighbours[v][0]

            remove_edge_from_matchedMST(MatchedMSTree, v, w)

            del neighbours[v][(neighbours[v].index(w))]
            del neighbours[w][(neighbours[w].index(v))]

            i += 1
            EP.insert(i, w)

            v = w

    return EP


def remove_edge_from_matchedMST(MatchedMST, v1, v2):
    for i, item in enumerate(MatchedMST):
        if (item[0] == v2 and item[1] == v1) or (item[0] == v1 and item[1] == v2):
            del MatchedMST[i]
    return MatchedMST

def cost_change(cost_mat, n1, n2, n3, n4):
    return cost_mat[n1][n3] + cost_mat[n2][n4] - cost_mat[n1][n2] - cost_mat[n3][n4]


def two_opt(route, cost_mat):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
                if cost_change(cost_mat, best[i - 1], best[i], best[j - 1], best[j]) < 0:
                    best[i:j] = best[j - 1:i - 1:-1]
                    improved = True
        route = best
    return best

def solve_tsp_nearest(distances):
    num_cities = len(distances)
    visited = [False] * num_cities
    tour = []
    total_distance = 0
    
    # Start at the first city
    current_city = 0
    tour.append(current_city)
    visited[current_city] = True
    
    # Repeat until all cities have been visited
    while len(tour) < num_cities:
        nearest_city = None
        nearest_distance = 100000000

        # Find the nearest unvisited city
        for city in range(num_cities):
            if not visited[city]:
                distance = distances[current_city][city]
                if distance < nearest_distance:
                    nearest_city = city
                    nearest_distance = distance

        # Move to the nearest city
        current_city = nearest_city
        tour.append(current_city)
        visited[current_city] = True
        total_distance += nearest_distance

    # Complete the tour by returning to the starting city
    tour.append(0)
    total_distance += distances[current_city][0]

    return tour, total_distance

maxsize = float('inf')
def copyToFinal(curr_path):
	final_path[:N + 1] = curr_path[:]
	final_path[N] = curr_path[0]

def firstMin(adj, i):
	min = maxsize
	for k in range(N):
		if adj[i][k] < min and i != k:
			min = adj[i][k]
	return min

def secondMin(adj, i):
	first, second = maxsize, maxsize
	for j in range(N):
		if i == j:
			continue
		if adj[i][j] <= first:
			second = first
			first = adj[i][j]

		elif(adj[i][j] <= second and
			adj[i][j] != first):
			second = adj[i][j]

	return second

def TSPRec(adj, curr_bound, curr_weight,level, curr_path, visited):
	global final_res
	if level == N:
		if adj[curr_path[level - 1]][curr_path[0]] != 0:
			curr_res = curr_weight + adj[curr_path[level - 1]]\
										[curr_path[0]]
			if curr_res < final_res:
				copyToFinal(curr_path)
				final_res = curr_res
		return
	for i in range(N):
		if (adj[curr_path[level-1]][i] != 0 and
							visited[i] == False):
			temp = curr_bound
			curr_weight += adj[curr_path[level - 1]][i]
			if level == 1:
				curr_bound -= ((firstMin(adj, curr_path[level - 1]) +firstMin(adj, i)) / 2)
			else:
				curr_bound -= ((secondMin(adj, curr_path[level - 1]) +firstMin(adj, i)) / 2)

			if curr_bound + curr_weight < final_res:
				curr_path[level] = i
				visited[i] = True
				TSPRec(adj, curr_bound, curr_weight,
					level + 1, curr_path, visited)
			curr_weight -= adj[curr_path[level - 1]][i]
			curr_bound = temp
			visited = [False] * len(visited)
			for j in range(level):
				if curr_path[j] != -1:
					visited[curr_path[j]] = True

def held_karp(i, mask,memo,n):
    if mask == ((1 << i) | 3):
        return cost_mat[1][i]
    if memo[i][mask] != -1:
        return memo[i][mask]
 
    res = 10**9
    for j in range(1, n+1):
        if (mask & (1 << j)) != 0 and j != i and j != 1:
            res = min(res, held_karp(j, mask & (~(1 << i))) + cost_mat[j][i])
    memo[i][mask] = res
    return res


def TSP(adj):
	curr_bound = 0
	curr_path = [-1] * (N + 1)
	visited = [False] * N
	for i in range(N):
		curr_bound += (firstMin(adj, i) +
					secondMin(adj, i))
	curr_bound = math.ceil(curr_bound / 2)
	visited[0] = True
	curr_path[0] = 0
	TSPRec(adj, curr_bound, 0, 1, curr_path, visited)


def path(data,pp,s):
    x=[]
    y=[]
    for i in range(len(pp)):
        x.append(data[pp[i]][0])
        y.append(data[pp[i]][1])
    plt.title(s)
    plt.scatter(x, y)
    plt.plot(x, y)
    plt.show()

lent,route,cost_mat=tsp(data)
two_opt_path = two_opt(route,cost_mat)
memo = [[-1]*((len(data)+1)) for _ in range(len(data)+1)]
N= len(data)
final_path = [None] * (N + 1)
visited = [False] * N
final_res = maxsize
TSP(cost_mat)
held_path=final_path
held_dist=final_res

d=0
for i in range (1,len(two_opt_path)):
    d+=cost_mat[two_opt_path[i]][two_opt_path[i-1]]

NN_path, total_distance = solve_tsp_nearest(cost_mat)

c=0
while(c!=6):
    print("Choose any algorithm: ")
    print("1) Christofides Algorithm")
    print("2) Nearest Neighbour Algorithm ")
    print("3) Two opt Optimization")
    print("4) Branch And Bound ")
    print("5) Held Karp Algorithm ")
    print("6) Exit")
    print("")
    c=int(input("Enter Your Choice "))
    if(c==1):
        print("Path for Christofides: ")
        print(route)
        print("Distance of the path :", lent)
        path(data,route,"Christofides")
    elif(c==2):
        print("Path for Nearest Neighbour : ")
        print(NN_path)
        print("Distance of the path :", total_distance)
        path(data,NN_path,"Nearest Neighbour")
    elif(c==3):
        print("Path for two-opt optimization : ")
        print(two_opt_path)
        print("Distance of the path :", d)
        path(data,two_opt_path,"Two opt ")
    elif(c==4):
        print("Path for Branch and Bound Exact Algorithm: ")
        print(final_path)
        print("Distance of the path :", final_res)
        path(data,final_path," Branch and Bound ")
    elif(c==5):
        print("Path for Held Karp Exact Algorithm: ")
        print(held_path)
        print("Distance of the path :", held_dist)
        path(data,held_path," Held Karp Algorithm ")
    elif(c==6):
        break
    else:
        print("Enter a valid Input ")
        print("")
