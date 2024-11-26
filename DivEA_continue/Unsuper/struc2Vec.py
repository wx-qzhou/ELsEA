import numpy as np
from gensim.models import Word2Vec

class Struc2Vec:
    def __init__(self, graph, dimensions=64, walk_length=10, num_walks=50, workers=4):
        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.node_embeddings = {}
        self.model = None
        self._prepare_walks()

    def _prepare_walks(self):
        self.walks = []
        for node in self.graph.nodes:
            for _ in range(self.num_walks):
                walk = self._random_walk(node)
                self.walks.append(walk)
        
    def _random_walk(self, start_node):
        walk = [start_node]
        current_node = start_node
        for _ in range(self.walk_length - 1):
            neighbors = list(self.graph.neighbors(current_node))
            if not neighbors:
                break
            next_node = np.random.choice(neighbors)
            walk.append(next_node)
            current_node = next_node
        return walk

    def train(self):
        # Flatten walks and train Word2Vec model
        flattened_walks = [list(map(str, walk)) for walk in self.walks]
        self.model = Word2Vec(sentences=flattened_walks, vector_size=self.dimensions, window=5, min_count=1, sg=1, workers=self.workers)

        # Retrieve embeddings
        self.node_embeddings_ = {node: self.model.wv[str(node)] for node in self.graph.nodes}

        self.node_embeddings = {}
        for node in self.graph.nodes:
            index = 1
            self.node_embeddings[node] = self.node_embeddings_[node]
            for node_ in self.graph.neighbors(node):
                index += 1
                self.node_embeddings[node] = self.node_embeddings[node] + self.node_embeddings_[node_]
            self.node_embeddings[node] = self.node_embeddings[node] / index

    def get_embedding(self, node):
        return self.node_embeddings.get(node, None)

