import random
from collections import defaultdict

def build_de_bruijn_graph(sequence, k):
    """Builds a De Bruijn graph from a DNA sequence."""
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)

    # Create nodes (k-1 mers) and edges (k mers)
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        prefix = kmer[:-1]
        suffix = kmer[1:]
        
        graph[prefix].append(suffix)
        out_degree[prefix] += 1
        in_degree[suffix] += 1

    return graph, in_degree, out_degree

def get_start_node(graph, in_degree, out_degree):
    """Finds the starting node for the Eulerian path."""
    start_node = None
    for node in set(in_degree.keys()).union(set(out_degree.keys())):
        # In a directed graph, the start node of an Eulerian path has out_degree - in_degree == 1
        if out_degree[node] - in_degree[node] == 1:
            return node
            
    # If it's a perfect Eulerian circuit (balanced degrees), pick any node with outgoing edges
    if not start_node:
        start_node = list(graph.keys())[0]
    return start_node

def generate_adversarial_negative(sequence, k=3):
    """
    Generates a negative sequence with the exact same k-mer frequency 
    as the input sequence by finding an alternative Eulerian path.
    """
    graph, in_degree, out_degree = build_de_bruijn_graph(sequence, k)
    start_node = get_start_node(graph, in_degree, out_degree)

    # NOVEL STEP: Shuffle the outgoing edges at every node. 
    # This forces Hierholzer's algorithm to take alternative branches,
    # destroying long-range structure while preserving k-mer counts.
    for node in graph:
        random.shuffle(graph[node])

    # Hierholzer's algorithm to find an Eulerian path
    stack = [start_node]
    path = []

    while stack:
        u = stack[-1]
        if graph[u]:
            # If there are unexplored edges, follow one
            v = graph[u].pop()
            stack.append(v)
        else:
            # If stuck, backtrack and add to path
            path.append(stack.pop())

    # The path is constructed backwards, so reverse it
    path = path[::-1]

    # Reconstruct the string from the path of (k-1)-mers
    if not path:
        return ""
        
    adversarial_seq = path[0]
    for node in path[1:]:
        adversarial_seq += node[-1]

    return adversarial_seq

# --- Example Usage on Your Sequence ---
# if __name__ == "__main__":
#     # The sequence transcribed from your image
#     positive_sequence = "TAGAACAGCCTTGTACGGGAGCTGCAATCACCTCTGCTCGTCTCCATTATAGAACACTTG"
    
#     print("Original Positive Sequence:")
#     print(positive_sequence)
    
#     # We use k=3 to ensure enough branching points (hubs) in the graph 
#     # for a 60-bp sequence. For longer sequences, you can use k=4 or k=5.
#     negative_sequence = generate_adversarial_negative(positive_sequence, k=3)
    
#     print("\nGenerated Adversarial Negative Sequence:")
#     print(negative_sequence)