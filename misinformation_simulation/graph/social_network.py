"""
Social Network Graph Module
Creates and manages the social network graph using NetworkX.
Uses Barabási-Albert preferential attachment model for realistic topology.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving images
import random
from typing import List, Tuple, Optional
from collections import deque


class SocialNetwork:
    """
    Social Network Graph using Barabási-Albert model.
    Simulates a realistic social network with preferential attachment.
    """
    
    def __init__(self, num_nodes: int = 15, edges_per_new_node: int = 2):
        """
        Initialize the social network graph.
        
        Args:
            num_nodes: Number of nodes (users) in the network
            edges_per_new_node: Number of edges to attach from a new node to existing nodes
        """
        self.num_nodes = num_nodes
        self.edges_per_new_node = edges_per_new_node
        self.graph = self._create_graph()
        self.node_states = {node: "uninfected" for node in self.graph.nodes()}
        self.spread_path = []
        
    def _create_graph(self) -> nx.Graph:
        """Create a Barabási-Albert graph."""
        return nx.barabasi_albert_graph(self.num_nodes, self.edges_per_new_node)
    
    def reset_network(self):
        """Reset all node states and spread path."""
        self.node_states = {node: "uninfected" for node in self.graph.nodes()}
        self.spread_path = []
    
    def get_random_start_node(self) -> int:
        """Get a random node to start spreading from."""
        return random.choice(list(self.graph.nodes()))
    
    def get_high_degree_node(self) -> int:
        """Get a high-degree node (potential influencer)."""
        degrees = dict(self.graph.degree())
        return max(degrees, key=degrees.get)
    
    def spread_bfs(self, start_node: int, max_hops: int = 3) -> List[int]:
        """
        Spread information using Breadth-First Search.
        
        Args:
            start_node: Node to start spreading from
            max_hops: Maximum number of hops (levels) to spread
            
        Returns:
            List of nodes reached during spread
        """
        visited = set()
        queue = deque([(start_node, 0)])  # (node, current_hop)
        spread_path = []
        
        while queue:
            current_node, current_hop = queue.popleft()
            
            if current_node in visited or current_hop > max_hops:
                continue
                
            visited.add(current_node)
            spread_path.append(current_node)
            self.node_states[current_node] = "infected"
            
            # Add neighbors to queue
            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in visited:
                    queue.append((neighbor, current_hop + 1))
        
        self.spread_path = spread_path
        return spread_path
    
    def get_network_stats(self) -> dict:
        """Get basic network statistics."""
        infected_count = sum(1 for state in self.node_states.values() if state == "infected")
        return {
            "total_nodes": self.num_nodes,
            "total_edges": self.graph.number_of_edges(),
            "infected_nodes": infected_count,
            "uninfected_nodes": self.num_nodes - infected_count,
            "penetration_rate": f"{(infected_count / self.num_nodes) * 100:.1f}%",
            "spread_path_length": len(self.spread_path)
        }
    
    def visualize(self, save_path: str = "network_graph.png", title: str = "Social Network - Misinformation Spread") -> str:
        """
        Visualize the social network with colored nodes based on infection status.
        
        Args:
            save_path: Path to save the visualization
            title: Title for the graph
            
        Returns:
            Path to the saved image
        """
        plt.figure(figsize=(12, 8))
        
        # Define node colors based on state
        color_map = []
        for node in self.graph.nodes():
            if self.node_states[node] == "infected":
                color_map.append("#FF6B6B")  # Red for infected
            else:
                color_map.append("#4ECDC4")  # Teal for uninfected
        
        # Calculate node sizes based on degree (influence)
        degrees = dict(self.graph.degree())
        node_sizes = [300 + degrees[node] * 100 for node in self.graph.nodes()]
        
        # Create layout
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Draw the network
        nx.draw_networkx_edges(self.graph, pos, alpha=0.3, edge_color='gray')
        nx.draw_networkx_nodes(self.graph, pos, node_color=color_map, 
                               node_size=node_sizes, alpha=0.9)
        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_weight='bold')
        
        # Highlight spread path with edges
        if len(self.spread_path) > 1:
            spread_edges = [(self.spread_path[i], self.spread_path[i+1]) 
                           for i in range(len(self.spread_path)-1) 
                           if self.graph.has_edge(self.spread_path[i], self.spread_path[i+1])]
            nx.draw_networkx_edges(self.graph, pos, edgelist=spread_edges, 
                                   edge_color='#FF6B6B', width=2, alpha=0.8)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', 
                      markersize=15, label='Infected (Exposed to Misinformation)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4', 
                      markersize=15, label='Uninfected (Not Exposed)')
        ]
        plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return save_path
    
    def get_node_info(self, node: int) -> dict:
        """Get information about a specific node."""
        return {
            "node_id": node,
            "state": self.node_states[node],
            "degree": self.graph.degree(node),
            "neighbors": list(self.graph.neighbors(node))
        }


if __name__ == "__main__":
    # Test the social network
    network = SocialNetwork(num_nodes=15, edges_per_new_node=2)
    start = network.get_random_start_node()
    print(f"Starting spread from node: {start}")
    
    path = network.spread_bfs(start, max_hops=3)
    print(f"Spread path: {path}")
    print(f"Network stats: {network.get_network_stats()}")
    
    network.visualize("test_network.png")
    print("Network visualization saved to test_network.png")
