"""
Neutral Agent
Simulates average social media users who share content without verification.
Handles the spreading of content through the social network using BFS.
"""

from typing import List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.social_network import SocialNetwork


class NeutralAgent:
    """
    Agent that simulates typical social media users.
    Spreads content through the network without verification.
    """
    
    def __init__(self, network: SocialNetwork):
        """
        Initialize the Neutral Agent.
        
        Args:
            network: The social network graph to spread through
        """
        self.network = network
        
    def spread_content(self, claim: str, start_node: Optional[int] = None, 
                       max_hops: int = 3) -> dict:
        """
        Spread content through the social network using BFS.
        
        Args:
            claim: The claim/content to spread
            start_node: Starting node for spread (random if None)
            max_hops: Maximum number of hops to spread
            
        Returns:
            Dictionary containing spread information
        """
        # Reset network before new spread
        self.network.reset_network()
        
        # Get starting node
        if start_node is None:
            start_node = self.network.get_random_start_node()
        
        # Perform BFS spread
        spread_path = self.network.spread_bfs(start_node, max_hops)
        
        # Get network statistics
        stats = self.network.get_network_stats()
        
        return {
            "claim": claim,
            "start_node": start_node,
            "spread_path": spread_path,
            "nodes_reached": len(spread_path),
            "network_stats": stats
        }
    
    def get_spread_summary(self) -> str:
        """
        Get a human-readable summary of the spread.
        
        Returns:
            Summary string
        """
        stats = self.network.get_network_stats()
        path = self.network.spread_path
        
        summary = f"""
📊 SPREAD SUMMARY
================
• Starting Node: {path[0] if path else 'N/A'}
• Total Nodes Reached: {stats['infected_nodes']}
• Network Penetration: {stats['penetration_rate']}
• Spread Path: {' → '.join(map(str, path))}
"""
        return summary


if __name__ == "__main__":
    # Test the neutral agent
    network = SocialNetwork(num_nodes=15)
    agent = NeutralAgent(network)
    
    result = agent.spread_content("Test claim for spreading", max_hops=3)
    print(f"Spread Result: {result}")
    print(agent.get_spread_summary())
