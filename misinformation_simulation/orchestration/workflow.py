"""
LangGraph Workflow Orchestration
Coordinates all agents in a sequential pipeline for misinformation simulation.
"""

import os
import sys
from typing import TypedDict, Optional, Annotated
from operator import add

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from agents.misinformation_agent import MisinformationAgent
from agents.neutral_agent import NeutralAgent
from agents.fact_checker_agent import FactCheckerAgent
from agents.influencer_agent import InfluencerAgent
from agents.moderator_agent import ModeratorAgent
from graph.social_network import SocialNetwork

load_dotenv()


class SimulationState(TypedDict):
    """State object passed between agents in the workflow."""
    # Claim information
    original_claim: str
    modified_claim: str
    topic: Optional[str]
    
    # Network information
    start_node: int
    spread_path: list
    nodes_reached: int
    network_stats: dict
    
    # Verification information
    verdict: str
    confidence: str
    verification_reasoning: str
    evidence_summary: str
    
    # Influencer information
    influencer_action: str
    influencer_content: str
    
    # Moderation information
    moderation_action: str
    moderation_reason: str
    spread_allowed: bool
    
    # Workflow tracking
    current_step: str
    steps_completed: list
    error: Optional[str]
    
    # Visualization
    graph_image_path: str


class MisinformationWorkflow:
    """
    Orchestrates the multi-agent misinformation simulation workflow.
    Uses LangGraph to coordinate agents in a sequential pipeline.
    """
    
    def __init__(self, num_nodes: int = 15, edges_per_node: int = 2):
        """
        Initialize the workflow with all agents and network.
        
        Args:
            num_nodes: Number of nodes in the social network
            edges_per_node: Edges per new node in Barabási-Albert model
        """
        # Initialize social network
        self.network = SocialNetwork(num_nodes=num_nodes, edges_per_new_node=edges_per_node)
        
        # Initialize agents
        self.misinformation_agent = MisinformationAgent()
        self.neutral_agent = NeutralAgent(self.network)
        self.fact_checker_agent = FactCheckerAgent()
        self.influencer_agent = InfluencerAgent()
        self.moderator_agent = ModeratorAgent(use_llm=True)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create the state graph
        workflow = StateGraph(SimulationState)
        
        # Add nodes (agents) to the graph
        workflow.add_node("generate_claim", self._generate_claim_node)
        workflow.add_node("spread_content", self._spread_content_node)
        workflow.add_node("verify_claim", self._verify_claim_node)
        workflow.add_node("process_influencer", self._process_influencer_node)
        workflow.add_node("moderate_content", self._moderate_content_node)
        workflow.add_node("visualize_network", self._visualize_network_node)
        
        # Set entry point
        workflow.set_entry_point("generate_claim")
        
        # Add edges (sequential flow)
        workflow.add_edge("generate_claim", "spread_content")
        workflow.add_edge("spread_content", "verify_claim")
        workflow.add_edge("verify_claim", "process_influencer")
        workflow.add_edge("process_influencer", "moderate_content")
        workflow.add_edge("moderate_content", "visualize_network")
        workflow.add_edge("visualize_network", END)
        
        return workflow.compile()
    
    def _generate_claim_node(self, state: SimulationState) -> dict:
        """Node: Generate misinformation claim."""
        topic = state.get("topic")
        claim = self.misinformation_agent.generate_claim(topic)
        
        return {
            "original_claim": claim,
            "current_step": "claim_generated",
            "steps_completed": ["generate_claim"]
        }
    
    def _spread_content_node(self, state: SimulationState) -> dict:
        """Node: Spread content through the network."""
        claim = state["original_claim"]
        
        # Spread through network
        spread_result = self.neutral_agent.spread_content(claim, max_hops=3)
        
        return {
            "start_node": spread_result["start_node"],
            "spread_path": spread_result["spread_path"],
            "nodes_reached": spread_result["nodes_reached"],
            "network_stats": spread_result["network_stats"],
            "current_step": "content_spread",
            "steps_completed": state["steps_completed"] + ["spread_content"]
        }
    
    def _verify_claim_node(self, state: SimulationState) -> dict:
        """Node: Verify the claim using fact-checker."""
        claim = state["original_claim"]
        
        # Verify claim
        verification = self.fact_checker_agent.verify_claim(claim)
        evidence = self.fact_checker_agent.get_evidence_summary(claim)
        
        return {
            "verdict": verification["verdict"],
            "confidence": verification["confidence"],
            "verification_reasoning": verification["reasoning"],
            "evidence_summary": evidence,
            "current_step": "claim_verified",
            "steps_completed": state["steps_completed"] + ["verify_claim"]
        }
    
    def _process_influencer_node(self, state: SimulationState) -> dict:
        """Node: Process claim through influencer agent."""
        claim = state["original_claim"]
        verdict = state["verdict"]
        reasoning = state["verification_reasoning"]
        
        # Process through influencer
        result = self.influencer_agent.process_claim(claim, verdict, reasoning)
        
        return {
            "influencer_action": result["action"],
            "influencer_content": result["modified_content"],
            "modified_claim": result["modified_content"],
            "current_step": "influencer_processed",
            "steps_completed": state["steps_completed"] + ["process_influencer"]
        }
    
    def _moderate_content_node(self, state: SimulationState) -> dict:
        """Node: Make moderation decision."""
        claim = state["original_claim"]
        verdict = state["verdict"]
        confidence = state["confidence"]
        reasoning = state["verification_reasoning"]
        
        # Make moderation decision
        decision = self.moderator_agent.moderate_content(
            claim, verdict, confidence, reasoning
        )
        
        return {
            "moderation_action": decision["action"],
            "moderation_reason": decision["reason"],
            "spread_allowed": decision["spread_allowed"],
            "current_step": "content_moderated",
            "steps_completed": state["steps_completed"] + ["moderate_content"]
        }
    
    def _visualize_network_node(self, state: SimulationState) -> dict:
        """Node: Generate network visualization."""
        verdict = state.get("verdict", "UNKNOWN")
        action = state.get("moderation_action", "UNKNOWN")
        
        title = f"Social Network - Verdict: {verdict} | Action: {action}"
        image_path = self.network.visualize(
            save_path="network_visualization.png",
            title=title
        )
        
        return {
            "graph_image_path": image_path,
            "current_step": "visualization_complete",
            "steps_completed": state["steps_completed"] + ["visualize_network"]
        }
    
    def run_simulation(self, topic: Optional[str] = None) -> SimulationState:
        """
        Run the complete simulation workflow.
        
        Args:
            topic: Optional topic for claim generation
            
        Returns:
            Final state containing all simulation results
        """
        # Initialize state
        initial_state: SimulationState = {
            "original_claim": "",
            "modified_claim": "",
            "topic": topic,
            "start_node": 0,
            "spread_path": [],
            "nodes_reached": 0,
            "network_stats": {},
            "verdict": "",
            "confidence": "",
            "verification_reasoning": "",
            "evidence_summary": "",
            "influencer_action": "",
            "influencer_content": "",
            "moderation_action": "",
            "moderation_reason": "",
            "spread_allowed": True,
            "current_step": "initialized",
            "steps_completed": [],
            "error": None,
            "graph_image_path": ""
        }
        
        try:
            # Run the workflow
            final_state = self.workflow.invoke(initial_state)
            return final_state
        except Exception as e:
            initial_state["error"] = str(e)
            return initial_state
    
    def get_simulation_summary(self, state: SimulationState) -> str:
        """
        Generate a human-readable summary of the simulation.
        
        Args:
            state: Final simulation state
            
        Returns:
            Formatted summary string
        """
        action_emoji = {
            "BLOCK": "🛑",
            "FLAG": "⚠️",
            "ALLOW": "✅"
        }
        verdict_emoji = {
            "REAL": "✅",
            "FAKE": "❌",
            "UNVERIFIED": "❓"
        }
        
        mod_emoji = action_emoji.get(state.get("moderation_action", ""), "❓")
        ver_emoji = verdict_emoji.get(state.get("verdict", ""), "❓")
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║           MISINFORMATION SIMULATION RESULTS                  ║
╠══════════════════════════════════════════════════════════════╣

📰 ORIGINAL CLAIM:
{state.get('original_claim', 'N/A')}

📢 INFLUENCER VERSION:
{state.get('influencer_content', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 SPREAD INFORMATION:
• Starting Node: {state.get('start_node', 'N/A')}
• Nodes Reached: {state.get('nodes_reached', 0)}
• Spread Path: {' → '.join(map(str, state.get('spread_path', [])))}
• Network Penetration: {state.get('network_stats', {}).get('penetration_rate', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{ver_emoji} VERIFICATION:
• Verdict: {state.get('verdict', 'N/A')}
• Confidence: {state.get('confidence', 'N/A')}
• Reasoning: {state.get('verification_reasoning', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{mod_emoji} MODERATION DECISION:
• Action: {state.get('moderation_action', 'N/A')}
• Reason: {state.get('moderation_reason', 'N/A')}
• Spread Allowed: {'Yes' if state.get('spread_allowed', False) else 'No'}

╚══════════════════════════════════════════════════════════════╝
"""
        return summary


if __name__ == "__main__":
    # Test the workflow
    workflow = MisinformationWorkflow(num_nodes=15)
    result = workflow.run_simulation(topic="technology")
    print(workflow.get_simulation_summary(result))
