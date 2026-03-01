"""
Gradio UI for Misinformation Simulation
Interactive web interface for running and visualizing simulations.
"""

import os
import sys
import gradio as gr

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestration.workflow import MisinformationWorkflow


class SimulationUI:
    """
    Gradio-based UI for the misinformation simulation system.
    """
    
    def __init__(self):
        """Initialize the UI with workflow instance."""
        self.workflow = None
        self.last_state = None
    
    def run_simulation(self, num_nodes: int, edges_per_node: int, 
                       topic: str, progress=gr.Progress()) -> tuple:
        """
        Run the simulation and return results for the UI.
        
        Args:
            num_nodes: Number of nodes in network
            edges_per_node: Edges per new node
            topic: Topic for claim generation
            progress: Gradio progress indicator
            
        Returns:
            Tuple of (original_claim, influencer_content, spread_path, 
                     verdict, moderation_decision, evidence, graph_image)
        """
        progress(0, desc="Initializing simulation...")
        
        # Initialize workflow with user parameters
        self.workflow = MisinformationWorkflow(
            num_nodes=int(num_nodes),
            edges_per_node=int(edges_per_node)
        )
        
        progress(0.2, desc="Running multi-agent simulation...")
        
        # Run simulation
        topic_str = topic.strip() if topic.strip() else None
        state = self.workflow.run_simulation(topic=topic_str)
        self.last_state = state
        
        progress(0.8, desc="Generating results...")
        
        # Check for errors
        if state.get("error"):
            error_msg = f"Error: {state['error']}"
            return error_msg, "", "", "", "", "", None
        
        # Extract results
        original_claim = state.get("original_claim", "N/A")
        influencer_content = state.get("influencer_content", "N/A")
        
        # Format spread path
        spread_path = state.get("spread_path", [])
        spread_path_str = " → ".join(map(str, spread_path)) if spread_path else "N/A"
        nodes_reached = state.get("nodes_reached", 0)
        stats = state.get("network_stats", {})
        penetration = stats.get("penetration_rate", "N/A")
        
        spread_info = f"""
**Starting Node:** {state.get('start_node', 'N/A')}
**Nodes Reached:** {nodes_reached}
**Network Penetration:** {penetration}
**Spread Path:** {spread_path_str}
"""
        
        # Format verdict
        verdict = state.get("verdict", "UNKNOWN")
        confidence = state.get("confidence", "UNKNOWN")
        reasoning = state.get("verification_reasoning", "")
        
        verdict_emoji = {"REAL": "✅", "FAKE": "❌", "UNVERIFIED": "❓"}
        emoji = verdict_emoji.get(verdict, "❓")
        
        verdict_info = f"""
{emoji} **Verdict:** {verdict}
**Confidence:** {confidence}
**Reasoning:** {reasoning}
"""
        
        # Format moderation decision
        mod_action = state.get("moderation_action", "UNKNOWN")
        mod_reason = state.get("moderation_reason", "")
        spread_allowed = state.get("spread_allowed", False)
        
        action_emoji = {"BLOCK": "🛑", "FLAG": "⚠️", "ALLOW": "✅"}
        mod_emoji = action_emoji.get(mod_action, "❓")
        
        moderation_info = f"""
{mod_emoji} **Action:** {mod_action}
**Reason:** {mod_reason}
**Spread Allowed:** {'Yes ✅' if spread_allowed else 'No ❌'}
"""
        
        # Evidence summary
        evidence = state.get("evidence_summary", "No evidence collected.")
        
        # Graph image
        graph_path = state.get("graph_image_path", "")
        if graph_path and os.path.exists(graph_path):
            graph_image = graph_path
        else:
            graph_image = None
        
        progress(1.0, desc="Complete!")
        
        return (
            original_claim,
            influencer_content,
            spread_info,
            verdict_info,
            moderation_info,
            evidence,
            graph_image
        )
    
    def get_full_report(self) -> str:
        """Get the full simulation report."""
        if self.last_state and self.workflow:
            return self.workflow.get_simulation_summary(self.last_state)
        return "No simulation has been run yet. Click 'Run Simulation' first."


def create_app() -> gr.Blocks:
    """
    Create and configure the Gradio application.
    
    Returns:
        Configured Gradio Blocks application
    """
    ui = SimulationUI()
    
    # Custom CSS for styling
    custom_css = """
    .main-title {
        text-align: center;
        color: #2c3e50;
        font-size: 2em;
        margin-bottom: 0.5em;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 1em;
    }
    .result-box {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
    """
    
    with gr.Blocks(
        title="Misinformation Simulation System",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as app:
        
        # Header
        gr.Markdown(
            """
            # 🔍 AI Multi-Agent Misinformation Simulation System
            ### A Graph-Based Multi-Agent Platform for Social Network Information Dynamics
            
            This system simulates how misinformation spreads through social networks using 
            five AI agents: **Misinformation Generator**, **Neutral Spreader**, **Fact-Checker**, 
            **Influencer**, and **Moderator**.
            """
        )
        
        with gr.Row():
            # Left Column - Configuration
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")
                
                num_nodes = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=15,
                    step=1,
                    label="Number of Nodes (Users)",
                    info="Size of the social network"
                )
                
                edges_per_node = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1,
                    label="Edges per Node",
                    info="Connection density"
                )
                
                topic = gr.Textbox(
                    label="Topic (Optional)",
                    placeholder="e.g., health, technology, politics",
                    info="Leave empty for random topic"
                )
                
                run_btn = gr.Button(
                    "🚀 Run Simulation",
                    variant="primary",
                    size="lg"
                )
                
                report_btn = gr.Button(
                    "📋 Get Full Report",
                    variant="secondary"
                )
            
            # Right Column - Network Visualization
            with gr.Column(scale=2):
                gr.Markdown("### 🌐 Social Network Visualization")
                graph_output = gr.Image(
                    label="Network Graph",
                    type="filepath",
                    height=400
                )
        
        gr.Markdown("---")
        
        # Results Section
        gr.Markdown("### 📊 Simulation Results")
        
        with gr.Row():
            with gr.Column():
                original_claim = gr.Textbox(
                    label="📰 Original Claim (Misinformation Agent)",
                    lines=3,
                    interactive=False
                )
            
            with gr.Column():
                influencer_content = gr.Textbox(
                    label="📢 Influencer Modified Version",
                    lines=3,
                    interactive=False
                )
        
        with gr.Row():
            with gr.Column():
                spread_info = gr.Markdown(
                    label="📊 Spread Information"
                )
            
            with gr.Column():
                verdict_info = gr.Markdown(
                    label="🔍 Verification Verdict"
                )
            
            with gr.Column():
                moderation_info = gr.Markdown(
                    label="⚖️ Moderation Decision"
                )
        
        with gr.Accordion("🔍 Evidence Summary", open=False):
            evidence_output = gr.Textbox(
                label="Evidence Details",
                lines=4,
                interactive=False
            )
        
        with gr.Accordion("📋 Full Simulation Report", open=False):
            full_report = gr.Textbox(
                label="Complete Report",
                lines=20,
                interactive=False
            )
        
        # Footer
        gr.Markdown(
            """
            ---
            **MCA Final Year Project** | AI Multi-Agent System for Misinformation Spread, 
            Verification and Moderation in Social Networks
            """
        )
        
        # Event handlers
        run_btn.click(
            fn=ui.run_simulation,
            inputs=[num_nodes, edges_per_node, topic],
            outputs=[
                original_claim,
                influencer_content,
                spread_info,
                verdict_info,
                moderation_info,
                evidence_output,
                graph_output
            ],
            show_progress=True
        )
        
        report_btn.click(
            fn=ui.get_full_report,
            inputs=[],
            outputs=[full_report]
        )
    
    return app


def launch_app(share: bool = False, server_port: int = 7860):
    """
    Launch the Gradio application.
    
    Args:
        share: Whether to create a public link
        server_port: Port to run the server on
    """
    app = create_app()
    app.launch(
        share=share,
        server_port=server_port,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    launch_app()
