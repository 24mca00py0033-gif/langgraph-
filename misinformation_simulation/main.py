"""
AI Multi-Agent Misinformation Simulation System
Main entry point for running the application.

Usage:
    python main.py          # Run Gradio UI
    python main.py --cli    # Run CLI simulation
"""

import argparse
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_ui(share: bool = False, port: int = 7860):
    """Run the Gradio UI."""
    from ui.gradio_app import launch_app
    print("🚀 Starting Misinformation Simulation UI...")
    print(f"   Open http://localhost:{port} in your browser")
    launch_app(share=share, server_port=port)


def run_cli(topic: str = None, num_nodes: int = 15):
    """Run a CLI-based simulation."""
    from orchestration.workflow import MisinformationWorkflow
    
    print("=" * 60)
    print("🔍 AI Multi-Agent Misinformation Simulation")
    print("=" * 60)
    
    print("\n📊 Initializing simulation...")
    workflow = MisinformationWorkflow(num_nodes=num_nodes)
    
    print("🤖 Running multi-agent workflow...")
    result = workflow.run_simulation(topic=topic)
    
    if result.get("error"):
        print(f"\n❌ Error: {result['error']}")
        return
    
    # Print results
    print(workflow.get_simulation_summary(result))
    
    # Show graph path
    graph_path = result.get("graph_image_path", "")
    if graph_path and os.path.exists(graph_path):
        print(f"\n📈 Network visualization saved to: {graph_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Multi-Agent Misinformation Simulation System"
    )
    
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in CLI mode instead of UI"
    )
    
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Topic for claim generation (CLI mode)"
    )
    
    parser.add_argument(
        "--nodes",
        type=int,
        default=15,
        help="Number of nodes in the social network"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for Gradio UI"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link for Gradio"
    )
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  Warning: GROQ_API_KEY not found in environment.")
        print("   Please set your API key in the .env file.")
        print("   Get your key at: https://console.groq.com/keys")
        return
    
    if args.cli:
        run_cli(topic=args.topic, num_nodes=args.nodes)
    else:
        run_ui(share=args.share, port=args.port)


if __name__ == "__main__":
    main()
