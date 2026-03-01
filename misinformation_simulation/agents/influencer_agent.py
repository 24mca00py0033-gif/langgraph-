"""
Influencer Agent
Uses advanced prompt engineering to rewrite content for maximum viral spread.
Can also create warning messages for fake content.
"""

import os
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


class InfluencerAgent:
    """
    Agent that rewrites content to maximize viral potential.
    Can either amplify real content or create warnings for fake content.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the Influencer Agent.
        
        Args:
            model_name: Name of the Groq model to use
        """
        self.model_name = model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.llm = ChatGroq(
            model=self.model_name,
            temperature=0.7,
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        self.amplify_prompt = """You are a social media influencer skilled at making content go viral.
Your task is to rewrite the given content to maximize engagement and sharing.
Use these techniques:
- Emotional hooks and compelling language
- Clear call-to-action
- Relatable framing
- Urgency or importance indicators

Keep it concise (2-3 sentences max) and suitable for social media.
Output ONLY the rewritten content, nothing else."""

        self.warning_prompt = """You are a responsible social media influencer helping combat misinformation.
The following claim has been identified as FAKE or UNVERIFIED.
Your task is to create a WARNING message that:
- Clearly states this is misinformation
- Urges people NOT to share
- Provides a brief reason why it's problematic
- Encourages fact-checking before sharing

Keep it concise (2-3 sentences) and impactful.
Output ONLY the warning message, nothing else."""

    def amplify_content(self, claim: str) -> str:
        """
        Rewrite content to maximize viral potential (for real/verified content).
        
        Args:
            claim: The original claim to amplify
            
        Returns:
            Amplified version of the content
        """
        messages = [
            SystemMessage(content=self.amplify_prompt),
            HumanMessage(content=f"Rewrite this for maximum engagement: {claim}")
        ]
        
        response = self.llm.invoke(messages)
        return response.content.strip()
    
    def create_warning(self, claim: str, reasoning: str = "") -> str:
        """
        Create a warning message for fake/unverified content.
        
        Args:
            claim: The fake/unverified claim
            reasoning: Reason why it's fake (from fact-checker)
            
        Returns:
            Warning message
        """
        context = f"Claim: {claim}"
        if reasoning:
            context += f"\nReason it's problematic: {reasoning}"
            
        messages = [
            SystemMessage(content=self.warning_prompt),
            HumanMessage(content=context)
        ]
        
        response = self.llm.invoke(messages)
        return response.content.strip()
    
    def process_claim(self, claim: str, verdict: str, reasoning: str = "") -> dict:
        """
        Process a claim based on its verification verdict.
        
        Args:
            claim: The original claim
            verdict: Verification verdict (REAL, FAKE, UNVERIFIED)
            reasoning: Reasoning from fact-checker
            
        Returns:
            Dictionary with processed content and action taken
        """
        if verdict == "REAL":
            modified_content = self.amplify_content(claim)
            action = "AMPLIFIED"
        else:  # FAKE or UNVERIFIED
            modified_content = self.create_warning(claim, reasoning)
            action = "WARNING_CREATED"
        
        return {
            "original_claim": claim,
            "verdict": verdict,
            "action": action,
            "modified_content": modified_content
        }


if __name__ == "__main__":
    agent = InfluencerAgent()
    
    # Test amplification
    real_claim = "New study shows regular exercise can improve mental health by 40%"
    amplified = agent.amplify_content(real_claim)
    print(f"Amplified: {amplified}")
    
    # Test warning
    fake_claim = "Government secretly adding tracking chips to vaccines"
    warning = agent.create_warning(fake_claim, "No evidence supports this claim")
    print(f"Warning: {warning}")
