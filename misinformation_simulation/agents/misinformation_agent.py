"""
Misinformation Agent
Generates realistic fake news and misleading claims using LLM.
"""

import os
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


class MisinformationAgent:
    """
    Agent responsible for generating realistic misinformation claims.
    Uses LLM to create plausible but false news-like claims.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the Misinformation Agent.
        
        Args:
            model_name: Name of the Groq model to use
        """
        self.model_name = model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.llm = ChatGroq(
            model=self.model_name,
            temperature=0.8,  # Higher temperature for creative generation
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        self.system_prompt = """You are a misinformation simulation agent for research purposes.
Your task is to generate a SHORT, realistic-looking news claim that could be either true or false.
The claim should be:
- Brief (1-2 sentences maximum)
- Sound like a real news headline or social media post
- Cover topics like technology, health, politics, science, or current events
- Plausible enough that someone might believe and share it

IMPORTANT: This is for academic research on misinformation spread. 
Generate ONLY the claim text, nothing else. No explanations or labels."""

    def generate_claim(self, topic: Optional[str] = None) -> str:
        """
        Generate a misinformation claim.
        
        Args:
            topic: Optional topic for the claim (e.g., "health", "technology")
            
        Returns:
            Generated claim text
        """
        if topic:
            user_prompt = f"Generate a short news-like claim about {topic} that may be real or fake."
        else:
            user_prompt = "Generate a short news-like claim that may be real or fake."
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content.strip()
    
    def generate_multiple_claims(self, count: int = 3, topic: Optional[str] = None) -> list:
        """
        Generate multiple misinformation claims.
        
        Args:
            count: Number of claims to generate
            topic: Optional topic for the claims
            
        Returns:
            List of generated claims
        """
        claims = []
        for _ in range(count):
            claim = self.generate_claim(topic)
            claims.append(claim)
        return claims


if __name__ == "__main__":
    agent = MisinformationAgent()
    claim = agent.generate_claim()
    print(f"Generated Claim: {claim}")
