"""
Fact-Checker Agent
Verifies claims by searching for evidence and evaluating truthfulness.
Uses LLM with function/tool calling capabilities.
"""

import os
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()


@tool
def web_search(query: str) -> str:
    """
    Simulate a web search for fact-checking.
    In a real implementation, this would call a search API.
    
    Args:
        query: Search query for fact-checking
        
    Returns:
        Simulated search results
    """
    # Simulated search results for demonstration
    return f"Simulated search results for: '{query}'. Multiple sources found with varying claims. Further verification needed."


@tool
def check_official_sources(claim: str) -> str:
    """
    Check official government or institutional sources.
    
    Args:
        claim: The claim to verify against official sources
        
    Returns:
        Simulated official source check results
    """
    return f"Official source check for: '{claim}'. No official confirmation found in government or institutional databases."


class FactCheckerAgent:
    """
    Agent responsible for verifying claims.
    Uses LLM reasoning and tool calling for verification.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the Fact-Checker Agent.
        
        Args:
            model_name: Name of the Groq model to use
        """
        self.model_name = model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.llm = ChatGroq(
            model=self.model_name,
            temperature=0.1,  # Lower temperature for analytical reasoning
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Bind tools to the LLM
        self.tools = [web_search, check_official_sources]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        self.system_prompt = """You are a fact-checking agent specialized in verifying claims.
Your task is to analyze the given claim and determine its truthfulness.

Analyze the claim carefully and provide:
1. VERDICT: Must be one of: "REAL", "FAKE", or "UNVERIFIED"
2. CONFIDENCE: A confidence level (LOW, MEDIUM, HIGH)
3. REASONING: A brief explanation for your verdict

Format your response EXACTLY as:
VERDICT: [REAL/FAKE/UNVERIFIED]
CONFIDENCE: [LOW/MEDIUM/HIGH]
REASONING: [Your brief explanation]

Be critical and skeptical. Look for signs of misinformation such as:
- Sensationalist language
- Lack of specific sources
- Claims that seem too good/bad to be true
- Missing verifiable details"""

    def verify_claim(self, claim: str) -> dict:
        """
        Verify a claim and return the verdict.
        
        Args:
            claim: The claim to verify
            
        Returns:
            Dictionary containing verdict, confidence, and reasoning
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Verify this claim: {claim}")
        ]
        
        response = self.llm.invoke(messages)
        response_text = response.content.strip()
        
        # Parse the response
        verdict = "UNVERIFIED"
        confidence = "LOW"
        reasoning = response_text
        
        lines = response_text.split('\n')
        for line in lines:
            line_upper = line.upper()
            if 'VERDICT:' in line_upper:
                verdict_text = line.split(':', 1)[1].strip().upper()
                if 'REAL' in verdict_text:
                    verdict = 'REAL'
                elif 'FAKE' in verdict_text:
                    verdict = 'FAKE'
                else:
                    verdict = 'UNVERIFIED'
            elif 'CONFIDENCE:' in line_upper:
                conf_text = line.split(':', 1)[1].strip().upper()
                if 'HIGH' in conf_text:
                    confidence = 'HIGH'
                elif 'MEDIUM' in conf_text:
                    confidence = 'MEDIUM'
                else:
                    confidence = 'LOW'
            elif 'REASONING:' in line_upper:
                reasoning = line.split(':', 1)[1].strip()
        
        return {
            "claim": claim,
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": reasoning,
            "raw_response": response_text
        }
    
    def get_evidence_summary(self, claim: str) -> str:
        """
        Get a summary of evidence found for the claim.
        
        Args:
            claim: The claim to find evidence for
            
        Returns:
            Evidence summary string
        """
        # Simulate evidence gathering
        web_result = web_search.invoke({"query": claim})
        official_result = check_official_sources.invoke({"claim": claim})
        
        return f"""
🔍 EVIDENCE SUMMARY
==================
Web Search: {web_result}
Official Sources: {official_result}
"""


if __name__ == "__main__":
    agent = FactCheckerAgent()
    result = agent.verify_claim("Scientists discover new planet made entirely of diamonds")
    print(f"Verification Result: {result}")
