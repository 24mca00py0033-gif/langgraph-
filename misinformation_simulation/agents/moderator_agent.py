"""
Moderator Agent
Makes intelligent decisions to flag, block, or allow content based on verification results.
"""

import os
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


class ModeratorAgent:
    """
    Agent responsible for making moderation decisions.
    Decides whether to flag, block, or allow content based on verification.
    """
    
    def __init__(self, model_name: Optional[str] = None, use_llm: bool = True):
        """
        Initialize the Moderator Agent.
        
        Args:
            model_name: Name of the Groq model to use
            use_llm: If True, use LLM for nuanced decisions. If False, use rule-based logic.
        """
        self.use_llm = use_llm
        self.model_name = model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        
        if use_llm:
            self.llm = ChatGroq(
                model=self.model_name,
                temperature=0.2,  # Low temperature for consistent decisions
                api_key=os.getenv("GROQ_API_KEY")
            )
        
        self.system_prompt = """You are a content moderator for a social media platform.
Based on the claim and its verification verdict, decide what action to take.

Available actions:
1. BLOCK - Stop the content from spreading (for clearly fake/harmful content)
2. FLAG - Mark for review and warn users (for unverified or questionable content)
3. ALLOW - Let the content spread normally (for verified real content)

Consider:
- The severity of potential harm if the content spreads
- The verification confidence level
- Whether the content could cause panic, discrimination, or harm

Respond in this format:
ACTION: [BLOCK/FLAG/ALLOW]
REASON: [Brief explanation for your decision]"""

    def moderate_content(self, claim: str, verdict: str, confidence: str, 
                        reasoning: str = "") -> dict:
        """
        Make a moderation decision for the given content.
        
        Args:
            claim: The claim to moderate
            verdict: Verification verdict (REAL, FAKE, UNVERIFIED)
            confidence: Confidence level (LOW, MEDIUM, HIGH)
            reasoning: Reasoning from fact-checker
            
        Returns:
            Dictionary containing moderation decision and explanation
        """
        if self.use_llm:
            return self._llm_moderate(claim, verdict, confidence, reasoning)
        else:
            return self._rule_based_moderate(claim, verdict, confidence)
    
    def _rule_based_moderate(self, claim: str, verdict: str, confidence: str) -> dict:
        """
        Make moderation decision using rule-based logic.
        """
        if verdict == "FAKE":
            if confidence in ["HIGH", "MEDIUM"]:
                action = "BLOCK"
                reason = "Content verified as fake with sufficient confidence. Blocking to prevent spread."
            else:
                action = "FLAG"
                reason = "Content likely fake but low confidence. Flagging for review."
        elif verdict == "UNVERIFIED":
            action = "FLAG"
            reason = "Content cannot be verified. Flagging to warn users to verify before sharing."
        else:  # REAL
            action = "ALLOW"
            reason = "Content verified as real. Allowing normal spread."
        
        return {
            "claim": claim,
            "verdict": verdict,
            "confidence": confidence,
            "action": action,
            "reason": reason,
            "spread_allowed": action == "ALLOW"
        }
    
    def _llm_moderate(self, claim: str, verdict: str, confidence: str, 
                     reasoning: str) -> dict:
        """
        Make moderation decision using LLM reasoning.
        """
        context = f"""
Claim: {claim}
Verification Verdict: {verdict}
Confidence: {confidence}
Fact-Checker Reasoning: {reasoning}
"""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Make a moderation decision for:\n{context}")
        ]
        
        response = self.llm.invoke(messages)
        response_text = response.content.strip()
        
        # Parse response
        action = "FLAG"  # Default to FLAG for safety
        reason = response_text
        
        lines = response_text.split('\n')
        for line in lines:
            line_upper = line.upper()
            if 'ACTION:' in line_upper:
                action_text = line.split(':', 1)[1].strip().upper()
                if 'BLOCK' in action_text:
                    action = 'BLOCK'
                elif 'ALLOW' in action_text:
                    action = 'ALLOW'
                else:
                    action = 'FLAG'
            elif 'REASON:' in line_upper:
                reason = line.split(':', 1)[1].strip()
        
        return {
            "claim": claim,
            "verdict": verdict,
            "confidence": confidence,
            "action": action,
            "reason": reason,
            "spread_allowed": action == "ALLOW",
            "raw_response": response_text
        }
    
    def get_moderation_summary(self, decision: dict) -> str:
        """
        Get a human-readable summary of the moderation decision.
        
        Args:
            decision: Moderation decision dictionary
            
        Returns:
            Formatted summary string
        """
        action_emoji = {
            "BLOCK": "🛑",
            "FLAG": "⚠️",
            "ALLOW": "✅"
        }
        
        emoji = action_emoji.get(decision["action"], "❓")
        
        return f"""
{emoji} MODERATION DECISION
======================
Action: {decision['action']}
Verdict: {decision['verdict']}
Confidence: {decision['confidence']}
Reason: {decision['reason']}
Spread Allowed: {'Yes' if decision['spread_allowed'] else 'No'}
"""


if __name__ == "__main__":
    # Test with rule-based moderation
    agent = ModeratorAgent(use_llm=False)
    
    decision = agent.moderate_content(
        claim="Breaking: Government secretly tracking all citizens",
        verdict="FAKE",
        confidence="HIGH",
        reasoning="No evidence supports this claim"
    )
    
    print(agent.get_moderation_summary(decision))
