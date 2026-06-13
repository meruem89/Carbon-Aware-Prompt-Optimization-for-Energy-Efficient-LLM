import re

class TokenHealingLayer:
    def __init__(self):
        # High-precision patterns for code snippets, math constructs, and system variables
        self.protected_patterns = [
            r"https?://\S+",                 # URLs
            r"[\w\.-]+@[\w\.-]+\.\w+",       # Emails
            r"[{}[\]()\"]",                  # Structural JSON/AST Brackets & Quotes
            r"==|!=|<=|>=|=>|->|\+=|-=",     # Programming Operators & Arrow Functions
            r"def\s+\w+|class\s+\w+",        # Python function/class declarations
            r"<\w+[^>]*>.*?<\/\w+>",         # HTML/XML Inline Tags
            r"\$\d+(?:\.\d{2})?"             # Currency Metrics
        ]

    def extract_force_tokens(self, text: str) -> list:
        """
        Scans input prose for architectural primitives and code syntax,
        compiling a deduplicated list of explicit tokens that MUST survive compression.
        """
        force_set = set(["\n", ".", "?", "!", ",", ":"]) # Global structural baselines
        
        for pattern in self.protected_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Add the literal text string to the force-retention array
                force_set.add(str(match))
                
        return list(force_set)

if __name__ == "__main__":
    # Rapid Unit Verification
    layer = TokenHealingLayer()
    sample_code_prompt = "Fix this endpoint: def process_data(url='https://api.com'): return {'status': 200} if x <= 10 else None"
    tokens_to_heal = layer.extract_force_tokens(sample_code_prompt)
    print("--- Token Healing Verification ---")
    print(f"Detected Protected Structural Tokens:\n{tokens_to_heal}")