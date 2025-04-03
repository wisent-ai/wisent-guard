import os
from anthropic import Anthropic
 
 # Function to call Claude and evaluate the response
 
def call_claude_and_evaluate(actual_response: str, good_response: str, bad_response: str) -> int:
     """Evaluate if a response resembles a good response more than a bad response.
     
     Args:
         actual_response: The response to evaluate
         good_response: Example of a good response
         bad_response: Example of a bad response
         
     Returns:
         int: 1 if the actual response resembles the good response more,
              0 if it resembles the bad response more,
              -1 if the evaluation is ambiguous
     """
     # Initialize Anthropic client
     anthropic = Anthropic(
         api_key=os.environ.get("ANTHROPIC_API_KEY")
     )
     
     # Format the prompt for evaluation
     prompt = f"""
     I'm going to show you a question with three responses: an actual response, a good response example, and a bad response example.
     
     Your task is to determine whether the actual response is MORE SIMILAR to the good response or the bad response.
     
     Actual response:
     ---
     {actual_response}
     ---
     
     Good response example:
     ---
     {good_response}
     ---
     
     Bad response example:
     ---
     {bad_response}
     ---
     
     Is the actual response more similar to the good response than the bad response?
     Please respond with ONLY '1' if the actual response is more similar to the good response,
     or ONLY '0' if the actual response is more similar to the bad response.
     """
     
     # Call Claude API
     response = anthropic.messages.create(
         model="claude-3-sonnet-20240229",
         max_tokens=1024,
         messages=[{
             "role": "user",
             "content": prompt
         }]
     )
     
     # Get the response text
     response_text = response.content[0].text
     
     # Parse the response
     if response_text.strip() == "0":
         return 0
     elif response_text.strip() == "1":
         return 1
     elif "0" in response_text and "1" in response_text:
         return -1
     else:
         return -1