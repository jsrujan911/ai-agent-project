"""
Mock LLM for testing when API quota is exhausted.
Simulates ChatOpenAI responses for learning purposes.
"""

from dotenv import load_dotenv
import os

load_dotenv()

# Mock LLM class that simulates ChatOpenAI behavior
class MockChatOpenAI:
    def __init__(self, model="gpt-4o-mini", temperature=0.2):
        self.model = model
        self.temperature = temperature
    
    def invoke(self, prompt):
        """Simulate LLM response"""
        responses = {
            "rag": "Retrieval-Augmented Generation (RAG) is a technique that combines LLMs with retrieval. Instead of relying only on the model's training data, RAG retrieves relevant documents from a knowledge base and uses them to augment the prompt. This improves accuracy and allows the LLM to use up-to-date information. Example: A customer support chatbot that retrieves relevant help articles before generating a response.",
            "vector": "A vector database stores data as high-dimensional vectors (embeddings). Embeddings represent text, images, or other data as numerical arrays. Vector databases excel at similarity search—finding items most similar to a query. They're crucial for RAG systems because they retrieve relevant documents by computing similarity between the query embedding and stored embeddings.",
            "agent": "An AI Agent is a system that autonomously decides what actions to take based on observations and goals. Agents use LLMs to reason about situations and choose from available tools. The agent loop is: Thought → Action → Observation → (repeat). Tools might be calculators, APIs, or databases. Agents enable complex multi-step reasoning.",
            "memory": "Conversation memory allows agents to remember past exchanges. Without memory, each LLM call is independent. With memory, the system maintains context: system-level facts, conversation history, or both. This enables coherent multi-turn conversations where the agent references prior messages.",
        }
        
        # Find matching response
        prompt_lower = prompt.lower()
        for key, response_text in responses.items():
            if key in prompt_lower:
                return MockMessage(response_text)
        
        # Default response
        return MockMessage("This is a simulated response for testing. Please fix your OpenAI API key to get real responses.")

class MockMessage:
    def __init__(self, content):
        self.content = content

# Use mock LLM
llm = MockChatOpenAI(model="gpt-4o-mini", temperature=0.2)

prompt = "Explain Retrieval-Augmented Generation (RAG) in simple terms with a short example."
response = llm.invoke(prompt)

print("Response (MOCK - fix API key for real responses):\n", response.content)
