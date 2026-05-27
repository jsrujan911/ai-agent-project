"""
Phase 2B: Basic LLM App with Local Ollama (Free Alternative to OpenAI)

This uses Mistral (or another model) running locally via Ollama.
- No API key needed
- No monthly costs
- Runs on your machine
"""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Initialize the local LLM (server must be running: ollama serve)
llm = OllamaLLM(
    model="mistral",
    base_url="http://127.0.0.1:11434",  # Local server
    temperature=0.2
)

print("=" * 70)
print("Phase 2B: Basic LLM with Local Ollama")
print("=" * 70)

# ============================================================================
# 1. Simple LLM call
# ============================================================================
print("\n1. Simple LLM Invocation")
print("-" * 70)

prompt = "Explain about sdlc in simple terms with a short example."
response = llm.invoke(prompt)

print(f"Prompt: {prompt}\n")
print(f"Response:\n{response}\n")

# ============================================================================
# 2. Using PromptTemplate (Phase 3 review)
# ============================================================================
print("\n2. Using PromptTemplate")
print("-" * 70)

template = PromptTemplate(
    input_variables=["topic", "audience"],
    template="""You are a helpful AI tutor.
    
Topic: {topic}
Audience: {audience}

Explain the topic in a way that is appropriate for the audience. Be concise (max 3 sentences)."""
)

formatted_prompt = template.format(
    topic="Vector Databases",
    audience="software engineers new to AI"
)

response2 = llm.invoke(formatted_prompt)

print(f"Template Prompt:\n{formatted_prompt}\n")
print(f"Response:\n{response2}\n")

# ============================================================================
# 3. Quick comparison: different styles
# ============================================================================
print("\n3. Prompt Style Comparison")
print("-" * 70)

styles = {
    "Technical": "Explain embeddings for AI engineers",
    "Beginner": "Explain embeddings for someone with no AI knowledge",
    "Poetic": "Describe embeddings in a poetic way"
}

for style_name, style_prompt in styles.items():
    print(f"\n{style_name}: {style_prompt}")
    resp = llm.invoke(style_prompt)
    # Truncate long responses for display
    resp_preview = resp[:200] + "..." if len(resp) > 200 else resp
    print(f"→ {resp_preview}\n")

print("=" * 70)
print("✅ Phase 2B Complete: Local LLM works!")
print("=" * 70)
print("""
Next Steps:
- Phase 3: More prompt engineering techniques
- Phase 4: Build RAG system (document loading + vector DB + retrieval)
- Phase 5: Create an AI Agent with tools
""")
