"""
PHASE 3: Prompt Engineering - Master the Art of Structured Prompts

Key Concepts:
- PromptTemplate: reusable prompt patterns with variables
- Structured prompting: role-based system messages + clear instructions
- Best practices: context, examples, constraints, output format
- Comparison: Bad prompt vs. Good prompt vs. Expert prompt
"""

from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ============================================================================
# 1. BASIC PROMPT TEMPLATE - Simple variable substitution
# ============================================================================
print("=" * 70)
print("1. BASIC PROMPT TEMPLATE - Variable Substitution")
print("=" * 70)

simple_template = PromptTemplate(
    input_variables=["topic", "style"],
    template="""Explain {topic} in a {style} manner."""
)

# Use the template
prompt1 = simple_template.format(topic="Machine Learning", style="beginner-friendly")
print("\nSimple Prompt:")
print(prompt1)
print()

# ============================================================================
# 2. MULTI-VARIABLE PROMPT - More complex template
# ============================================================================
print("=" * 70)
print("2. MULTI-VARIABLE PROMPT - Task + Context + Example")
print("=" * 70)

multi_template = PromptTemplate(
    input_variables=["task", "context", "output_format"],
    template="""
Task: {task}

Context: {context}

Output Format: {output_format}

Please complete the task based on the context provided.
""".strip()
)

prompt2 = multi_template.format(
    task="Summarize the key points",
    context="Vector databases store high-dimensional data and enable fast similarity search.",
    output_format="Bullet points"
)
print("\nMulti-Variable Prompt:")
print(prompt2)
print()

# ============================================================================
# 3. BEST PRACTICE: STRUCTURED PROMPT - Role + Task + Context + Examples
# ============================================================================
print("=" * 70)
print("3. STRUCTURED PROMPT (BEST PRACTICE)")
print("=" * 70)

structured_template = PromptTemplate(
    input_variables=["role", "task", "context", "constraint", "examples"],
    template="""You are {role}.

Your Task: {task}

Context: {context}

Constraint: {constraint}

Examples:
{examples}

Now complete the task accurately and concisely."""
)

prompt3 = structured_template.format(
    role="an expert AI tutor",
    task="Explain a technical concept in simple terms",
    context="The audience is a beginner with no technical background",
    constraint="Keep the explanation under 5 sentences",
    examples="""
Example 1:
Q: What is an API?
A: An API is like a waiter in a restaurant. You tell the waiter what you want (your request), the waiter goes to the kitchen (the server), and brings back your food (the response).

Example 2:
Q: What is a database?
A: A database is like a digital filing cabinet. Instead of storing papers, it stores organized data. You can quickly find, add, or update information.
""".strip()
)

print("\nStructured Prompt (Expert Level):")
print(prompt3)
print()

# ============================================================================
# 4. COMPARISON: BAD vs GOOD vs EXPERT PROMPTS
# ============================================================================
print("=" * 70)
print("4. COMPARISON: PROMPT QUALITY MATTERS")
print("=" * 70)

print("\n❌ BAD PROMPT (vague, unclear):")
bad_prompt = "Explain RAG"
print(f'"{bad_prompt}"')
print("   → Result: Generic, unfocused response\n")

print("✅ GOOD PROMPT (clear, specific):")
good_prompt_template = PromptTemplate(
    input_variables=["concept"],
    template="Explain {concept} step-by-step with a real-world example."
)
good_prompt = good_prompt_template.format(concept="RAG (Retrieval-Augmented Generation)")
print(f'"{good_prompt}"')
print("   → Result: Structured, example-driven response\n")

print("🌟 EXPERT PROMPT (structured, detailed, constraints):")
expert_template = PromptTemplate(
    input_variables=["concept", "audience", "max_sentences"],
    template="""You are a technical writer explaining complex concepts to {audience}.

Concept: {concept}

Instructions:
1. Define the concept in one simple sentence
2. Explain how it works (step-by-step)
3. Provide a real-world example that {audience} can relate to
4. List 2-3 key benefits

Constraints:
- Use simple language
- Max {max_sentences} sentences per section
- Avoid jargon; if you must use technical terms, define them
- Make it memorable and relatable

Response:""".strip()
)

expert_prompt = expert_template.format(
    concept="RAG (Retrieval-Augmented Generation)",
    audience="beginners with no AI background",
    max_sentences="3"
)
print(expert_prompt)
print("   → Result: Well-structured, guided response\n")

# ============================================================================
# 5. RAG-SPECIFIC PROMPT TEMPLATE (for Phase 4)
# ============================================================================
print("=" * 70)
print("5. RAG-SPECIFIC PROMPT TEMPLATE (Preview of Phase 4)")
print("=" * 70)

rag_template = PromptTemplate(
    input_variables=["question", "context", "instructions"],
    template="""Based on the following context, answer the question.

Context:
{context}

Question: {question}

{instructions}

Answer:""".strip()
)

rag_prompt = rag_template.format(
    question="What is a vector database?",
    context="A vector database is a specialized database that stores and retrieves high-dimensional vectors (embeddings) efficiently. It uses algorithms like k-nearest neighbors to find similar vectors quickly.",
    instructions="- Be concise (max 3 sentences)\n- If the context doesn't contain the answer, say 'I don't know from the provided context'"
)

print("\nRAG-Ready Prompt Template:")
print(rag_prompt)
print()

# ============================================================================
# 6. PROMPT TEMPLATE WITH VALIDATION
# ============================================================================
print("=" * 70)
print("6. PROMPT TEMPLATES - INPUT VALIDATION")
print("=" * 70)

validated_template = PromptTemplate(
    input_variables=["user_question"],
    template="Answer this question in one sentence: {user_question}",
    validate_template=True  # Ensures all variables are provided
)

try:
    validated_prompt = validated_template.format(user_question="What is AI?")
    print("\n✅ Valid prompt:")
    print(validated_prompt)
except KeyError as e:
    print(f"\n❌ Error: Missing variable {e}")

# ============================================================================
# SUMMARY & BEST PRACTICES
# ============================================================================
print("\n" + "=" * 70)
print("BEST PRACTICES FOR PROMPT ENGINEERING")
print("=" * 70)
print("""
1. BE SPECIFIC
   - Define the role: "You are a technical writer"
   - Be clear about task: "Explain X in Y manner for Z audience"
   
2. PROVIDE CONTEXT
   - Background information helps the LLM understand intent
   - Clarify any assumptions
   
3. GIVE EXAMPLES (Few-Shot Prompting)
   - 2-3 examples of the desired output format
   - This guides the LLM's behavior
   
4. SET CONSTRAINTS
   - Length limits: "Keep it under 200 words"
   - Format requirements: "Output as JSON"
   - Tone: "Be professional but friendly"
   
5. USE TEMPLATES
   - Reuse successful prompts
   - Easy to modify for similar tasks
   - Maintain consistency across calls
   
6. TEST & ITERATE
   - Compare different prompt variations
   - Measure quality of outputs
   - Refine based on results

7. CHAIN PROMPTS (Advanced)
   - Use output of one prompt as input to next
   - Break complex tasks into steps
   - Each prompt has a clear role
""")

print("=" * 70)
print("NEXT PHASE: RAG SYSTEM")
print("=" * 70)
print("""
Phase 4 will combine PromptTemplate with:
- Document loading and chunking
- Embeddings and vector database
- Retrieval-augmented generation chain
- This is where prompts meet data!
""")
