import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Configure LangSmith tracing if enabled
if os.getenv('LANGCHAIN_TRACING_V2') == 'true':
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv('LANGCHAIN_PROJECT', 'Assignment Grader')
    if os.getenv('LANGCHAIN_API_KEY'):
        os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
        print("✅ LangSmith tracing enabled")

def create_groq_llm(model: str = "llama-3.1-8b-instant", temperature: float = 0.7) -> ChatGroq:
    """Create and configure Groq LLM instance."""
    api_key = os.getenv('GROQ_API_KEY')

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is required. "
            "Please set it in your .env file or environment variables."
        )

    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=api_key
    )


# Initialize LLM only when imported, with proper error handling
try:
    groq_llm = create_groq_llm()
except ValueError as e:
    print(f"[ERROR] {e}")
    groq_llm = None
