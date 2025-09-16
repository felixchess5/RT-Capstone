import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    print("[WARNING] Gemini LangChain integration not available due to dependency conflicts")
    GEMINI_AVAILABLE = False

    # Fallback: Use Google AI client directly
    try:
        import google.generativeai as genai
        GOOGLE_AI_AVAILABLE = True
    except ImportError:
        print("[WARNING] Google AI client not available")
        GOOGLE_AI_AVAILABLE = False

load_dotenv()

# Configure LangSmith tracing if enabled
if os.getenv('LANGCHAIN_TRACING_V2') == 'true':
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv('LANGCHAIN_PROJECT', 'Assignment Grader')
    if os.getenv('LANGCHAIN_API_KEY'):
        os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
        print("LangSmith tracing enabled")

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

def create_gemini_llm(model: str = "gemini-1.5-pro", temperature: float = 0.7):
    """Create and configure Gemini LLM instance."""
    api_key = os.getenv('GEMINI_API_KEY')

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is required. "
            "Please set it in your .env file or environment variables."
        )

    if GEMINI_AVAILABLE:
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key
        )
    elif GOOGLE_AI_AVAILABLE:
        # Fallback to direct Google AI client
        genai.configure(api_key=api_key)
        return GeminiWrapper(model=model, temperature=temperature)
    else:
        raise ValueError("No Gemini integration available")

class GeminiWrapper:
    """Wrapper for Google AI client to match LangChain interface."""

    def __init__(self, model: str = "gemini-1.5-pro", temperature: float = 0.7):
        self.model_name = model
        self.temperature = temperature
        if GOOGLE_AI_AVAILABLE:
            self.model = genai.GenerativeModel(model)

    def invoke(self, prompt: str):
        """Invoke the Gemini model with a prompt."""
        if not GOOGLE_AI_AVAILABLE:
            raise Exception("Google AI client not available")

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature
                )
            )
            # Create a response object that matches LangChain format
            class Response:
                def __init__(self, content):
                    self.content = content

            return Response(response.text)
        except Exception as e:
            raise Exception(f"Gemini API call failed: {e}")

def get_available_llm():
    """Get the first available LLM with redundancy fallback."""
    # Try Groq first
    try:
        groq_llm = create_groq_llm()
        print("Using Groq LLM")
        return groq_llm
    except ValueError as e:
        print(f"[WARNING] Groq LLM unavailable: {e}")

    # Fallback to Gemini
    try:
        gemini_llm = create_gemini_llm()
        print("Using Gemini LLM as fallback")
        return gemini_llm
    except ValueError as e:
        print(f"[WARNING] Gemini LLM unavailable: {e}")

    # If both fail
    print("[ERROR] No LLM providers available")
    return None

def invoke_with_fallback(prompt: str, primary_llm=None, fallback_llm=None):
    """Invoke LLM with automatic fallback if primary fails."""
    if primary_llm is None:
        primary_llm = groq_llm
    if fallback_llm is None:
        fallback_llm = gemini_llm

    # Try primary LLM
    try:
        if primary_llm is not None:
            response = primary_llm.invoke(prompt)
            return response
    except Exception as e:
        print(f"[WARNING] Primary LLM failed: {e}")

    # Try fallback LLM
    try:
        if fallback_llm is not None:
            print("Switching to fallback LLM")
            response = fallback_llm.invoke(prompt)
            return response
    except Exception as e:
        print(f"[ERROR] Fallback LLM also failed: {e}")

    raise Exception("All LLM providers failed")

# Initialize LLMs with proper error handling
try:
    groq_llm = create_groq_llm()
    print("Groq LLM initialized successfully")
except ValueError as e:
    print(f"[ERROR] Groq LLM initialization failed: {e}")
    groq_llm = None

try:
    gemini_llm = create_gemini_llm()
    print("Gemini LLM initialized successfully")
except ValueError as e:
    print(f"[ERROR] Gemini LLM initialization failed: {e}")
    gemini_llm = None

# Set the primary LLM to the first available one
primary_llm = get_available_llm()
