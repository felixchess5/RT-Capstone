"""
Multi-LLM Provider System with Configurable Priority and Automatic Failover

This module provides a comprehensive LLM management system that supports multiple
providers (Groq, OpenAI, Anthropic, Gemini, Local) with configurable priority
ordering, automatic failover, circuit breaker patterns, and specialized routing.
"""

import os
import time
import yaml
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from dotenv import load_dotenv

# LangChain imports with error handling
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    try:
        import google.generativeai as genai
        GOOGLE_AI_AVAILABLE = True
        GEMINI_AVAILABLE = False
    except ImportError:
        GOOGLE_AI_AVAILABLE = False
        GEMINI_AVAILABLE = False

try:
    from langchain_community.llms import Ollama
    LOCAL_AVAILABLE = True
except ImportError:
    LOCAL_AVAILABLE = False

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure LangSmith tracing if enabled
if os.getenv('LANGCHAIN_TRACING_V2') == 'true':
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv('LANGCHAIN_PROJECT', 'Assignment Grader')
    if os.getenv('LANGCHAIN_API_KEY'):
        os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
        logger.info("LangSmith tracing enabled")


@dataclass
class ProviderHealth:
    """Track health status of LLM providers."""
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    is_healthy: bool = True
    last_success_time: Optional[datetime] = None
    total_requests: int = 0
    successful_requests: int = 0
    average_response_time: float = 0.0


@dataclass
class LLMResponse:
    """Standardized response format for all LLM providers."""
    content: str
    provider: str
    model: str
    response_time: float
    tokens_used: Optional[int] = None
    cost: Optional[float] = None


class CircuitBreaker:
    """Circuit breaker pattern for LLM providers."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def can_execute(self) -> bool:
        """Check if the circuit breaker allows execution."""
        if self.state == "CLOSED":
            return True

        if self.state == "OPEN":
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout):
                self.state = "HALF_OPEN"
                return True
            return False

        if self.state == "HALF_OPEN":
            return True

        return False

    def record_success(self):
        """Record a successful operation."""
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class GeminiWrapper:
    """Wrapper for Google AI client to match LangChain interface."""

    def __init__(self, model: str = "gemini-1.5-pro", temperature: float = 0.7):
        self.model_name = model
        self.temperature = temperature
        if GOOGLE_AI_AVAILABLE:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model)
            else:
                raise ValueError("GEMINI_API_KEY not found")

    def invoke(self, prompt: str) -> LLMResponse:
        """Invoke the Gemini model with a prompt."""
        if not GOOGLE_AI_AVAILABLE:
            raise Exception("Google AI client not available")

        start_time = time.time()
        try:
            if isinstance(prompt, str):
                content = prompt
            else:
                content = prompt.content if hasattr(prompt, 'content') else str(prompt)

            response = self.model.generate_content(
                content,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature
                )
            )

            response_time = time.time() - start_time
            return LLMResponse(
                content=response.text,
                provider="gemini",
                model=self.model_name,
                response_time=response_time
            )
        except Exception as e:
            raise Exception(f"Gemini API call failed: {e}")


class MultiLLMManager:
    """Comprehensive LLM management system with failover and monitoring."""

    def __init__(self, config_path: str = "llm_config.yaml"):
        self.config = self._load_config(config_path)
        self.provider_health: Dict[str, ProviderHealth] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.providers: Dict[str, Any] = {}

        # Initialize health tracking and circuit breakers
        for provider_name in self.config.get('providers', {}):
            self.provider_health[provider_name] = ProviderHealth()
            threshold = self.config.get('failover', {}).get('circuit_breaker_threshold', 5)
            timeout = self.config.get('failover', {}).get('circuit_breaker_timeout', 300)
            self.circuit_breakers[provider_name] = CircuitBreaker(threshold, timeout)

        self._initialize_providers()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default configuration if config file is not found."""
        return {
            'provider_priority': {1: 'groq', 2: 'openai', 3: 'anthropic', 4: 'gemini'},
            'providers': {
                'groq': {'enabled': True, 'models': {'default': 'llama-3.1-8b-instant'}},
                'openai': {'enabled': True, 'models': {'default': 'gpt-4o-mini'}},
                'anthropic': {'enabled': True, 'models': {'default': 'claude-3-5-sonnet-20241022'}},
                'gemini': {'enabled': True, 'models': {'default': 'gemini-1.5-pro'}}
            }
        }

    def _initialize_providers(self):
        """Initialize all enabled LLM providers."""
        for provider_name, config in self.config.get('providers', {}).items():
            if not config.get('enabled', False):
                continue

            try:
                provider = self._create_provider(provider_name, config)
                if provider:
                    self.providers[provider_name] = provider
                    logger.info(f"✓ {provider_name.title()} LLM initialized successfully")
            except Exception as e:
                logger.error(f"✗ {provider_name.title()} LLM initialization failed: {e}")

    def _create_provider(self, provider_name: str, config: Dict) -> Optional[Any]:
        """Create a specific LLM provider instance."""
        model = config.get('models', {}).get('default', '')
        temperature = config.get('temperature', 0.7)
        max_tokens = config.get('max_tokens', 4096)
        timeout = config.get('timeout', 60)

        if provider_name == 'groq' and GROQ_AVAILABLE:
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable required")
            return ChatGroq(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                api_key=api_key
            )

        elif provider_name == 'openai' and OPENAI_AVAILABLE:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable required")
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                api_key=api_key
            )

        elif provider_name == 'anthropic' and ANTHROPIC_AVAILABLE:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable required")
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                api_key=api_key
            )

        elif provider_name == 'gemini':
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable required")

            if GEMINI_AVAILABLE:
                return ChatGoogleGenerativeAI(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    google_api_key=api_key
                )
            elif GOOGLE_AI_AVAILABLE:
                return GeminiWrapper(model=model, temperature=temperature)

        elif provider_name == 'local' and LOCAL_AVAILABLE:
            base_url = config.get('base_url', 'http://localhost:11434')
            return Ollama(
                model=model,
                base_url=base_url,
                temperature=temperature
            )

        return None

    def get_priority_order(self, use_case: Optional[str] = None) -> List[str]:
        """Get provider priority order, optionally for specific use cases."""
        if use_case and use_case in self.config.get('specialized_routing', {}):
            routing_config = self.config['specialized_routing'][use_case]
            return routing_config.get('preferred_providers', [])

        # Use global priority order
        priority_dict = self.config.get('provider_priority', {})
        return [priority_dict[i] for i in sorted(priority_dict.keys()) if priority_dict[i] in self.providers]

    def _record_success(self, provider_name: str, response_time: float):
        """Record successful request for monitoring."""
        health = self.provider_health[provider_name]
        health.total_requests += 1
        health.successful_requests += 1
        health.last_success_time = datetime.now()
        health.is_healthy = True

        # Update average response time
        if health.average_response_time == 0:
            health.average_response_time = response_time
        else:
            health.average_response_time = (health.average_response_time + response_time) / 2

        self.circuit_breakers[provider_name].record_success()

    def _record_failure(self, provider_name: str):
        """Record failed request for monitoring."""
        health = self.provider_health[provider_name]
        health.total_requests += 1
        health.consecutive_failures += 1
        health.last_failure_time = datetime.now()

        # Mark as unhealthy after threshold
        circuit_breaker_threshold = self.config.get('failover', {}).get('circuit_breaker_threshold', 5)
        if health.consecutive_failures >= circuit_breaker_threshold:
            health.is_healthy = False

        self.circuit_breakers[provider_name].record_failure()

    def invoke_with_fallback(self, prompt: str, use_case: Optional[str] = None, max_retries: int = None) -> LLMResponse:
        """Invoke LLM with automatic failover across providers."""
        if max_retries is None:
            max_retries = self.config.get('failover', {}).get('max_total_attempts', 10)

        priority_order = self.get_priority_order(use_case)
        attempts = 0
        last_error = None

        for provider_name in priority_order:
            if attempts >= max_retries:
                break

            if provider_name not in self.providers:
                continue

            # Check circuit breaker
            if not self.circuit_breakers[provider_name].can_execute():
                logger.warning(f"Circuit breaker open for {provider_name}, skipping")
                continue

            provider = self.providers[provider_name]
            retry_attempts = self.config.get('providers', {}).get(provider_name, {}).get('retry_attempts', 3)
            retry_delay = self.config.get('providers', {}).get(provider_name, {}).get('retry_delay', 2)

            # Try this provider with retries
            for retry in range(retry_attempts):
                attempts += 1
                start_time = time.time()

                try:
                    logger.info(f"Attempting {provider_name} (attempt {retry + 1}/{retry_attempts})")

                    # Handle different provider response formats
                    if isinstance(provider, GeminiWrapper):
                        response = provider.invoke(prompt)
                    else:
                        # Standard LangChain providers
                        llm_response = provider.invoke(prompt)
                        response_time = time.time() - start_time

                        # Extract content based on response type
                        if hasattr(llm_response, 'content'):
                            content = llm_response.content
                        elif hasattr(llm_response, 'text'):
                            content = llm_response.text
                        else:
                            content = str(llm_response)

                        # Get model name
                        model_name = getattr(provider, 'model_name', getattr(provider, 'model', 'unknown'))

                        response = LLMResponse(
                            content=content,
                            provider=provider_name,
                            model=model_name,
                            response_time=response_time
                        )

                    # Record success
                    self._record_success(provider_name, response.response_time)

                    if self.config.get('monitoring', {}).get('log_response_times', False):
                        logger.info(f"✓ {provider_name} succeeded in {response.response_time:.2f}s")

                    return response

                except Exception as e:
                    last_error = e
                    logger.warning(f"✗ {provider_name} attempt {retry + 1} failed: {e}")

                    if retry < retry_attempts - 1:  # Don't sleep on last retry
                        time.sleep(retry_delay)

            # All retries for this provider failed
            self._record_failure(provider_name)

        # All providers failed
        error_msg = f"All LLM providers failed after {attempts} attempts. Last error: {last_error}"
        logger.error(error_msg)
        raise Exception(error_msg)

    def get_health_status(self) -> Dict[str, Dict]:
        """Get comprehensive health status of all providers."""
        status = {}
        for provider_name, health in self.provider_health.items():
            success_rate = 0
            if health.total_requests > 0:
                success_rate = (health.successful_requests / health.total_requests) * 100

            status[provider_name] = {
                'is_healthy': health.is_healthy,
                'total_requests': health.total_requests,
                'successful_requests': health.successful_requests,
                'success_rate': f"{success_rate:.1f}%",
                'consecutive_failures': health.consecutive_failures,
                'average_response_time': f"{health.average_response_time:.2f}s",
                'circuit_breaker_state': self.circuit_breakers[provider_name].state,
                'last_success': health.last_success_time.isoformat() if health.last_success_time else None,
                'last_failure': health.last_failure_time.isoformat() if health.last_failure_time else None
            }

        return status

    def reset_provider_health(self, provider_name: str):
        """Reset health status for a specific provider."""
        if provider_name in self.provider_health:
            self.provider_health[provider_name] = ProviderHealth()
            self.circuit_breakers[provider_name] = CircuitBreaker()
            logger.info(f"Health status reset for {provider_name}")


# Initialize the global multi-LLM manager
try:
    llm_manager = MultiLLMManager()
    logger.info(f"Multi-LLM Manager initialized with {len(llm_manager.providers)} providers")
except Exception as e:
    logger.error(f"Failed to initialize Multi-LLM Manager: {e}")
    llm_manager = None


# Backward compatibility functions
def get_available_llm():
    """Get the first available LLM (backward compatibility)."""
    if llm_manager and llm_manager.providers:
        priority_order = llm_manager.get_priority_order()
        for provider_name in priority_order:
            if provider_name in llm_manager.providers:
                provider = llm_manager.providers[provider_name]
                logger.info(f"Using {provider_name} LLM")
                return provider

    logger.error("No LLM providers available")
    return None


def invoke_with_fallback(prompt: str, primary_llm=None, fallback_llm=None):
    """Invoke LLM with fallback (backward compatibility)."""
    if llm_manager:
        try:
            response = llm_manager.invoke_with_fallback(prompt)
            return type('Response', (), {'content': response.content})()
        except Exception as e:
            logger.error(f"Multi-LLM invoke failed: {e}")
            raise
    else:
        raise Exception("Multi-LLM Manager not available")


# Legacy LLM creation functions (for backward compatibility)
def create_groq_llm(model: str = "llama-3.1-8b-instant", temperature: float = 0.7):
    """Create Groq LLM instance (backward compatibility)."""
    if llm_manager and 'groq' in llm_manager.providers:
        return llm_manager.providers['groq']

    if not GROQ_AVAILABLE:
        raise ImportError("Groq provider not available")

    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable required")

    return ChatGroq(model=model, temperature=temperature, api_key=api_key)


def create_gemini_llm(model: str = "gemini-1.5-pro", temperature: float = 0.7):
    """Create Gemini LLM instance (backward compatibility)."""
    if llm_manager and 'gemini' in llm_manager.providers:
        return llm_manager.providers['gemini']

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable required")

    if GEMINI_AVAILABLE:
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)
    elif GOOGLE_AI_AVAILABLE:
        return GeminiWrapper(model=model, temperature=temperature)
    else:
        raise ValueError("No Gemini integration available")


# Initialize legacy variables for backward compatibility
try:
    primary_llm = get_available_llm()
    groq_llm = llm_manager.providers.get('groq') if llm_manager else None
    gemini_llm = llm_manager.providers.get('gemini') if llm_manager else None
except Exception as e:
    logger.error(f"Failed to initialize legacy LLM variables: {e}")
    primary_llm = None
    groq_llm = None
    gemini_llm = None