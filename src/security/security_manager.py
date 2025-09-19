"""
Enterprise-grade Security Manager for RT-Capstone.

Provides comprehensive protection against:
- Prompt injection attacks
- Prompt manipulation
- Content extraction attempts
- Data exfiltration
- Input validation bypass
- Output manipulation
- Rate limiting bypass
- Authentication bypass
"""

import re
import hashlib
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import base64

import bleach
import validators
from pydantic import BaseModel, ValidationError, Field
from cryptography.fernet import Fernet

# Configure security logger
security_logger = logging.getLogger("security")
security_handler = logging.FileHandler("security.log")
security_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
security_handler.setFormatter(security_formatter)
security_logger.addHandler(security_handler)
security_logger.setLevel(logging.INFO)


@dataclass
class SecurityThreat:
    """Represents a detected security threat."""
    threat_type: str
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    description: str
    input_content: str
    timestamp: datetime = field(default_factory=datetime.now)
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    blocked: bool = False


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    enable_prompt_injection_protection: bool = True
    enable_content_filtering: bool = True
    enable_rate_limiting: bool = True
    enable_input_validation: bool = True
    enable_output_sanitization: bool = True
    enable_audit_logging: bool = True
    max_input_length: int = 100000  # 100KB
    max_tokens_per_minute: int = 1000
    max_requests_per_minute: int = 60
    blocked_patterns_file: Optional[str] = None
    whitelist_domains: Set[str] = field(default_factory=set)
    encryption_key: Optional[bytes] = None


class InputValidator(BaseModel):
    """Pydantic model for input validation."""
    content: str = Field(..., max_length=100000, min_length=1)
    content_type: str = Field(default="text", pattern=r"^(text|file|json)$")
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        str_strip_whitespace = True
        validate_assignment = True


class SecurityManager:
    """Enterprise-grade security manager for RT-Capstone."""

    def __init__(self, config: SecurityConfig = None):
        """Initialize security manager with configuration."""
        self.config = config or SecurityConfig()
        self.threats: List[SecurityThreat] = []
        self.rate_limiter = RateLimiter(
            max_requests=self.config.max_requests_per_minute,
            time_window=60
        )
        self.content_filter = ContentFilter()
        self.prompt_guardian = PromptInjectionGuard()
        self.input_sanitizer = InputSanitizer()
        self.output_sanitizer = OutputSanitizer()

        # Initialize encryption if key provided
        self.cipher_suite = None
        if self.config.encryption_key:
            self.cipher_suite = Fernet(self.config.encryption_key)

        # Load threat patterns
        self._load_threat_patterns()

        security_logger.info("SecurityManager initialized with enterprise-grade protection")

    def _load_threat_patterns(self):
        """Load threat detection patterns."""
        # This would load from external threat intelligence feeds
        # For now, we'll use built-in patterns
        pass

    def validate_and_sanitize_input(
        self,
        content: str,
        content_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        source_ip: Optional[str] = None
    ) -> Tuple[str, List[SecurityThreat]]:
        """
        Comprehensive input validation and sanitization.

        Returns:
            Tuple of (sanitized_content, detected_threats)
        """
        threats = []

        try:
            # 1. Rate limiting check
            if self.config.enable_rate_limiting:
                if not self.rate_limiter.allow_request(user_id or source_ip or "anonymous"):
                    threat = SecurityThreat(
                        threat_type="RATE_LIMIT_EXCEEDED",
                        severity="MEDIUM",
                        description="Rate limit exceeded",
                        input_content=content[:100],
                        source_ip=source_ip,
                        user_id=user_id,
                        blocked=True
                    )
                    threats.append(threat)
                    self._log_security_event(threat)
                    raise SecurityError("Rate limit exceeded")

            # 2. Input validation
            if self.config.enable_input_validation:
                try:
                    validated_input = InputValidator(
                        content=content,
                        content_type=content_type,
                        metadata=metadata
                    )
                    content = validated_input.content
                except ValidationError as e:
                    threat = SecurityThreat(
                        threat_type="INVALID_INPUT",
                        severity="MEDIUM",
                        description=f"Input validation failed: {str(e)}",
                        input_content=content[:100],
                        source_ip=source_ip,
                        user_id=user_id,
                        blocked=True
                    )
                    threats.append(threat)
                    self._log_security_event(threat)
                    raise SecurityError(f"Input validation failed: {str(e)}")

            # 3. Prompt injection detection
            if self.config.enable_prompt_injection_protection:
                injection_threats = self.prompt_guardian.detect_injection(content)
                for injection_threat in injection_threats:
                    injection_threat.source_ip = source_ip
                    injection_threat.user_id = user_id
                    threats.append(injection_threat)
                    self._log_security_event(injection_threat)

            # 4. Content filtering
            if self.config.enable_content_filtering:
                content_threats = self.content_filter.filter_content(content)
                for content_threat in content_threats:
                    content_threat.source_ip = source_ip
                    content_threat.user_id = user_id
                    threats.append(content_threat)
                    self._log_security_event(content_threat)

            # 5. Input sanitization
            sanitized_content = self.input_sanitizer.sanitize(content)

            # Check if any critical threats were detected
            critical_threats = [t for t in threats if t.severity == "CRITICAL"]
            if critical_threats:
                for threat in critical_threats:
                    threat.blocked = True
                raise SecurityError("Critical security threat detected")

            return sanitized_content, threats

        except SecurityError:
            raise
        except Exception as e:
            threat = SecurityThreat(
                threat_type="VALIDATION_ERROR",
                severity="HIGH",
                description=f"Unexpected validation error: {str(e)}",
                input_content=content[:100],
                source_ip=source_ip,
                user_id=user_id,
                blocked=True
            )
            threats.append(threat)
            self._log_security_event(threat)
            raise SecurityError(f"Security validation failed: {str(e)}")

    def sanitize_output(self, content: str, content_type: str = "text") -> str:
        """Sanitize output content before returning to user."""
        if not self.config.enable_output_sanitization:
            return content

        try:
            return self.output_sanitizer.sanitize(content, content_type)
        except Exception as e:
            security_logger.error(f"Output sanitization failed: {str(e)}")
            return "Content unavailable due to security policy"

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        if not self.cipher_suite:
            raise SecurityError("Encryption not configured")

        return base64.b64encode(
            self.cipher_suite.encrypt(data.encode())
        ).decode()

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        if not self.cipher_suite:
            raise SecurityError("Encryption not configured")

        return self.cipher_suite.decrypt(
            base64.b64decode(encrypted_data.encode())
        ).decode()

    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)

        recent_threats = [t for t in self.threats if t.timestamp >= last_24h]

        threat_counts = defaultdict(int)
        severity_counts = defaultdict(int)

        for threat in recent_threats:
            threat_counts[threat.threat_type] += 1
            severity_counts[threat.severity] += 1

        return {
            "report_generated": now.isoformat(),
            "total_threats_24h": len(recent_threats),
            "blocked_attempts_24h": len([t for t in recent_threats if t.blocked]),
            "threat_types": dict(threat_counts),
            "severity_distribution": dict(severity_counts),
            "rate_limiter_stats": self.rate_limiter.get_stats(),
            "security_config": {
                "prompt_injection_protection": self.config.enable_prompt_injection_protection,
                "content_filtering": self.config.enable_content_filtering,
                "rate_limiting": self.config.enable_rate_limiting,
                "input_validation": self.config.enable_input_validation,
                "output_sanitization": self.config.enable_output_sanitization
            }
        }

    def _log_security_event(self, threat: SecurityThreat):
        """Log security events for monitoring and analysis."""
        self.threats.append(threat)

        log_data = {
            "threat_type": threat.threat_type,
            "severity": threat.severity,
            "description": threat.description,
            "timestamp": threat.timestamp.isoformat(),
            "source_ip": threat.source_ip,
            "user_id": threat.user_id,
            "blocked": threat.blocked,
            "input_hash": hashlib.sha256(threat.input_content.encode()).hexdigest()[:16]
        }

        security_logger.warning(f"Security threat detected: {json.dumps(log_data)}")


class PromptInjectionGuard:
    """Advanced prompt injection detection and prevention."""

    def __init__(self):
        """Initialize prompt injection guard with detection patterns."""
        self.injection_patterns = self._load_injection_patterns()
        self.encoding_patterns = self._load_encoding_patterns()
        self.system_commands = self._load_system_commands()

    def _load_injection_patterns(self) -> List[re.Pattern]:
        """Load prompt injection detection patterns."""
        patterns = [
            # Direct instruction override attempts
            r"ignore\s+(?:previous|all|the|above|your)\s+(?:instructions?|prompts?|commands?)",
            r"forget\s+(?:everything|all|previous|the\s+above)",
            r"disregard\s+(?:previous|all|the|above)\s+(?:instructions?|prompts?)",
            r"new\s+(?:instructions?|prompts?|tasks?)\s*:",
            r"(?:instead|now)\s+(?:do|perform|execute|run)\s+(?:this|the\s+following)",

            # Role manipulation
            r"you\s+are\s+(?:now|instead)\s+(?:a|an)\s+\w+",
            r"act\s+as\s+(?:a|an)\s+\w+",
            r"pretend\s+(?:to\s+be|you\s+are)\s+(?:a|an)\s+\w+",
            r"roleplay\s+as\s+(?:a|an)\s+\w+",

            # System prompt extraction
            r"(?:show|display|print|output|reveal)\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions?)",
            r"what\s+(?:are|is)\s+your\s+(?:system\s+)?(?:prompt|instructions?)",
            r"repeat\s+your\s+(?:system\s+)?(?:prompt|instructions?)",

            # Jailbreak attempts
            r"hypothetically?\s*,?\s*if\s+you\s+(?:were|could|had)",
            r"in\s+a\s+fictional\s+(?:scenario|world|universe)",
            r"for\s+(?:educational|research|academic)\s+purposes\s+only",
            r"(?:simulate|imagine)\s+you\s+(?:are|were|can|could)",

            # Code injection
            r"```\s*(?:python|javascript|bash|sql|html)",
            r"<script[^>]*>",
            r"eval\s*\(",
            r"exec\s*\(",
            r"import\s+os|import\s+subprocess",

            # Prompt leakage
            r"(?:original|initial|first)\s+(?:prompt|instructions?|system\s+message)",
            r"training\s+(?:data|prompt|instructions?)",
            r"model\s+(?:weights|parameters|configuration)",

            # Delimiter manipulation
            r"---+\s*(?:END|STOP|BREAK)",
            r"```+\s*(?:END|STOP|BREAK)",
            r"\*\*\*+\s*(?:END|STOP|BREAK)",

            # Unicode and encoding attacks
            r"\\u[0-9a-fA-F]{4}",
            r"\\x[0-9a-fA-F]{2}",
            r"%[0-9a-fA-F]{2}",

            # SQL injection patterns
            r"(?:union|select|insert|update|delete|drop)\s+",
            r"or\s+1\s*=\s*1",
            r"and\s+1\s*=\s*1",

            # Command injection
            r"[;&|`$]\s*(?:cat|ls|pwd|whoami|id|uname)",
            r">\s*/(?:etc|var|tmp)/",
        ]

        return [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in patterns]

    def _load_encoding_patterns(self) -> List[re.Pattern]:
        """Load patterns for detecting encoding-based attacks."""
        patterns = [
            r"base64\s*[:=]\s*[A-Za-z0-9+/]+=*",
            r"data:(?:text|application)/[^;]+;base64,",
            r"\\\\x[0-9a-fA-F]{2}(?:[0-9a-fA-F]{2}){3,}",  # Long hex sequences
            r"(?:%[0-9a-fA-F]{2}){4,}",  # URL encoding chains
        ]

        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def _load_system_commands(self) -> Set[str]:
        """Load known system commands that shouldn't appear in prompts."""
        return {
            'rm', 'del', 'rmdir', 'format', 'fdisk', 'mkfs',
            'chmod', 'chown', 'sudo', 'su', 'passwd',
            'wget', 'curl', 'nc', 'netcat', 'telnet',
            'ssh', 'scp', 'ftp', 'tftp', 'rsync',
            'ps', 'kill', 'killall', 'pkill', 'top',
            'netstat', 'ss', 'lsof', 'tcpdump', 'wireshark'
        }

    def detect_injection(self, content: str) -> List[SecurityThreat]:
        """Detect prompt injection attempts in content."""
        threats = []
        content_lower = content.lower()

        # 1. Pattern-based detection
        for pattern in self.injection_patterns:
            matches = pattern.findall(content)
            if matches:
                threat = SecurityThreat(
                    threat_type="PROMPT_INJECTION",
                    severity="HIGH",
                    description=f"Prompt injection pattern detected: {pattern.pattern}",
                    input_content=content[:200]
                )
                threats.append(threat)

        # 2. Encoding attack detection
        for pattern in self.encoding_patterns:
            if pattern.search(content):
                threat = SecurityThreat(
                    threat_type="ENCODING_ATTACK",
                    severity="MEDIUM",
                    description="Suspicious encoding detected",
                    input_content=content[:200]
                )
                threats.append(threat)

        # 3. System command detection
        words = re.findall(r'\b\w+\b', content_lower)
        suspicious_commands = [word for word in words if word in self.system_commands]
        if suspicious_commands:
            threat = SecurityThreat(
                threat_type="SYSTEM_COMMAND",
                severity="MEDIUM",
                description=f"System commands detected: {suspicious_commands}",
                input_content=content[:200]
            )
            threats.append(threat)

        # 4. Length-based anomaly detection
        if len(content) > 50000:  # Unusually long input
            threat = SecurityThreat(
                threat_type="ANOMALOUS_LENGTH",
                severity="MEDIUM",
                description="Unusually long input detected",
                input_content=content[:200]
            )
            threats.append(threat)

        # 5. Repetition-based detection
        if self._detect_repetition_attack(content):
            threat = SecurityThreat(
                threat_type="REPETITION_ATTACK",
                severity="MEDIUM",
                description="Repetition-based attack detected",
                input_content=content[:200]
            )
            threats.append(threat)

        return threats

    def _detect_repetition_attack(self, content: str) -> bool:
        """Detect repetition-based prompt injection attacks."""
        lines = content.split('\n')
        if len(lines) < 10:
            return False

        # Check for repeated identical lines
        line_counts = defaultdict(int)
        for line in lines:
            line_counts[line.strip()] += 1

        # If any line is repeated more than 5 times, it's suspicious
        return any(count > 5 for count in line_counts.values())


class ContentFilter:
    """Advanced content filtering for malicious content detection."""

    def __init__(self):
        """Initialize content filter."""
        self.malicious_patterns = self._load_malicious_patterns()
        self.sensitive_data_patterns = self._load_sensitive_data_patterns()

    def _load_malicious_patterns(self) -> List[re.Pattern]:
        """Load patterns for malicious content detection."""
        patterns = [
            # PII extraction attempts
            r"(?:ssn|social\s+security)\s*:?\s*\d{3}-?\d{2}-?\d{4}",
            r"credit\s+card\s*:?\s*\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}",
            r"password\s*:?\s*\w+",

            # Malicious URLs
            r"https?://(?:bit\.ly|tinyurl|t\.co|shortened\.link)/\w+",
            r"(?:phishing|malware|virus|trojan)\s+(?:site|url|link)",

            # Data exfiltration
            r"(?:copy|send|email|transmit)\s+(?:all|this|the)\s+(?:data|information|content)",
            r"(?:download|export|extract)\s+(?:database|files|documents)",
        ]

        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def _load_sensitive_data_patterns(self) -> List[re.Pattern]:
        """Load patterns for sensitive data detection."""
        patterns = [
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            r"\b\d{3}-\d{3}-\d{4}\b",  # Phone number
            r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",  # IP address
        ]

        return [re.compile(pattern) for pattern in patterns]

    def filter_content(self, content: str) -> List[SecurityThreat]:
        """Filter content for malicious patterns."""
        threats = []

        # Check for malicious patterns
        for pattern in self.malicious_patterns:
            if pattern.search(content):
                threat = SecurityThreat(
                    threat_type="MALICIOUS_CONTENT",
                    severity="HIGH",
                    description=f"Malicious pattern detected: {pattern.pattern}",
                    input_content=content[:200]
                )
                threats.append(threat)

        # Check for sensitive data
        for pattern in self.sensitive_data_patterns:
            matches = pattern.findall(content)
            if matches:
                threat = SecurityThreat(
                    threat_type="SENSITIVE_DATA",
                    severity="MEDIUM",
                    description=f"Potential sensitive data detected: {len(matches)} matches",
                    input_content=content[:200]
                )
                threats.append(threat)

        return threats


class InputSanitizer:
    """Input sanitization for safe processing."""

    def __init__(self):
        """Initialize input sanitizer."""
        self.html_tags = [
            'script', 'iframe', 'object', 'embed', 'form',
            'input', 'button', 'select', 'textarea', 'style'
        ]

    def sanitize(self, content: str) -> str:
        """Sanitize input content."""
        # 1. HTML sanitization
        content = bleach.clean(
            content,
            tags=[],  # Remove all HTML tags
            attributes={},
            strip=True
        )

        # 2. Remove null bytes and control characters
        content = ''.join(char for char in content if ord(char) >= 32 or char in '\n\r\t')

        # 3. Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()

        # 4. Limit length
        if len(content) > 100000:
            content = content[:100000] + "... [truncated for security]"

        return content


class OutputSanitizer:
    """Output sanitization for safe response delivery."""

    def sanitize(self, content: str, content_type: str = "text") -> str:
        """Sanitize output content."""
        if content_type == "html":
            # Strict HTML sanitization
            allowed_tags = ['p', 'br', 'strong', 'em', 'ul', 'ol', 'li']
            content = bleach.clean(
                content,
                tags=allowed_tags,
                attributes={},
                strip=True
            )
        else:
            # Text sanitization
            content = bleach.clean(content, tags=[], attributes={}, strip=True)

        # Remove potential data exfiltration attempts
        content = re.sub(r'data:[^;]+;base64,[A-Za-z0-9+/=]+', '[DATA_REMOVED]', content)

        return content


class RateLimiter:
    """Token bucket rate limiter for request throttling."""

    def __init__(self, max_requests: int = 60, time_window: int = 60):
        """Initialize rate limiter."""
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(deque)
        self.stats = defaultdict(int)

    def allow_request(self, identifier: str) -> bool:
        """Check if request is allowed under rate limiting."""
        now = time.time()
        window_start = now - self.time_window

        # Clean old requests
        user_requests = self.requests[identifier]
        while user_requests and user_requests[0] < window_start:
            user_requests.popleft()

        # Check if under limit
        if len(user_requests) >= self.max_requests:
            self.stats[f"{identifier}_blocked"] += 1
            return False

        # Add current request
        user_requests.append(now)
        self.stats[f"{identifier}_allowed"] += 1
        return True

    def get_stats(self) -> Dict[str, int]:
        """Get rate limiting statistics."""
        return dict(self.stats)


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


# Security middleware for integration
class SecurityMiddleware:
    """Security middleware for web applications."""

    def __init__(self, security_manager: SecurityManager):
        """Initialize security middleware."""
        self.security_manager = security_manager

    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request through security filters."""
        content = request_data.get('content', '')
        user_id = request_data.get('user_id')
        source_ip = request_data.get('source_ip')

        try:
            sanitized_content, threats = self.security_manager.validate_and_sanitize_input(
                content=content,
                user_id=user_id,
                source_ip=source_ip
            )

            return {
                'content': sanitized_content,
                'threats_detected': len(threats),
                'security_status': 'PASSED'
            }

        except SecurityError as e:
            return {
                'content': '',
                'error': str(e),
                'security_status': 'BLOCKED'
            }

    def process_response(self, response_data: str) -> str:
        """Process outgoing response through security filters."""
        return self.security_manager.sanitize_output(response_data)