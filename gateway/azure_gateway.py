"""
Azure / LLM Gateway Responsibilities
Documentation of what the platform layer handles (NOT implemented in throttler)
"""


class AzureGatewayResponsibilities:
    """
    This class documents responsibilities that belong to Azure OpenAI,
    Azure API Management, or other gateway/platform services.
    
    These are NOT implemented in the model throttler.
    This separation is critical for architectural clarity.
    """
    
    def __init__(self):
        self.responsibilities = {
            'Rate Limiting': [
                'Requests per minute (RPM) enforcement per deployment',
                'Tokens per minute (TPM) enforcement per deployment',
                'Per-user or per-API-key rate limiting',
                'Quota management across multiple deployments',
                'Burst allowance and rate smoothing'
            ],
            'Token Management': [
                'Token counting for requests and responses',
                'Token-based billing calculations',
                'Maximum token limit enforcement',
                'Token budget allocation across users',
                'Context window management'
            ],
            'Retry & Resilience': [
                'Automatic retry with exponential backoff',
                'Circuit breaker patterns for failing deployments',
                '429 (rate limit) and 503 (service unavailable) handling',
                'Timeout management and cancellation',
                'Health check monitoring'
            ],
            'Security & Authentication': [
                'API key validation and rotation',
                'Azure AD / OAuth authentication',
                'Network security and firewall rules',
                'VNet integration and private endpoints',
                'SSL/TLS encryption',
                'Request signing and validation'
            ],
            'Infrastructure': [
                'Connection pooling and management',
                'Load balancing across backend instances',
                'Geographic routing and latency optimization',
                'Auto-scaling based on traffic patterns',
                'Cache management for repeated queries'
            ],
            'Monitoring & Observability': [
                'Request/response logging',
                'Latency and performance metrics',
                'Error rate tracking',
                'Cost attribution and chargeback',
                'Distributed tracing'
            ],
            'Data Governance': [
                'Content filtering (prompt shields, content safety)',
                'PII detection and masking',
                'Data residency compliance',
                'Audit logging for compliance',
                'GDPR and regulatory compliance'
            ]
        }
    
    def get_all_responsibilities(self):
        """Get all gateway responsibilities organized by category"""
        return self.responsibilities
    
    def get_category(self, category_name):
        """Get responsibilities for a specific category"""
        return self.responsibilities.get(category_name, [])
    
    def get_flat_list(self):
        """Get a flat list of all responsibilities"""
        all_items = []
        for category, items in self.responsibilities.items():
            all_items.extend(items)
        return all_items
    
    @staticmethod
    def get_explanation():
        """
        Get explanation of why this separation matters.
        
        Returns:
            String explaining the architectural separation
        """
        return """
        The Model Throttler focuses on BUSINESS LOGIC:
        - Which model should handle this request?
        - What's the fallback strategy?
        - How should we queue when capacity is limited?
        
        The Azure/Gateway handles PLATFORM MECHANICS:
        - How do we enforce API limits?
        - How do we count tokens?
        - How do we handle network failures?
        
        This separation ensures:
        1. Clear ownership boundaries
        2. Independent scaling of concerns
        3. Easier testing and maintenance
        4. Flexibility to swap platform components
        """
