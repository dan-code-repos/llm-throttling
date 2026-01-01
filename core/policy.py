"""
Routing Policy
Defines business rules for model selection based on request intent
"""

class RoutingPolicy:
    """
    Defines the routing policy for different request types.
    
    This is where business logic lives:
    - Which model should handle which intent?
    - What's the fallback strategy?
    - Can requests be queued?
    
    This is SEPARATE from platform concerns like rate limiting or token counting.
    """
    
    def __init__(self):
        # Define routing rules for each intent type
        self.routing_rules = {
            'chat': {
                'primary': 'gpt-4o-mini',
                'fallback': 'o1',
                'can_queue': True,
                'description': 'Interactive chat requests prioritize speed'
            },
            'agent': {
                'primary': 'o1',
                'fallback': 'gpt-4o-mini',
                'can_queue': True,
                'description': 'Agent tasks prioritize accuracy'
            },
            'batch': {
                'primary': 'gpt-4.1-mini',
                'fallback': None,  # Batch requests have no fallback
                'can_queue': True,
                'description': 'Batch processing uses dedicated model'
            }
        }
    
    def get_primary_model(self, intent):
        """
        Get the primary model for a given request intent.
        
        Args:
            intent: The type of request (chat, agent, batch)
            
        Returns:
            Model name or None
        """
        rule = self.routing_rules.get(intent)
        return rule['primary'] if rule else None
    
    def get_fallback_model(self, intent):
        """
        Get the fallback model for a given request intent.
        
        Args:
            intent: The type of request (chat, agent, batch)
            
        Returns:
            Model name or None if no fallback exists
        """
        rule = self.routing_rules.get(intent)
        return rule['fallback'] if rule else None
    
    def can_queue(self, intent):
        """
        Check if requests of this intent can be queued.
        
        Args:
            intent: The type of request (chat, agent, batch)
            
        Returns:
            Boolean indicating if queueing is allowed
        """
        rule = self.routing_rules.get(intent)
        return rule['can_queue'] if rule else False
    
    def get_routing_rule(self, intent):
        """
        Get the complete routing rule for an intent.
        
        Args:
            intent: The type of request (chat, agent, batch)
            
        Returns:
            Dictionary with primary, fallback, and queueing info
        """
        return self.routing_rules.get(intent, {})
    
    def get_all_intents(self):
        """Get list of all supported intent types"""
        return list(self.routing_rules.keys())
