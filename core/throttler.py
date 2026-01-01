"""
Model Throttler
Production-grade throttling and routing logic for LLM requests
"""

import random
from typing import Dict, Optional, List


class ModelThrottler:
    """
    Production-grade model throttler implementing:
    - Intent-aware routing
    - Primary/fallback model selection
    - Queue management
    - Queue draining when capacity becomes available
    
    This represents what a real production throttler does.
    This does NOT implement platform concerns like:
    - Rate limiting (RPM/TPM)
    - Token counting
    - Retry logic
    - Network security
    """
    
    def __init__(self, model_pool, routing_policy):
        """
        Initialize the throttler.
        
        Args:
            model_pool: ModelPool instance with model metadata
            routing_policy: RoutingPolicy instance with routing rules
        """
        self.model_pool = model_pool
        self.routing_policy = routing_policy
        
        # Track state per model
        self.model_states = {}
        for model_name in model_pool.get_all_models():
            self.model_states[model_name] = {
                'active': 0,      # Currently executing requests
                'queued': 0,      # Waiting requests
                'queue': []       # Queue of request IDs
            }
    
    def route_request(self, request_id: str, intent: str) -> Dict:
        """
        Route a request based on intent and current capacity.
        
        This is the core routing logic that:
        1. Tries primary model
        2. Falls back if primary is full
        3. Queues if both are unavailable
        
        Args:
            request_id: Unique identifier for the request
            intent: Type of request (chat, agent, batch)
            
        Returns:
            Dictionary with routing decision details
        """
        # Get primary and fallback models from policy
        primary_model = self.routing_policy.get_primary_model(intent)
        fallback_model = self.routing_policy.get_fallback_model(intent)
        can_queue = self.routing_policy.can_queue(intent)
        
        decision = {
            'request_id': request_id,
            'intent': intent,
            'primary_model': primary_model,
            'fallback_model': fallback_model
        }
        
        # Try primary model first
        if self._has_capacity(primary_model):
            self._add_active_request(primary_model, request_id)
            decision['selected_model'] = primary_model
            decision['decision'] = 'PRIMARY_ACCEPTED'
            decision['reason'] = f'Primary model {primary_model} has capacity'
            return decision
        
        # Try fallback model if primary is full
        if fallback_model and self._has_capacity(fallback_model):
            self._add_active_request(fallback_model, request_id)
            decision['selected_model'] = fallback_model
            decision['decision'] = 'FALLBACK_ACCEPTED'
            decision['reason'] = f'Primary {primary_model} full, using fallback {fallback_model}'
            return decision
        
        # Queue if allowed and both models are full
        if can_queue:
            # Queue to primary model by default
            self._add_to_queue(primary_model, request_id)
            decision['selected_model'] = primary_model
            decision['decision'] = 'QUEUED'
            decision['reason'] = f'Both primary and fallback full, queued to {primary_model}'
            return decision
        
        # Request rejected (no capacity, no fallback, no queue)
        decision['selected_model'] = None
        decision['decision'] = 'REJECTED'
        decision['reason'] = 'No capacity, no fallback, queueing not allowed'
        return decision
    
    def _has_capacity(self, model_name: str) -> bool:
        """
        Check if a model has available capacity.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model can accept more requests
        """
        if not model_name:
            return False
        
        state = self.model_states.get(model_name)
        max_concurrency = self.model_pool.get_max_concurrency(model_name)
        
        return state['active'] < max_concurrency
    
    def _add_active_request(self, model_name: str, request_id: str):
        """
        Add a request to a model's active (executing) list.
        
        Args:
            model_name: Model to add request to
            request_id: Request identifier
        """
        self.model_states[model_name]['active'] += 1
    
    def _add_to_queue(self, model_name: str, request_id: str):
        """
        Add a request to a model's queue.
        
        Args:
            model_name: Model to queue request for
            request_id: Request identifier
        """
        state = self.model_states[model_name]
        state['queued'] += 1
        state['queue'].append(request_id)
    
    def complete_request(self, model_name: str) -> Optional[str]:
        """
        Simulate completion of an active request.
        
        When a request completes:
        1. Decrement active count
        2. Try to drain queue (move queued -> active)
        
        Args:
            model_name: Model completing a request
            
        Returns:
            Request ID if queue was drained, None otherwise
        """
        state = self.model_states[model_name]
        
        # Decrement active requests
        if state['active'] > 0:
            state['active'] -= 1
        
        # Try to drain queue
        return self._drain_queue(model_name)
    
    def _drain_queue(self, model_name: str) -> Optional[str]:
        """
        Move a queued request to active if capacity is available.
        
        This is critical for queue draining behavior:
        - When a request completes, check the queue
        - If queue has items and capacity exists, promote to active
        
        Args:
            model_name: Model to drain queue for
            
        Returns:
            Request ID that was drained, or None
        """
        state = self.model_states[model_name]
        
        # Check if we have capacity and queued items
        if self._has_capacity(model_name) and state['queue']:
            # Remove from queue
            request_id = state['queue'].pop(0)
            state['queued'] -= 1
            
            # Add to active
            state['active'] += 1
            
            return request_id
        
        return None
    
    def simulate_completions(self) -> List[str]:
        """
        Simulate probabilistic completion of active requests.
        
        For demonstration purposes:
        - Each active request has a chance to complete
        - Completed requests trigger queue draining
        
        Returns:
            List of models that had completions
        """
        completed_models = []
        
        for model_name, state in self.model_states.items():
            # Simulate completion probability
            if state['active'] > 0:
                # 30% chance per active request to complete
                completions = sum(1 for _ in range(state['active']) 
                                 if random.random() < 0.3)
                
                for _ in range(completions):
                    drained_request = self.complete_request(model_name)
                    if model_name not in completed_models:
                        completed_models.append(model_name)
        
        return completed_models
    
    def get_state(self) -> Dict:
        """
        Get current state of all models.
        
        Returns:
            Dictionary with per-model state information
        """
        return {
            model_name: {
                'active': state['active'],
                'queued': state['queued'],
                'capacity': self.model_pool.get_max_concurrency(model_name)
            }
            for model_name, state in self.model_states.items()
        }
    
    def get_total_queued(self) -> int:
        """Get total number of queued requests across all models"""
        return sum(state['queued'] for state in self.model_states.values())
    
    def get_total_active(self) -> int:
        """Get total number of active requests across all models"""
        return sum(state['active'] for state in self.model_states.values())
