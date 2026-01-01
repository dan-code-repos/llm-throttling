"""
Simulation Engine
Generates traffic and simulates request lifecycle
"""

import random
from typing import List, Dict, Tuple


class SimulationEngine:
    """
    Simulation engine that:
    - Generates random requests with different intents
    - Advances time in discrete steps
    - Simulates request completions
    - Triggers queue draining
    - Logs all important events
    
    This is purely for POC demonstration purposes.
    """
    
    def __init__(self, throttler):
        """
        Initialize simulation engine.
        
        Args:
            throttler: ModelThrottler instance to route requests through
        """
        self.throttler = throttler
        self.current_step = 0
        self.total_requests = 0
        self.total_completed = 0
        self.request_counter = 0
        
        # Available request intents
        self.intent_types = ['chat', 'agent', 'batch']
        
        # Weighted distribution (chat is more common)
        self.intent_weights = [0.5, 0.3, 0.2]  # chat, agent, batch
        
        # Configurable completion probability
        self.completion_probability = 0.3  # 30% default
    
    def generate_request(self) -> Dict:
        """
        Generate a random request with an intent type.
        
        Returns:
            Dictionary with request details
        """
        intent = random.choices(self.intent_types, weights=self.intent_weights)[0]
        self.request_counter += 1
        
        return {
            'request_id': f'req_{self.current_step}_{self.request_counter}',
            'intent': intent,
            'timestamp': self.current_step
        }
    
    def run_step(self, num_requests: int) -> Tuple[List[Dict], List[str]]:
        """
        Run a single simulation step.
        
        A step consists of:
        1. Generate new requests
        2. Route them through throttler
        3. Simulate some completions
        4. Drain queues where possible
        
        Args:
            num_requests: Number of requests to generate in this step
            
        Returns:
            Tuple of (routing decisions, event log entries)
        """
        self.current_step += 1
        decisions = []
        events = []
        
        events.append(f"📅 Step {self.current_step} started - generating {num_requests} requests")
        
        # Generate and route new requests
        for _ in range(num_requests):
            request = self.generate_request()
            self.total_requests += 1
            
            # Route through throttler
            decision = self.throttler.route_request(
                request['request_id'],
                request['intent']
            )
            
            # Add step information
            decision['step'] = self.current_step
            decision['timestamp'] = request['timestamp']
            
            decisions.append(decision)
            
            # Log important events
            if decision['decision'] == 'FALLBACK_ACCEPTED':
                event_msg = (f"⚠️ FALLBACK ROUTING: {decision['intent']} request "
                           f"({decision['request_id']}) → {decision['selected_model']} "
                           f"(primary {decision['primary_model']} full)")
                events.append(event_msg)
            
            elif decision['decision'] == 'QUEUED':
                event_msg = (f"🔴 QUEUED: {decision['request_id']} to {decision['selected_model']} "
                           f"(both primary and fallback full)")
                events.append(event_msg)
        
        # Simulate request completions and queue draining
        completion_events = self._simulate_completions_with_draining()
        events.extend(completion_events)
        
        return decisions, events
    
    def _simulate_completions_with_draining(self) -> List[str]:
        """
        Simulate probabilistic completion of active requests and queue draining.
        
        Returns:
            List of event log entries
        """
        events = []
        
        for model_name, state in self.throttler.model_states.items():
            # Simulate completion probability for each active request
            if state['active'] > 0:
                completions = sum(1 for _ in range(state['active']) 
                                 if random.random() < self.completion_probability)
                
                for _ in range(completions):
                    # Complete request (this also tries to drain queue)
                    drained_request_id = self.throttler.complete_request(model_name)
                    self.total_completed += 1
                    
                    events.append(f"✓ COMPLETED: Request finished on {model_name}")
                    
                    # If a queued request was drained
                    if drained_request_id:
                        events.append(
                            f"✅ QUEUE DRAINED: {drained_request_id} promoted from "
                            f"queue to active on {model_name}"
                        )
        
        return events
    
    def get_statistics(self) -> Dict:
        """
        Get simulation statistics.
        
        Returns:
            Dictionary with current statistics
        """
        throttler_state = self.throttler.get_state()
        
        return {
            'current_step': self.current_step,
            'total_requests': self.total_requests,
            'total_completed': self.total_completed,
            'total_active': self.throttler.get_total_active(),
            'total_queued': self.throttler.get_total_queued(),
            'model_states': throttler_state
        }
    
    def reset(self):
        """Reset simulation to initial state"""
        self.current_step = 0
        self.total_requests = 0
        self.total_completed = 0
        self.request_counter = 0
        self.completion_probability = 0.3
        
        # Reset throttler state
        for model_name in self.throttler.model_states:
            self.throttler.model_states[model_name] = {
                'active': 0,
                'queued': 0,
                'queue': []
            }
