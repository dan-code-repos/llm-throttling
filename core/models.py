"""
Model Pool Definition
Defines available LLM models with their metadata and capabilities
"""

class ModelPool:
    """
    Manages the pool of available LLM models with their characteristics.
    
    Each model has:
    - role: The type of workload it's optimized for
    - max_concurrency: Maximum number of concurrent requests it can handle
    - speed: Relative processing speed
    - accuracy: Relative accuracy/quality level
    """
    
    def __init__(self):
        self.models = {
            'gpt-4o-mini': {
                'role': 'chat',
                'max_concurrency': 3,
                'speed': 'fast',
                'accuracy': 'medium',
                'description': 'Optimized for interactive chat with fast responses'
            },
            'o1': {
                'role': 'agent',
                'max_concurrency': 2,
                'speed': 'slow',
                'accuracy': 'high',
                'description': 'High-accuracy model for complex agent tasks'
            },
            'gpt-4.1-mini': {
                'role': 'batch',
                'max_concurrency': 1,
                'speed': 'slow',
                'accuracy': 'low',
                'description': 'Bulk processing model for batch workloads'
            }
        }
    
    def get_model(self, model_name):
        """Get model metadata by name"""
        return self.models.get(model_name)
    
    def get_models_by_role(self, role):
        """Get all models that can handle a specific role"""
        return [
            name for name, info in self.models.items() 
            if info['role'] == role
        ]
    
    def get_all_models(self):
        """Get all available model names"""
        return list(self.models.keys())
    
    def get_max_concurrency(self, model_name):
        """Get maximum concurrency for a model"""
        model = self.get_model(model_name)
        return model['max_concurrency'] if model else 0
