from typing import Callable, Dict, Any, Union, NewType, Literal, get_type_hints
from functools import wraps
import inspect
from fastapi import Depends

# Type definitions
CallableRef = NewType('CallableRef', str)

class FunctionValidator:
    """Validates function calls against their metadata."""
    
    @staticmethod
    def validate_parameters(func: Callable, kwargs: Dict[str, Any]) -> None:
        """Validate parameters against function signature and type hints."""
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        # Get required and optional parameters
        required_params = {
            name for name, param in sig.parameters.items()
            if param.default == inspect.Parameter.empty
        }
        
        # Check for missing required parameters
        missing_params = required_params - set(kwargs.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
        
        # Check for unexpected parameters
        valid_params = set(sig.parameters.keys())
        unexpected_params = set(kwargs.keys()) - valid_params
        if unexpected_params:
            raise ValueError(f"Unexpected parameters: {unexpected_params}")
        
        # Validate parameter types
        for param_name, value in kwargs.items():
            if param_name in type_hints:
                expected_type = type_hints[param_name]
                
                # Handle Union types
                if hasattr(expected_type, "__origin__") and expected_type.__origin__ is Union:
                    if not any(isinstance(value, t) for t in expected_type.__args__):
                        raise TypeError(
                            f"Parameter '{param_name}' must be one of {expected_type.__args__}, "
                            f"got {type(value)}"
                        )
                else:
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"Parameter '{param_name}' must be of type {expected_type}, "
                            f"got {type(value)}"
                        )

    @staticmethod
    async def validate_against_db(session: Depends, 
                                  fn: str, 
                                  kwargs: Dict[str, Any], 
                                  func_type : Literal['validator', 'sampler']='validator') -> Dict:
        """Validate function call against database metadata."""
        result = await session["judges"].find_one({
            f"{func_type}.function.name": fn,
            "active": True
        })
        
        if result is None:
            raise ValueError(f"Function '{fn}' does not exist within registered functions")
            
        # Get function definition from database
        func_def = result["validator"]["function"]
        
        # Validate required parameters from DB definition
        required_params = {
            name for name, info in func_def["parameters"].items()
            if info.get("required", True)
        }
        
        missing_params = required_params - set(kwargs.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
        
        # Validate parameter types from DB definition
        for param_name, value in kwargs.items():
            if param_name not in func_def["parameters"]:
                raise ValueError(f"Unexpected parameter: {param_name}")
            
            param_info = func_def["parameters"][param_name]
            
            # Handle union types
            if param_info["type"] == "union":
                valid_types = param_info["types"]
                if not any(isinstance(value, eval(t)) for t in valid_types if t != "None"):
                    if value is not None or "None" not in valid_types:
                        raise TypeError(
                            f"Parameter '{param_name}' must be one of {valid_types}, "
                            f"got {type(value).__name__}"
                        )
            else:
                expected_type = eval(param_info["type"])
                if not isinstance(value, expected_type):
                    raise TypeError(
                        f"Parameter '{param_name}' must be of type {param_info['type']}, "
                        f"got {type(value).__name__}"
                    )
        
        return result

class FunctionRegistry:
    """Class to hold the samplers and validators registry"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FunctionRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize samplers and validators once"""
        if not hasattr(self, 'samplers'):
            self.samplers = {}
        if not hasattr(self, 'validators'):
            self.validators = {}

    def register_sampler(self, name: str, func: Callable) -> None:
        """Register a sampler function"""
        if name in self.samplers:
            raise ValueError("Name already registered")
        self.samplers[name] = func
    
    def register_validator(self, name: str, func: Callable) -> None:
        """Register a validator function"""
        if name in self.validators:
            raise ValueError("Name already registered")
        self.validators[name] = func
    
    def get_sampler(self, name: str) -> Callable:
        """Retrieve a sampler function by name"""
        if name not in self.samplers:
            raise ValueError("Sampler not found")
        return self.samplers[name]
    
    def get_validator(self, name: str) -> Callable:
        """Retrieve a validator function by name"""
        if name not in self.validators:
            raise ValueError("Validator not found")
        return self.validators[name]

registry = FunctionRegistry()    

class FunctionDecorator:
    """Class decorator to register function in FunctionRegistry"""

    def __init__(self, function_type: str):
        """Initialize with the type: 'sampler' or 'validator'"""
        if function_type not in ('sampler', 'validator'):
            raise ValueError("Invalid function type")
        self.function_type = function_type
    
    def __call__(self, func: Callable):
        """When the decorator is applied, register function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        func_name = func.__name__

        if self.function_type == 'sampler':
            registry.register_sampler(func_name, wrapper)
        elif self.function_type == 'validator':
            registry.register_validator(func_name, wrapper)
        
        return wrapper
    



