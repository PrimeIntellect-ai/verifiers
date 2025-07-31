# Multi-Turn Extension for TRL Integration

This document describes the multi-turn extension that enables backwards-compatible multi-turn rollout support in the VerifiersGRPOTrainer while using TRL as the base training framework.

## Overview

The multi-turn extension provides:

1. **Automatic Environment Detection**: Automatically detects whether an environment requires single-turn or multi-turn handling
2. **Backwards Compatibility**: Existing single-turn code continues to work without changes
3. **Full Multi-Turn Support**: Complex multi-turn conversations with tool calls, environment responses, and state management
4. **TRL Integration**: Uses TRL's GRPOTrainer as the base while preserving verifiers' unique capabilities

## Architecture

### Key Components

1. **MultiTurnMixin** (`verifiers/trainers/multi_turn_mixin.py`): Core mixin that provides multi-turn detection and handling capabilities
2. **VerifiersGRPOTrainer** (`verifiers/trainers/verifiers_grpo_trainer.py`): Extended trainer that inherits from both MultiTurnMixin and TRL's GRPOTrainer
3. **Environment Detection Logic**: Smart detection of single-turn vs multi-turn environments

### Detection Logic

The system automatically detects multi-turn environments using the following criteria:

```python
def is_multi_turn_environment(self, env: Environment) -> bool:
    # Special case: SingleTurnEnv is treated as single-turn
    if type(env).__name__ == 'SingleTurnEnv':
        return False
        
    # Check for tools (usually indicates multi-turn)
    if hasattr(env, 'oai_tools') and env.oai_tools:
        return True
        
    # Check for actual multi-turn classes
    if isinstance(env, MultiTurnEnv) and type(env).__name__ != 'SingleTurnEnv':
        return True
        
    return False
```

## Usage Examples

### Single-Turn Environment (Backwards Compatible)

```python
from verifiers import SingleTurnEnv, Rubric
from verifiers.trainers import VerifiersGRPOTrainer, VerifiersGRPOConfig

# Create single-turn environment (works exactly as before)
env = SingleTurnEnv(dataset=your_dataset, rubric=your_rubric)

# Create trainer - automatically detects single-turn
trainer = VerifiersGRPOTrainer(
    model="your-model",
    env=env,
    args=VerifiersGRPOConfig("single-turn-run")
)

# Training works exactly as before
trainer.train()
```

### Multi-Turn Environment (New Capability)

```python
from verifiers.envs import ToolEnv
from verifiers.trainers import VerifiersGRPOTrainer, VerifiersGRPOConfig

# Create multi-turn environment with tools
def my_tool(query: str) -> str:
    return f"Result for {query}"

env = ToolEnv(
    tools=[my_tool],
    dataset=your_dataset,
    system_prompt="You can use tools to help answer questions."
)

# Create trainer - automatically detects multi-turn
trainer = VerifiersGRPOTrainer(
    model="your-model", 
    env=env,
    args=VerifiersGRPOConfig("multi-turn-run")
)

# Training automatically uses multi-turn rollouts
trainer.train()
```

### Custom Multi-Turn Environment

```python
from verifiers.envs import MultiTurnEnv

class CustomMultiTurnEnv(MultiTurnEnv):
    def is_completed(self, messages, state, **kwargs):
        # Custom completion logic
        return len(messages) >= 5
    
    def env_response(self, messages, state, **kwargs):
        # Custom environment response
        response = {"role": "user", "content": "Environment feedback"}
        return [response], state

# Automatically detected as multi-turn
env = CustomMultiTurnEnv(dataset=your_dataset)
trainer = VerifiersGRPOTrainer(model="your-model", env=env)
```

## Technical Details

### Generation Process

#### Single-Turn Environments
- Uses TRL's standard generation pipeline
- Simple prompt → completion mapping
- Direct reward computation

#### Multi-Turn Environments  
- Uses verifiers' full rollout system
- Generates complete conversations with environment interactions
- Preserves conversation context for accurate reward computation
- Converts final conversations to TRL-compatible format

### Dataset Handling

#### Single-Turn
- Uses static datasets from environment
- Standard TRL dataloader behavior

#### Multi-Turn
- Generates datasets dynamically during training
- Each training step creates fresh multi-turn conversations
- Preserves full conversation history for reward computation

### Reward Computation

#### Single-Turn
- Uses EnvironmentRewardAdapter to convert Environment rubrics to TRL reward functions
- Simple string-based reward computation

#### Multi-Turn
- Uses full conversation context including:
  - Original prompts (message lists)
  - Complete conversation history
  - Environment state information
  - Tool call results
- Preserves verifiers' rich reward computation capabilities

## Configuration

### VerifiersGRPOConfig Extensions

```python
config = VerifiersGRPOConfig(
    "run-name",
    # Standard TRL parameters
    learning_rate=1e-6,
    per_device_train_batch_size=8,
    
    # Verifiers-specific parameters
    num_batches_ahead=1,           # Async batch generation
    enable_async_generation=True,   # Enable async capabilities
    async_timeout=300,             # Timeout for generation
)
```

## Backwards Compatibility

The extension maintains full backwards compatibility:

1. **Existing Code**: All existing verifiers code continues to work unchanged
2. **SingleTurnEnv**: Specifically detected and handled as single-turn despite inheriting from MultiTurnEnv
3. **API Compatibility**: All existing methods and interfaces preserved
4. **Configuration**: Existing configurations continue to work

## Testing

Comprehensive test coverage includes:

1. **Detection Tests**: Verify correct single-turn vs multi-turn detection
2. **Generation Tests**: Test both single-turn and multi-turn generation paths
3. **Reward Tests**: Verify reward computation for both scenarios
4. **Backwards Compatibility**: Ensure existing code continues to work
5. **Error Handling**: Test fallback behaviors when multi-turn fails

## Performance Considerations

### Single-Turn Performance
- No performance impact on existing single-turn workflows
- Direct routing to TRL's optimized pipelines

### Multi-Turn Performance
- Dynamic dataset generation adds some overhead
- Rich conversation context requires more memory
- Async batch generation can improve throughput
- Environment rollouts may be slower than simple generation

## Migration Guide

### From Original GRPOTrainer

```python
# Before (using original GRPOTrainer)
from verifiers.trainers import GRPOTrainer

trainer = GRPOTrainer(model=model, env=env)

# After (using VerifiersGRPOTrainer with TRL base)
from verifiers.trainers import VerifiersGRPOTrainer

trainer = VerifiersGRPOTrainer(model=model, env=env)
# Everything else works the same!
```

### Adding Multi-Turn Support

If you have an existing single-turn environment and want to add multi-turn capabilities:

1. Inherit from `MultiTurnEnv` instead of `Environment`
2. Implement `is_completed()` and `env_response()` methods
3. The trainer will automatically detect and handle multi-turn behavior

## Limitations

1. **Multi-Turn Complexity**: Multi-turn rollouts are more complex and slower than single-turn generation
2. **Memory Usage**: Full conversation histories require more memory
3. **Dynamic Datasets**: Multi-turn environments generate data dynamically, which may impact reproducibility
4. **TRL Compatibility**: Some advanced TRL features may not be fully compatible with multi-turn workflows

## Future Enhancements

Potential future improvements:

1. **Caching**: Cache multi-turn conversations to improve performance
2. **Batched Multi-Turn**: More efficient batched multi-turn generation
3. **Advanced TRL Integration**: Support for more TRL features in multi-turn mode
4. **Streaming**: Support for streaming multi-turn generation