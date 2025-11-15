# ComfyUI Pipeline Schema and Builder System

This module provides a Pydantic-based schema and builder system for defining, validating, and manipulating ComfyUI pipelines in a type-safe manner.

## Overview

The traditional approach in hordelib uses dictionaries and manual validation through `PAYLOAD_SCHEMA`. This new system replaces that with:

- **Type-safe Pydantic models** for all pipeline structures
- **Automatic validation** with clear error messages
- **IDE autocomplete** and type checking support
- **Builder pattern** for constructing pipelines programmatically
- **Improved maintainability** through explicit schemas

## Architecture

### Core Components

#### 1. Schema Module (`schema.py`)

Defines Pydantic models for:

- **ComfyUI Node Structure**
  - `NodeConnection`: Type-safe node connections
  - `NodeMeta`: Node metadata
  - `ComfyNode`: Individual node definition
  - `Pipeline`: Complete pipeline/workflow

- **Node Input Models**
  - `KSamplerInputs`, `CheckpointLoaderInputs`, `LoraLoaderInputs`, etc.
  - Each with field validation and type hints

- **Payload Models**
  - `GenerationPayload`: Main request payload (replaces PAYLOAD_SCHEMA)
  - `LoraPayload`, `TextualInversionPayload`, etc.
  - All with enums for valid values

#### 2. Builder Module (`builder.py`)

Provides builder classes:

- **`PipelineBuilder`**: Fluent API for building/modifying pipelines
- **`LoraChainBuilder`**: Dynamic LoRA chain injection
- **`PayloadToPipelineMapper`**: Apply payloads to pipelines

## Key Features

### Type Safety

```python
from hordelib.pipeline import GenerationPayload, SamplerName

# IDE provides autocomplete for all fields
payload = GenerationPayload(
    prompt="a beautiful landscape",
    sampler_name=SamplerName.K_EULER,  # Enum prevents typos
    cfg_scale=7.5,  # Type-checked as float
    width=512,  # Validated and auto-corrected to multiples of 64
)
```

### Automatic Validation

```python
# Dimensions automatically corrected
payload = GenerationPayload(
    width=700,   # Not divisible by 64
    height=530,  # Not divisible by 64
)
# Result: width=704, height=576 (rounded up)

# Range validation
payload = GenerationPayload(
    cfg_scale=150.0  # Error! Max is 100.0
)
# Raises: ValidationError
```

### Pipeline Building

```python
from hordelib.pipeline.builder import PipelineBuilder

# Build a pipeline programmatically
builder = PipelineBuilder()

# Add nodes with type-safe methods
model_id = builder.add_checkpoint_loader("model_loader", "Deliberate.ckpt")
latent_id = builder.add_empty_latent("latent", width=512, height=512)
sampler_id = builder.add_ksampler("sampler", seed=42, steps=20, cfg=8.0)

# Connect nodes
builder.connect(sampler_id, "model", model_id, 0)
builder.connect(sampler_id, "latent_image", latent_id, 0)

# Get the pipeline
pipeline = builder.build()
```

### Loading Existing Pipelines

```python
# Load from file
builder = PipelineBuilder.from_file("pipeline_stable_diffusion.json")

# Modify values
builder.set_inputs_by_dot_notation(
    sampler__cfg=9.0,
    sampler__steps=25,
    prompt__text="new prompt"
)

# Save modified pipeline
builder.save("modified_pipeline.json")
```

### Dynamic LoRA Chains

```python
from hordelib.pipeline.builder import LoraChainBuilder

# Create base pipeline
builder = PipelineBuilder.from_file("pipeline.json")

# Build LoRA chain
lora_chain = LoraChainBuilder(builder)
lora_chain.add_lora("detail.safetensors", model_strength=0.8, clip_strength=0.8)
lora_chain.add_lora("style.safetensors", model_strength=1.0, clip_strength=1.0)

# Connect to pipeline
lora_chain.connect_to_pipeline(
    model_loader_title="model_loader",
    sampler_title="sampler",
    clip_skip_title="clip_skip"
)

# LoRAs are automatically chained: model_loader -> lora_0 -> lora_1 -> sampler
```

### Payload to Pipeline Mapping

```python
from hordelib.pipeline.builder import PayloadToPipelineMapper

# Create validated payload
payload = GenerationPayload(
    prompt="cyberpunk city",
    width=768,
    height=512,
    cfg_scale=7.5,
    ddim_steps=30,
    sampler_name=SamplerName.K_DPMPP_2M
)

# Apply to pipeline
builder = PipelineBuilder.from_file("pipeline.json")
mapper = PayloadToPipelineMapper(payload)
mapper.apply_to_pipeline(builder)

# All values applied with type safety
```

## Migration from Old System

### Old Way (dict-based)

```python
# Manual validation
payload = {
    "sampler_name": "k_euler",
    "cfg_scale": 8.0,
    "width": 512,
    # Easy to make typos, no validation until runtime
}

# Manual schema definition
PAYLOAD_SCHEMA = {
    "sampler_name": {"datatype": str, "values": [...], "default": "k_euler"},
    "cfg_scale": {"datatype": float, "min": 1, "max": 100, "default": 8.0},
    # ...
}

# Manual validation function
validated = self._validate_data_structure(payload, PAYLOAD_SCHEMA)
```

### New Way (Pydantic)

```python
# Automatic validation with type hints
payload = GenerationPayload(
    sampler_name=SamplerName.K_EULER,  # IDE autocomplete!
    cfg_scale=8.0,  # Type-checked
    width=512,  # Validated
)
# All validation happens automatically in __init__

# Schema is part of the model
class GenerationPayload(BaseModel):
    sampler_name: SamplerName = Field(default=SamplerName.K_EULER)
    cfg_scale: float = Field(default=8.0, ge=1.0, le=100.0)
    # Type hints + validation in one place
```

## Benefits

### For Developers

1. **IDE Support**: Full autocomplete, go-to-definition, find-references
2. **Type Checking**: Catch errors with mypy/pyright before runtime
3. **Documentation**: Fields are self-documenting with descriptions
4. **Refactoring**: Rename operations work correctly
5. **Validation**: Pydantic provides clear error messages

### For Maintainers

1. **Single Source of Truth**: Schema definition is the model itself
2. **No Sync Issues**: Can't have schema drift between definition and validation
3. **Extensibility**: Easy to add new fields or node types
4. **Testing**: Models can be easily tested in isolation
5. **Migration Path**: Can coexist with old system during transition

## Usage Examples

See `examples.py` for comprehensive examples covering:

1. Basic payload validation
2. Payloads with LoRAs
3. Building pipelines from scratch
4. Loading and modifying existing pipelines
5. Dynamic LoRA chains
6. Payload to pipeline mapping
7. Type safety benefits
8. Working with node connections

Run examples:

```bash
python -m hordelib.pipeline.examples
```

## Node Type Reference

### Supported Node Types

All nodes inherit from `ComfyNode` with specific input models:

- **Sampling**: `KSamplerInputs`
- **Models**: `CheckpointLoaderInputs`, `LoraLoaderInputs`
- **Images**: `EmptyLatentImageInputs`, `LoadImageInputs`, `SaveImageInputs`
- **VAE**: `VAEDecodeInputs`, `VAEEncodeInputs`
- **CLIP**: `CLIPTextEncodeInputs`
- **ControlNet**: `ControlNetApplyInputs`

### Adding New Node Types

```python
class MyCustomNodeInputs(NodeInputs):
    """Inputs for my custom node."""

    my_param: str = Field(..., description="Parameter description")
    my_number: float = Field(default=1.0, ge=0.0, le=10.0)
    my_connection: NodeConnection = Field(...)

    @field_validator("my_param")
    @classmethod
    def validate_my_param(cls, v: str) -> str:
        # Custom validation
        return v.lower()
```

## API Reference

### PipelineBuilder Methods

- `add_node()`: Add generic node
- `add_checkpoint_loader()`: Add model loader
- `add_empty_latent()`: Add latent image node
- `add_ksampler()`: Add sampler node
- `add_clip_text_encode()`: Add prompt encoder
- `add_lora_loader()`: Add LoRA node
- `add_vae_decode()`: Add VAE decoder
- `add_save_image()`: Add image output
- `connect()`: Connect nodes by ID
- `connect_by_title()`: Connect nodes by title
- `set_input()`: Set node input value
- `set_input_by_title()`: Set input by node title
- `set_inputs_by_dot_notation()`: Batch set inputs (old _set() compatible)
- `build()`: Get final pipeline
- `to_comfy_format()`: Export to ComfyUI JSON
- `save()`: Save pipeline to file

### LoraChainBuilder Methods

- `add_lora()`: Add LoRA to chain
- `connect_to_pipeline()`: Wire chain into pipeline

### PayloadToPipelineMapper Methods

- `apply_to_pipeline()`: Apply all payload values to pipeline

## Testing

The schema system includes comprehensive validation:

```python
import pytest
from pydantic import ValidationError
from hordelib.pipeline import GenerationPayload

def test_dimension_validation():
    # Should auto-correct to multiples of 64
    payload = GenerationPayload(width=700, height=530)
    assert payload.width == 704
    assert payload.height == 576

def test_cfg_range():
    # Should enforce min/max
    with pytest.raises(ValidationError):
        GenerationPayload(cfg_scale=150.0)  # Max is 100.0

def test_sampler_enum():
    # Should validate sampler names
    payload = GenerationPayload(sampler_name=SamplerName.K_EULER)
    assert payload.sampler_name == "k_euler"  # Enum value
```

## Performance

Pydantic v2 is highly optimized:

- Validation is implemented in Rust for speed
- Minimal overhead compared to manual validation
- Lazy validation where possible
- Efficient serialization/deserialization

## Future Enhancements

Potential additions:

1. **Pipeline validation**: Ensure all connections are valid
2. **Type inference**: Automatically determine connection compatibility
3. **Pipeline visualization**: Generate diagrams from schemas
4. **More node types**: Add models for all ComfyUI nodes
5. **Migration tools**: Auto-convert old pipelines to new format
6. **JSON Schema export**: Generate OpenAPI-compatible schemas

## Contributing

When adding new features:

1. Define Pydantic models in `schema.py`
2. Add builder methods in `builder.py`
3. Include examples in `examples.py`
4. Update this README
5. Add tests

## License

Same as hordelib project.
