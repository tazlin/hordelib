"""Examples demonstrating the use of Pydantic pipeline schemas and builders.

This module contains comprehensive examples showing how to use the new
type-safe pipeline schema and builder system.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

from hordelib.pipeline import (
    GenerationPayload,
    LoraPayload,
    SamplerName,
    SchedulerType,
)
from hordelib.pipeline.builder import (
    LoraChainBuilder,
    PayloadToPipelineMapper,
    PipelineBuilder,
)


def example_1_basic_payload_validation():
    """Example 1: Basic payload validation.

    Demonstrates how to create and validate a generation payload
    with automatic type checking and constraint validation.
    """
    print("=== Example 1: Basic Payload Validation ===\n")

    # Create a basic payload - all fields have defaults
    payload = GenerationPayload()
    print(f"Default payload:\n{payload.model_dump_json(indent=2)}\n")

    # Create a custom payload with validation
    payload = GenerationPayload(
        prompt="a beautiful landscape, mountains, sunset",
        negative_prompt="blurry, low quality",
        width=768,  # Will be rounded to 768 (already divisible by 64)
        height=512,  # Will be rounded to 512 (already divisible by 64)
        cfg_scale=7.5,
        ddim_steps=25,
        sampler_name=SamplerName.K_EULER_A,
        scheduler=SchedulerType.KARRAS,
        seed=42,
    )

    print(f"Custom payload:\n{payload.model_dump_json(indent=2)}\n")

    # Try invalid dimensions - they'll be auto-corrected
    payload_corrected = GenerationPayload(
        width=700,  # Not divisible by 64, will become 704
        height=530,  # Not divisible by 64, will become 576
    )

    print(f"Auto-corrected dimensions:")
    print(f"  Original: 700x530")
    print(f"  Corrected: {payload_corrected.width}x{payload_corrected.height}\n")


def example_2_payload_with_loras():
    """Example 2: Payload with LoRAs.

    Shows how to add LoRAs to a generation request with proper validation.
    """
    print("=== Example 2: Payload with LoRAs ===\n")

    # Create LoRA specifications
    loras = [
        LoraPayload(
            name="add_detail.safetensors",
            model=0.8,
            clip=0.8,
            inject_trigger="any",
        ),
        LoraPayload(
            name="studio_lighting.safetensors",
            model=0.6,
            clip=0.6,
        ),
    ]

    # Create payload with LoRAs
    payload = GenerationPayload(
        prompt="a portrait photo",
        loras=loras,
        ddim_steps=30,
        cfg_scale=8.0,
    )

    print(f"Payload with LoRAs:\n{payload.model_dump_json(indent=2)}\n")

    # Validate LoRA strengths - Pydantic ensures they're in range [-10, 10]
    try:
        invalid_lora = LoraPayload(
            name="test.safetensors",
            model=15.0,  # Would be clamped or raise error
        )
    except ValueError as e:
        print(f"Validation error (expected): {e}\n")


def example_3_building_pipeline_from_scratch():
    """Example 3: Building a pipeline from scratch.

    Demonstrates how to programmatically build a ComfyUI pipeline
    using the type-safe builder API.
    """
    print("=== Example 3: Building Pipeline from Scratch ===\n")

    # Create a new pipeline builder
    builder = PipelineBuilder()

    # Add nodes
    model_loader_id = builder.add_checkpoint_loader(
        title="model_loader",
        ckpt_name="Deliberate.ckpt",
    )

    latent_id = builder.add_empty_latent(
        title="empty_latent_image",
        width=512,
        height=512,
        batch_size=1,
    )

    sampler_id = builder.add_ksampler(
        title="sampler",
        seed=12345,
        steps=20,
        cfg=8.0,
        sampler_name="euler",
        scheduler="normal",
        denoise=1.0,
    )

    vae_decode_id = builder.add_vae_decode(title="vae_decode")

    save_id = builder.add_save_image(
        title="output_image",
        filename_prefix="ComfyUI",
    )

    # Connect nodes
    builder.connect(sampler_id, "model", model_loader_id, 0)
    builder.connect(sampler_id, "latent_image", latent_id, 0)

    builder.connect(vae_decode_id, "samples", sampler_id, 0)
    builder.connect(vae_decode_id, "vae", model_loader_id, 2)

    builder.connect(save_id, "images", vae_decode_id, 0)

    # Get the pipeline
    pipeline = builder.build()

    print(f"Built pipeline with {len(pipeline.nodes)} nodes")
    print(f"Node titles: {[n._meta.title for n in pipeline.nodes.values()]}\n")

    # Export to ComfyUI format
    comfy_format = builder.to_comfy_format()
    print(f"Pipeline in ComfyUI format (first node):")
    first_node_id = list(comfy_format.keys())[0]
    print(f"  Node {first_node_id}: {comfy_format[first_node_id]}\n")


def example_4_loading_and_modifying_pipeline():
    """Example 4: Loading and modifying an existing pipeline.

    Shows how to load a pipeline from JSON and modify it programmatically.
    """
    print("=== Example 4: Loading and Modifying Pipeline ===\n")

    # Note: This would load an actual pipeline file
    # builder = PipelineBuilder.from_file("pipeline_stable_diffusion.json")

    # For this example, create a minimal pipeline
    builder = PipelineBuilder()
    builder.add_checkpoint_loader("model_loader", "model.ckpt")
    builder.add_empty_latent("empty_latent_image", width=512, height=512)
    builder.add_ksampler("sampler", seed=42, steps=20, cfg=8.0)

    # Modify values using dot notation (compatible with old _set() method)
    builder.set_inputs_by_dot_notation(
        sampler__cfg=9.0,
        sampler__steps=25,
        empty_latent_image__width=768,
        empty_latent_image__height=768,
    )

    # Or modify by title directly
    builder.set_input_by_title("sampler", "seed", 99999)

    pipeline = builder.build()
    sampler = pipeline.get_node_by_title("sampler")

    print(f"Modified sampler inputs:")
    print(f"  CFG: {sampler.get_input_value('cfg')}")
    print(f"  Steps: {sampler.get_input_value('steps')}")
    print(f"  Seed: {sampler.get_input_value('seed')}\n")


def example_5_dynamic_lora_chain():
    """Example 5: Dynamic LoRA chain injection.

    Demonstrates the LoRA chain builder which replicates the dynamic
    LoRA injection behavior from horde.py.
    """
    print("=== Example 5: Dynamic LoRA Chain ===\n")

    # Create a base pipeline
    builder = PipelineBuilder()
    builder.add_checkpoint_loader("model_loader", "model.ckpt", node_id="4")
    builder.add_ksampler("sampler", seed=42, steps=20, cfg=8.0, node_id="3")
    builder.add_node("clip_skip", "CLIPSetLastLayer", {"stop_at_clip_layer": -1}, node_id="10")

    # Create a LoRA chain
    lora_chain = LoraChainBuilder(builder)

    # Add multiple LoRAs - they'll be automatically chained
    lora_chain.add_lora("detail_lora.safetensors", model_strength=0.8, clip_strength=0.8)
    lora_chain.add_lora("style_lora.safetensors", model_strength=1.0, clip_strength=1.0)
    lora_chain.add_lora("lighting_lora.safetensors", model_strength=0.6, clip_strength=0.6)

    # Connect the chain to the pipeline
    lora_chain.connect_to_pipeline(
        model_loader_title="model_loader",
        sampler_title="sampler",
        clip_skip_title="clip_skip",
    )

    pipeline = builder.build()

    print(f"Pipeline with LoRA chain:")
    print(f"  Total nodes: {len(pipeline.nodes)}")
    print(f"  LoRA nodes: {len(lora_chain.lora_nodes)}")
    print(f"  LoRA chain: {' -> '.join(lora_chain.lora_nodes)}\n")

    # Check connections
    lora_0 = pipeline.get_node("lora_0")
    print(f"  lora_0 model connection: {lora_0.get_connection('model')}")
    print(f"  lora_0 clip connection: {lora_0.get_connection('clip')}\n")


def example_6_payload_to_pipeline_mapping():
    """Example 6: Applying payload to pipeline.

    Shows how to use PayloadToPipelineMapper to apply a validated
    payload to a pipeline.
    """
    print("=== Example 6: Payload to Pipeline Mapping ===\n")

    # Create a payload
    payload = GenerationPayload(
        prompt="a cyberpunk city at night",
        negative_prompt="blurry, bad quality",
        width=768,
        height=512,
        cfg_scale=7.5,
        ddim_steps=30,
        sampler_name=SamplerName.K_DPMPP_2M,
        scheduler=SchedulerType.KARRAS,
        clip_skip=2,
        seed=424242,
    )

    # Create a pipeline with the standard nodes
    builder = PipelineBuilder()
    builder.add_checkpoint_loader("model_loader", "model.ckpt")
    builder.add_empty_latent("empty_latent_image")
    builder.add_ksampler("sampler", seed=0, steps=0, cfg=0.0)
    builder.add_clip_text_encode("prompt", "")
    builder.add_clip_text_encode("negative_prompt", "")
    builder.add_node("clip_skip", "CLIPSetLastLayer", {"stop_at_clip_layer": -1})

    # Apply payload to pipeline
    mapper = PayloadToPipelineMapper(payload)
    mapper.apply_to_pipeline(builder)

    # Verify values were applied
    pipeline = builder.build()
    sampler = pipeline.get_node_by_title("sampler")
    latent = pipeline.get_node_by_title("empty_latent_image")
    prompt_node = pipeline.get_node_by_title("prompt")
    clip_skip = pipeline.get_node_by_title("clip_skip")

    print("Payload applied to pipeline:")
    print(f"  Sampler CFG: {sampler.get_input_value('cfg')}")
    print(f"  Sampler steps: {sampler.get_input_value('steps')}")
    print(f"  Sampler name: {sampler.get_input_value('sampler_name')}")
    print(f"  Dimensions: {latent.get_input_value('width')}x{latent.get_input_value('height')}")
    print(f"  Prompt: {prompt_node.get_input_value('text')}")
    print(f"  Clip skip: {clip_skip.get_input_value('stop_at_clip_layer')}\n")


def example_7_type_safety_benefits():
    """Example 7: Type safety benefits.

    Demonstrates the benefits of using Pydantic models for type safety.
    """
    print("=== Example 7: Type Safety Benefits ===\n")

    # IDE will provide autocomplete for all fields
    payload = GenerationPayload(
        prompt="test",  # IDE knows this is a string
        width=512,  # IDE knows this is an int
        cfg_scale=7.5,  # IDE knows this is a float
        sampler_name=SamplerName.K_EULER,  # IDE shows all valid samplers
    )

    print("Type safety features:")
    print("  ✓ IDE autocomplete for all fields")
    print("  ✓ Type checking catches errors before runtime")
    print("  ✓ Validation ensures constraints are met")
    print("  ✓ Enums prevent invalid values")
    print("  ✓ Documentation via field descriptions\n")

    # Try to create invalid payload
    try:
        # This will fail validation
        bad_payload = GenerationPayload(
            cfg_scale=150.0,  # Max is 100.0
        )
    except ValueError as e:
        print(f"Validation caught error: {e}\n")

    # Inspect schema
    print("Available samplers:")
    for sampler in SamplerName:
        print(f"  - {sampler.value}")


def example_8_working_with_connections():
    """Example 8: Working with node connections.

    Shows how to work with NodeConnection objects for type-safe
    pipeline wiring.
    """
    print("\n=== Example 8: Working with Node Connections ===\n")

    builder = PipelineBuilder()

    # Add nodes
    model_id = builder.add_checkpoint_loader("model_loader", "model.ckpt")
    sampler_id = builder.add_ksampler("sampler", seed=42, steps=20, cfg=8.0)

    # Connect using helper methods
    builder.connect(sampler_id, "model", model_id, 0)

    # Or connect by title (more readable)
    builder.connect_by_title("sampler", "model", "model_loader", 0)

    # Inspect connections
    pipeline = builder.build()
    sampler = pipeline.get_node_by_title("sampler")
    model_connection = sampler.get_connection("model")

    if model_connection:
        print(f"Sampler model connection:")
        print(f"  Connected to node: {model_connection.node_id}")
        print(f"  Output index: {model_connection.output_index}")
        print(f"  ComfyUI format: {model_connection.to_comfy_format()}\n")


def run_all_examples():
    """Run all examples."""
    example_1_basic_payload_validation()
    example_2_payload_with_loras()
    example_3_building_pipeline_from_scratch()
    example_4_loading_and_modifying_pipeline()
    example_5_dynamic_lora_chain()
    example_6_payload_to_pipeline_mapping()
    example_7_type_safety_benefits()
    example_8_working_with_connections()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples()
