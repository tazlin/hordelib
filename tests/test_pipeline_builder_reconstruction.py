"""Tests for programmatically building pipelines from scratch.

These tests demonstrate that we can reconstruct actual pipeline structures
using only the PipelineBuilder API, verifying the builder provides complete
coverage of pipeline construction capabilities.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_modules():
    """Load required modules without triggering __init__.py imports."""
    import importlib.util

    # Mock PIL.Image
    import types

    pil_module = types.ModuleType("PIL")
    image_module = types.ModuleType("Image")
    image_module.Image = type("Image", (), {})
    pil_module.Image = image_module.Image
    sys.modules["PIL"] = pil_module
    sys.modules["PIL.Image"] = image_module

    # Load schema
    schema_path = Path(__file__).parent.parent / "hordelib" / "pipeline" / "schema.py"
    spec = importlib.util.spec_from_file_location("schema", schema_path)
    schema = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema)

    # Load builder
    builder_path = Path(__file__).parent.parent / "hordelib" / "pipeline" / "builder.py"
    spec = importlib.util.spec_from_file_location("builder", builder_path)
    builder = importlib.util.module_from_spec(spec)

    # Make schema available to builder
    sys.modules["hordelib.pipeline.schema"] = schema
    spec.loader.exec_module(builder)

    return schema, builder


def normalize_for_comparison(data):
    """Normalize data for structural comparison.

    Since programmatically built pipelines will have different node IDs
    (auto-generated vs. original), we compare structure, not exact IDs.
    """
    if isinstance(data, dict):
        # For nodes, extract just the structure we care about
        if "class_type" in data and "inputs" in data:
            return {
                "class_type": data["class_type"],
                "inputs": {
                    k: normalize_for_comparison(v) if not isinstance(v, list) else "connection"
                    for k, v in data.get("inputs", {}).items()
                },
                "_meta": data.get("_meta", {}),
            }
        return {k: normalize_for_comparison(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [normalize_for_comparison(item) for item in data]
    return data


def test_build_basic_stable_diffusion_pipeline():
    """Test building a basic Stable Diffusion pipeline from scratch."""
    print("\n" + "=" * 80)
    print("Test: Build Basic Stable Diffusion Pipeline")
    print("=" * 80)

    schema, builder_module = load_modules()

    # Create a new pipeline
    builder = builder_module.PipelineBuilder()

    # Add checkpoint loader
    model_id = builder.add_checkpoint_loader(
        title="model_loader",
        ckpt_name="Deliberate.ckpt",
    )

    # Add empty latent image
    latent_id = builder.add_empty_latent(
        title="empty_latent_image",
        width=512,
        height=512,
        batch_size=1,
    )

    # Add CLIP skip
    clip_skip_id = builder.add_node(
        title="clip_skip",
        class_type="CLIPSetLastLayer",
        inputs={"stop_at_clip_layer": -1},
    )
    builder.connect(clip_skip_id, "clip", model_id, 1)

    # Add positive prompt
    pos_prompt_id = builder.add_clip_text_encode(
        title="prompt",
        text="dinosaur ",
    )
    builder.connect(pos_prompt_id, "clip", clip_skip_id, 0)

    # Add negative prompt
    neg_prompt_id = builder.add_clip_text_encode(
        title="negative_prompt",
        text="painting, drawing, artwork",
    )
    builder.connect(neg_prompt_id, "clip", clip_skip_id, 0)

    # Add sampler
    sampler_id = builder.add_ksampler(
        title="sampler",
        seed=325847265780417,
        steps=20,
        cfg=8,
        sampler_name="euler",
        scheduler="normal",
        denoise=1,
    )
    builder.connect(sampler_id, "model", model_id, 0)
    builder.connect(sampler_id, "positive", pos_prompt_id, 0)
    builder.connect(sampler_id, "negative", neg_prompt_id, 0)
    builder.connect(sampler_id, "latent_image", latent_id, 0)

    # Add VAE decode
    vae_decode_id = builder.add_vae_decode(title="vae_decode")
    builder.connect(vae_decode_id, "samples", sampler_id, 0)
    builder.connect(vae_decode_id, "vae", model_id, 2)

    # Add save image
    save_id = builder.add_save_image(
        title="output_image",
        filename_prefix="ComfyUI",
    )
    builder.connect(save_id, "images", vae_decode_id, 0)

    # Get the pipeline
    pipeline = builder.build()

    # Verify structure
    print(f"✓ Built pipeline with {len(pipeline.nodes)} nodes")

    # Verify all expected nodes exist
    expected_titles = [
        "model_loader",
        "empty_latent_image",
        "clip_skip",
        "prompt",
        "negative_prompt",
        "sampler",
        "vae_decode",
        "output_image",
    ]

    for title in expected_titles:
        node = pipeline.get_node_by_title(title)
        if node is None:
            print(f"✗ Missing node: {title}")
            return False
        print(f"  ✓ Node '{title}' ({node.class_type})")

    # Verify connections
    sampler = pipeline.get_node_by_title("sampler")
    assert sampler.get_connection("model") is not None, "Sampler missing model connection"
    assert sampler.get_connection("positive") is not None, "Sampler missing positive connection"
    assert sampler.get_connection("negative") is not None, "Sampler missing negative connection"
    assert sampler.get_connection("latent_image") is not None, "Sampler missing latent connection"

    vae_decode = pipeline.get_node_by_title("vae_decode")
    assert vae_decode.get_connection("samples") is not None, "VAE decode missing samples connection"
    assert vae_decode.get_connection("vae") is not None, "VAE decode missing vae connection"

    print("✓ All connections verified")

    # Export and verify it's valid ComfyUI format
    comfy_data = builder.to_comfy_format()
    assert isinstance(comfy_data, dict), "Output should be a dictionary"
    assert len(comfy_data) > 0, "Output should have nodes"

    # Verify each node has required structure
    for node_id, node_data in comfy_data.items():
        assert "class_type" in node_data, f"Node {node_id} missing class_type"
        assert "inputs" in node_data, f"Node {node_id} missing inputs"
        assert "_meta" in node_data, f"Node {node_id} missing _meta"
        assert "title" in node_data["_meta"], f"Node {node_id} _meta missing title"

    print("✓ ComfyUI format valid")

    return True


def test_build_pipeline_with_lora_chain():
    """Test building a pipeline with a LoRA chain."""
    print("\n" + "=" * 80)
    print("Test: Build Pipeline with LoRA Chain")
    print("=" * 80)

    schema, builder_module = load_modules()

    # Create base pipeline
    builder = builder_module.PipelineBuilder()

    # Add basic nodes
    model_id = builder.add_checkpoint_loader("model_loader", "model.ckpt")
    clip_skip_id = builder.add_node("clip_skip", "CLIPSetLastLayer", {"stop_at_clip_layer": -1})
    builder.connect(clip_skip_id, "clip", model_id, 1)

    sampler_id = builder.add_ksampler(
        "sampler",
        seed=42,
        steps=20,
        cfg=8.0,
        sampler_name="euler",
        scheduler="normal",
        denoise=1.0,
    )

    # Build LoRA chain
    lora_chain = builder_module.LoraChainBuilder(builder)

    # Add multiple LoRAs
    lora_chain.add_lora("detail.safetensors", model_strength=0.8, clip_strength=0.8)
    lora_chain.add_lora("style.safetensors", model_strength=1.0, clip_strength=1.0)
    lora_chain.add_lora("lighting.safetensors", model_strength=0.6, clip_strength=0.6)

    print(f"✓ Created LoRA chain with {len(lora_chain.lora_nodes)} nodes")

    # Connect the chain
    lora_chain.connect_to_pipeline(
        model_loader_title="model_loader",
        sampler_title="sampler",
        clip_skip_title="clip_skip",
    )

    # Verify chain structure
    pipeline = builder.build()

    # Check all LoRA nodes exist
    assert pipeline.get_node("lora_0") is not None, "lora_0 should exist"
    assert pipeline.get_node("lora_1") is not None, "lora_1 should exist"
    assert pipeline.get_node("lora_2") is not None, "lora_2 should exist"

    # Check chain connections
    lora_0 = pipeline.get_node("lora_0")
    lora_1 = pipeline.get_node("lora_1")
    lora_2 = pipeline.get_node("lora_2")

    # lora_1 should connect to lora_0
    assert lora_1.get_connection("model").node_id == "lora_0", "lora_1 model should connect to lora_0"
    assert lora_1.get_connection("clip").node_id == "lora_0", "lora_1 clip should connect to lora_0"

    # lora_2 should connect to lora_1
    assert lora_2.get_connection("model").node_id == "lora_1", "lora_2 model should connect to lora_1"
    assert lora_2.get_connection("clip").node_id == "lora_1", "lora_2 clip should connect to lora_1"

    print("✓ LoRA chain properly connected: model_loader -> lora_0 -> lora_1 -> lora_2")

    # Verify LoRA parameters
    assert lora_0.get_input_value("lora_name") == "detail.safetensors"
    assert lora_0.get_input_value("strength_model") == 0.8
    assert lora_1.get_input_value("lora_name") == "style.safetensors"
    assert lora_2.get_input_value("lora_name") == "lighting.safetensors"

    print("✓ LoRA parameters correct")

    return True


def test_load_and_modify_pipeline():
    """Test loading an existing pipeline and modifying it."""
    print("\n" + "=" * 80)
    print("Test: Load and Modify Existing Pipeline")
    print("=" * 80)

    schema, builder_module = load_modules()

    # Load the stable diffusion pipeline
    pipeline_path = Path(__file__).parent.parent / "hordelib" / "pipelines" / "pipeline_stable_diffusion.json"

    with open(pipeline_path) as f:
        original_data = json.load(f)

    # Load into builder
    builder = builder_module.PipelineBuilder.from_dict(original_data)

    print(f"✓ Loaded pipeline with {len(builder.pipeline.nodes)} nodes")

    # Modify using dot notation (old _set() compatible)
    builder.set_inputs_by_dot_notation(**{
        "sampler.cfg": 9.5,
        "sampler.steps": 30,
        "sampler.seed": 999999,
        "empty_latent_image.width": 768,
        "empty_latent_image.height": 512,
        "prompt.text": "a cyberpunk city",
        "negative_prompt.text": "blurry",
    })

    # Verify modifications
    sampler = builder.pipeline.get_node_by_title("sampler")
    assert sampler.get_input_value("cfg") == 9.5, "CFG should be updated"
    assert sampler.get_input_value("steps") == 30, "Steps should be updated"
    assert sampler.get_input_value("seed") == 999999, "Seed should be updated"

    latent = builder.pipeline.get_node_by_title("empty_latent_image")
    assert latent.get_input_value("width") == 768, "Width should be updated"
    assert latent.get_input_value("height") == 512, "Height should be updated"

    prompt = builder.pipeline.get_node_by_title("prompt")
    assert prompt.get_input_value("text") == "a cyberpunk city", "Prompt should be updated"

    neg_prompt = builder.pipeline.get_node_by_title("negative_prompt")
    assert neg_prompt.get_input_value("text") == "blurry", "Negative prompt should be updated"

    print("✓ All modifications applied correctly")

    # Export back to ComfyUI format
    modified_data = builder.to_comfy_format()

    # Verify it still has the same structure (nodes, class_types, etc.)
    assert len(modified_data) == len(original_data), "Node count should match"

    for node_id in original_data.keys():
        assert node_id in modified_data, f"Node {node_id} should still exist"
        assert (
            original_data[node_id]["class_type"] == modified_data[node_id]["class_type"]
        ), f"Node {node_id} class_type should match"

    print("✓ Modified pipeline structure preserved")

    return True


def run_all_tests():
    """Run all builder reconstruction tests."""
    print("\n" + "=" * 80)
    print("Programmatic Pipeline Builder Tests")
    print("=" * 80)

    tests = [
        test_build_basic_stable_diffusion_pipeline,
        test_build_pipeline_with_lora_chain,
        test_load_and_modify_pipeline,
    ]

    results = []
    for test_func in tests:
        try:
            success = test_func()
            results.append((test_func.__name__, success, None))
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            results.append((test_func.__name__, False, f"{e}\n{tb}"))
            print(f"\n✗ Test failed with exception: {e}")
            print(tb)

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed

    print(f"\nTotal: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed tests:")
        for name, success, error in results:
            if not success:
                print(f"  - {name}")
                if error:
                    print(f"    Error: {error[:200]}")

    print()
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
