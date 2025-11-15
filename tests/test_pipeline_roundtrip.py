"""Standalone test script to verify pipeline schema round-trip conversion.

This script tests that we can load all existing pipeline JSON files,
convert them to our Pydantic schema, and convert back to JSON with
perfect fidelity (no data loss).

Run without pytest: python tests/test_pipeline_roundtrip.py
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_schema_module():
    """Load the schema module without triggering __init__.py imports."""
    import importlib.util

    schema_path = Path(__file__).parent.parent / "hordelib" / "pipeline" / "schema.py"
    spec = importlib.util.spec_from_file_location("schema", schema_path)
    schema = importlib.util.module_from_spec(spec)

    # Mock PIL.Image since we don't need it for structure tests
    import types

    pil_module = types.ModuleType("PIL")
    image_module = types.ModuleType("Image")
    image_module.Image = type("Image", (), {})
    pil_module.Image = image_module.Image
    sys.modules["PIL"] = pil_module
    sys.modules["PIL.Image"] = image_module

    # Mock pydantic - wait, we need pydantic. Let me check if it's available.
    try:
        import pydantic

        spec.loader.exec_module(schema)
        return schema
    except ImportError as e:
        print(f"Error: Missing dependency: {e}")
        print("Please install: pip install pydantic pillow")
        return None


def normalize_json(obj):
    """Normalize JSON for comparison (sort keys, handle floats)."""
    if isinstance(obj, dict):
        return {k: normalize_json(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return [normalize_json(item) for item in obj]
    elif isinstance(obj, float):
        # Handle float precision issues
        return round(obj, 10)
    return obj


def test_pipeline_roundtrip(schema, pipeline_path: Path) -> tuple[bool, str]:
    """Test that a pipeline can be loaded and round-tripped perfectly.

    Args:
        schema: The schema module
        pipeline_path: Path to the pipeline JSON file

    Returns:
        (success, message) tuple
    """
    try:
        # Load original JSON
        with open(pipeline_path) as f:
            original_data = json.load(f)

        # Convert to Pipeline schema
        pipeline = schema.Pipeline.from_comfy_format(original_data)

        # Convert back to ComfyUI format
        converted_data = pipeline.to_comfy_format()

        # Normalize both for comparison
        original_norm = normalize_json(original_data)
        converted_norm = normalize_json(converted_data)

        # Compare
        if original_norm == converted_norm:
            return True, "✓ Perfect round-trip"

        # Find differences
        differences = []
        all_keys = set(original_norm.keys()) | set(converted_norm.keys())

        for key in sorted(all_keys):
            if key not in original_norm:
                differences.append(f"  - Node '{key}' added in conversion")
            elif key not in converted_norm:
                differences.append(f"  - Node '{key}' lost in conversion")
            elif original_norm[key] != converted_norm[key]:
                orig_node = original_norm[key]
                conv_node = converted_norm[key]

                # Check class_type
                if orig_node.get("class_type") != conv_node.get("class_type"):
                    differences.append(
                        f"  - Node '{key}' class_type mismatch: "
                        f"{orig_node.get('class_type')} -> {conv_node.get('class_type')}"
                    )

                # Check inputs
                if orig_node.get("inputs") != conv_node.get("inputs"):
                    orig_inputs = orig_node.get("inputs", {})
                    conv_inputs = conv_node.get("inputs", {})
                    all_input_keys = set(orig_inputs.keys()) | set(conv_inputs.keys())

                    for input_key in sorted(all_input_keys):
                        if input_key not in orig_inputs:
                            differences.append(
                                f"  - Node '{key}' input '{input_key}' added: {conv_inputs[input_key]}"
                            )
                        elif input_key not in conv_inputs:
                            differences.append(
                                f"  - Node '{key}' input '{input_key}' lost: {orig_inputs[input_key]}"
                            )
                        elif orig_inputs[input_key] != conv_inputs[input_key]:
                            differences.append(
                                f"  - Node '{key}' input '{input_key}' changed: "
                                f"{orig_inputs[input_key]} -> {conv_inputs[input_key]}"
                            )

                # Check _meta
                if orig_node.get("_meta") != conv_node.get("_meta"):
                    differences.append(
                        f"  - Node '{key}' _meta mismatch: "
                        f"{orig_node.get('_meta')} -> {conv_node.get('_meta')}"
                    )

        if differences:
            diff_msg = "\n".join(differences)
            return False, f"✗ Round-trip mismatch:\n{diff_msg}"

        return False, "✗ Data mismatch (couldn't identify specific differences)"

    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        return False, f"✗ Exception: {e}\n{tb}"


def test_all_pipelines():
    """Test all pipeline files."""
    print("=" * 80)
    print("Pipeline Round-Trip Conversion Tests")
    print("=" * 80)

    # Load schema
    print("\nLoading schema module...")
    schema = load_schema_module()
    if schema is None:
        return False

    print("✓ Schema loaded successfully\n")

    # Find all pipeline files
    pipeline_dir = Path(__file__).parent.parent / "hordelib" / "pipelines"
    pipeline_files = sorted(pipeline_dir.glob("*.json"))

    if not pipeline_files:
        print(f"✗ No pipeline files found in {pipeline_dir}")
        return False

    print(f"Found {len(pipeline_files)} pipeline files\n")

    # Test each pipeline
    results = []
    for pipeline_path in pipeline_files:
        pipeline_name = pipeline_path.name
        print(f"Testing: {pipeline_name}")

        success, message = test_pipeline_roundtrip(schema, pipeline_path)
        results.append((pipeline_name, success, message))

        # Print result
        print(f"  {message}")
        print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed

    print(f"\nTotal: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed tests:")
        for name, success, message in results:
            if not success:
                print(f"  - {name}")

    print()
    return failed == 0


if __name__ == "__main__":
    success = test_all_pipelines()
    sys.exit(0 if success else 1)
