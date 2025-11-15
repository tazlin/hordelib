"""Tests for the Pydantic pipeline schema and builder system."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from hordelib.pipeline import (
    ComfyNode,
    GenerationPayload,
    LoraPayload,
    NodeConnection,
    NodeMeta,
    Pipeline,
    SamplerName,
    SchedulerType,
)
from hordelib.pipeline.builder import LoraChainBuilder, PayloadToPipelineMapper, PipelineBuilder


class TestNodeConnection:
    """Tests for NodeConnection model."""

    def test_create_connection(self):
        """Test creating a node connection."""
        conn = NodeConnection(node_id="4", output_index=0)
        assert conn.node_id == "4"
        assert conn.output_index == 0

    def test_to_comfy_format(self):
        """Test converting to ComfyUI format."""
        conn = NodeConnection(node_id="4", output_index=1)
        assert conn.to_comfy_format() == ["4", 1]

    def test_from_comfy_format(self):
        """Test creating from ComfyUI format."""
        conn = NodeConnection.from_comfy_format(["4", 0])
        assert conn.node_id == "4"
        assert conn.output_index == 0

    def test_invalid_comfy_format(self):
        """Test that invalid formats raise errors."""
        with pytest.raises(ValueError):
            NodeConnection.from_comfy_format(["4"])  # Missing output_index
        with pytest.raises(ValueError):
            NodeConnection.from_comfy_format("4")  # Not a list


class TestComfyNode:
    """Tests for ComfyNode model."""

    def test_create_node(self):
        """Test creating a node."""
        node = ComfyNode(
            inputs={"seed": 42, "steps": 20},
            class_type="KSampler",
            meta=NodeMeta(title="sampler"),
        )
        assert node.class_type == "KSampler"
        assert node.meta.title == "sampler"
        assert node.get_input_value("seed") == 42

    def test_set_input_value(self):
        """Test setting input values."""
        node = ComfyNode(
            inputs={},
            class_type="KSampler",
            meta=NodeMeta(title="sampler"),
        )
        node.set_input_value("cfg", 7.5)
        assert node.get_input_value("cfg") == 7.5

    def test_connect_input(self):
        """Test connecting inputs."""
        node = ComfyNode(
            inputs={},
            class_type="KSampler",
            meta=NodeMeta(title="sampler"),
        )
        node.connect_input("model", "4", 0)
        conn = node.get_connection("model")
        assert conn is not None
        assert conn.node_id == "4"
        assert conn.output_index == 0

    def test_parse_connections_from_dict(self):
        """Test that connections are parsed from dict format."""
        node = ComfyNode(
            inputs={
                "model": ["4", 0],  # ComfyUI format
                "cfg": 7.5,  # Regular value
            },
            class_type="KSampler",
            meta=NodeMeta(title="sampler"),
        )
        conn = node.get_connection("model")
        assert conn is not None
        assert conn.node_id == "4"
        assert node.get_input_value("cfg") == 7.5


class TestPipeline:
    """Tests for Pipeline model."""

    def test_create_pipeline(self):
        """Test creating a pipeline."""
        pipeline = Pipeline()
        assert len(pipeline.nodes) == 0

    def test_add_node(self):
        """Test adding nodes."""
        pipeline = Pipeline()
        node = ComfyNode(
            inputs={},
            class_type="KSampler",
            meta=NodeMeta(title="sampler"),
        )
        pipeline.add_node("3", node)
        assert len(pipeline.nodes) == 1
        assert pipeline.get_node("3") == node

    def test_get_node_by_title(self):
        """Test getting nodes by title."""
        pipeline = Pipeline()
        node = ComfyNode(
            inputs={},
            class_type="KSampler",
            meta=NodeMeta(title="my_sampler"),
        )
        pipeline.add_node("3", node)
        found = pipeline.get_node_by_title("my_sampler")
        assert found == node

    def test_remove_node(self):
        """Test removing nodes."""
        pipeline = Pipeline()
        node = ComfyNode(
            inputs={},
            class_type="KSampler",
            meta=NodeMeta(title="sampler"),
        )
        pipeline.add_node("3", node)
        pipeline.remove_node("3")
        assert len(pipeline.nodes) == 0

    def test_to_comfy_format(self):
        """Test exporting to ComfyUI format."""
        pipeline = Pipeline()
        node = ComfyNode(
            inputs={"seed": 42, "model": NodeConnection(node_id="4", output_index=0)},
            class_type="KSampler",
            meta=NodeMeta(title="sampler"),
        )
        pipeline.add_node("3", node)

        comfy_data = pipeline.to_comfy_format()
        assert "3" in comfy_data
        assert comfy_data["3"]["class_type"] == "KSampler"
        assert comfy_data["3"]["_meta"]["title"] == "sampler"
        assert comfy_data["3"]["inputs"]["seed"] == 42
        assert comfy_data["3"]["inputs"]["model"] == ["4", 0]

    def test_from_comfy_format(self):
        """Test loading from ComfyUI format."""
        comfy_data = {
            "3": {
                "inputs": {"seed": 42, "model": ["4", 0]},
                "class_type": "KSampler",
                "_meta": {"title": "sampler"},
            },
        }
        pipeline = Pipeline.from_comfy_format(comfy_data)
        assert len(pipeline.nodes) == 1
        node = pipeline.get_node("3")
        assert node is not None
        assert node.class_type == "KSampler"
        assert node.get_input_value("seed") == 42
        conn = node.get_connection("model")
        assert conn is not None
        assert conn.node_id == "4"


class TestGenerationPayload:
    """Tests for GenerationPayload model."""

    def test_default_payload(self):
        """Test creating payload with defaults."""
        payload = GenerationPayload()
        assert payload.sampler_name == SamplerName.K_EULER
        assert payload.cfg_scale == 8.0
        assert payload.width == 512
        assert payload.height == 512

    def test_dimension_validation(self):
        """Test that dimensions are rounded to multiples of 64."""
        payload = GenerationPayload(width=700, height=530)
        assert payload.width == 704  # Rounded up to nearest 64
        assert payload.height == 576  # Rounded up to nearest 64

    def test_cfg_range_validation(self):
        """Test CFG scale range validation."""
        # Valid range
        payload = GenerationPayload(cfg_scale=7.5)
        assert payload.cfg_scale == 7.5

        # Out of range should fail
        with pytest.raises(ValidationError):
            GenerationPayload(cfg_scale=150.0)

    def test_steps_validation(self):
        """Test steps range validation."""
        payload = GenerationPayload(ddim_steps=30)
        assert payload.ddim_steps == 30

        with pytest.raises(ValidationError):
            GenerationPayload(ddim_steps=0)  # Min is 1

        with pytest.raises(ValidationError):
            GenerationPayload(ddim_steps=600)  # Max is 500

    def test_sampler_enum(self):
        """Test sampler name enum."""
        payload = GenerationPayload(sampler_name=SamplerName.K_EULER_A)
        assert payload.sampler_name == "k_euler_a"

    def test_with_loras(self):
        """Test payload with LoRAs."""
        loras = [
            LoraPayload(name="test1.safetensors", model=0.8, clip=0.8),
            LoraPayload(name="test2.safetensors", model=1.0, clip=1.0),
        ]
        payload = GenerationPayload(loras=loras)
        assert len(payload.loras) == 2
        assert payload.loras[0].name == "test1.safetensors"
        assert payload.loras[0].model == 0.8


class TestLoraPayload:
    """Tests for LoraPayload model."""

    def test_create_lora(self):
        """Test creating a LoRA payload."""
        lora = LoraPayload(name="test.safetensors", model=0.8, clip=0.8)
        assert lora.name == "test.safetensors"
        assert lora.model == 0.8
        assert lora.clip == 0.8

    def test_strength_validation(self):
        """Test that strengths are validated."""
        # Valid range
        lora = LoraPayload(name="test.safetensors", model=5.0, clip=-5.0)
        assert lora.model == 5.0
        assert lora.clip == -5.0

        # Out of range
        with pytest.raises(ValidationError):
            LoraPayload(name="test.safetensors", model=15.0)  # Max is 10.0

        with pytest.raises(ValidationError):
            LoraPayload(name="test.safetensors", clip=-15.0)  # Min is -10.0


class TestPipelineBuilder:
    """Tests for PipelineBuilder."""

    def test_create_builder(self):
        """Test creating a builder."""
        builder = PipelineBuilder()
        assert builder.pipeline is not None
        assert len(builder.pipeline.nodes) == 0

    def test_add_checkpoint_loader(self):
        """Test adding checkpoint loader."""
        builder = PipelineBuilder()
        node_id = builder.add_checkpoint_loader("model_loader", "test.ckpt")
        assert node_id is not None
        node = builder.pipeline.get_node(node_id)
        assert node is not None
        assert node.class_type == "CheckpointLoaderSimple"
        assert node.get_input_value("ckpt_name") == "test.ckpt"

    def test_add_ksampler(self):
        """Test adding KSampler."""
        builder = PipelineBuilder()
        node_id = builder.add_ksampler("sampler", seed=42, steps=20, cfg=8.0)
        node = builder.pipeline.get_node(node_id)
        assert node is not None
        assert node.class_type == "KSampler"
        assert node.get_input_value("seed") == 42
        assert node.get_input_value("steps") == 20
        assert node.get_input_value("cfg") == 8.0

    def test_connect_nodes(self):
        """Test connecting nodes."""
        builder = PipelineBuilder()
        model_id = builder.add_checkpoint_loader("model_loader", "test.ckpt")
        sampler_id = builder.add_ksampler("sampler", seed=42, steps=20, cfg=8.0)

        builder.connect(sampler_id, "model", model_id, 0)

        sampler = builder.pipeline.get_node(sampler_id)
        conn = sampler.get_connection("model")
        assert conn is not None
        assert conn.node_id == model_id

    def test_connect_by_title(self):
        """Test connecting nodes by title."""
        builder = PipelineBuilder()
        builder.add_checkpoint_loader("model_loader", "test.ckpt")
        builder.add_ksampler("sampler", seed=42, steps=20, cfg=8.0)

        builder.connect_by_title("sampler", "model", "model_loader", 0)

        sampler = builder.pipeline.get_node_by_title("sampler")
        conn = sampler.get_connection("model")
        assert conn is not None

    def test_set_input_by_title(self):
        """Test setting inputs by title."""
        builder = PipelineBuilder()
        builder.add_ksampler("sampler", seed=42, steps=20, cfg=8.0)
        builder.set_input_by_title("sampler", "cfg", 9.5)

        sampler = builder.pipeline.get_node_by_title("sampler")
        assert sampler.get_input_value("cfg") == 9.5

    def test_set_inputs_by_dot_notation(self):
        """Test setting multiple inputs with dot notation."""
        builder = PipelineBuilder()
        builder.add_ksampler("sampler", seed=42, steps=20, cfg=8.0)
        builder.add_empty_latent("latent", width=512, height=512)

        builder.set_inputs_by_dot_notation(
            sampler__cfg=9.0,
            sampler__steps=25,
            latent__width=768,
        )

        sampler = builder.pipeline.get_node_by_title("sampler")
        latent = builder.pipeline.get_node_by_title("latent")

        assert sampler.get_input_value("cfg") == 9.0
        assert sampler.get_input_value("steps") == 25
        assert latent.get_input_value("width") == 768

    def test_to_comfy_format(self):
        """Test exporting to ComfyUI format."""
        builder = PipelineBuilder()
        model_id = builder.add_checkpoint_loader("model_loader", "test.ckpt")
        sampler_id = builder.add_ksampler("sampler", seed=42, steps=20, cfg=8.0)
        builder.connect(sampler_id, "model", model_id, 0)

        comfy_data = builder.to_comfy_format()
        assert isinstance(comfy_data, dict)
        assert sampler_id in comfy_data
        assert comfy_data[sampler_id]["class_type"] == "KSampler"


class TestLoraChainBuilder:
    """Tests for LoraChainBuilder."""

    def test_create_lora_chain(self):
        """Test creating a LoRA chain."""
        builder = PipelineBuilder()
        lora_chain = LoraChainBuilder(builder)
        assert len(lora_chain.lora_nodes) == 0

    def test_add_loras(self):
        """Test adding LoRAs to chain."""
        builder = PipelineBuilder()
        lora_chain = LoraChainBuilder(builder)

        lora_chain.add_lora("lora1.safetensors", model_strength=0.8, clip_strength=0.8)
        lora_chain.add_lora("lora2.safetensors", model_strength=1.0, clip_strength=1.0)

        assert len(lora_chain.lora_nodes) == 2
        assert "lora_0" in lora_chain.lora_nodes
        assert "lora_1" in lora_chain.lora_nodes

    def test_lora_chain_connections(self):
        """Test that LoRAs are properly chained."""
        builder = PipelineBuilder()
        lora_chain = LoraChainBuilder(builder)

        lora_chain.add_lora("lora1.safetensors", model_strength=0.8, clip_strength=0.8)
        lora_chain.add_lora("lora2.safetensors", model_strength=1.0, clip_strength=1.0)

        # Second LoRA should connect to first
        lora_1 = builder.pipeline.get_node("lora_1")
        model_conn = lora_1.get_connection("model")
        clip_conn = lora_1.get_connection("clip")

        assert model_conn is not None
        assert model_conn.node_id == "lora_0"
        assert clip_conn is not None
        assert clip_conn.node_id == "lora_0"


class TestPayloadToPipelineMapper:
    """Tests for PayloadToPipelineMapper."""

    def test_apply_payload(self):
        """Test applying payload to pipeline."""
        payload = GenerationPayload(
            prompt="test prompt",
            cfg_scale=7.5,
            ddim_steps=30,
            width=768,
            height=512,
        )

        builder = PipelineBuilder()
        builder.add_ksampler("sampler", seed=0, steps=0, cfg=0.0)
        builder.add_empty_latent("empty_latent_image")
        builder.add_clip_text_encode("prompt", "")

        mapper = PayloadToPipelineMapper(payload)
        mapper.apply_to_pipeline(builder)

        sampler = builder.pipeline.get_node_by_title("sampler")
        latent = builder.pipeline.get_node_by_title("empty_latent_image")
        prompt_node = builder.pipeline.get_node_by_title("prompt")

        assert sampler.get_input_value("cfg") == 7.5
        assert sampler.get_input_value("steps") == 30
        assert latent.get_input_value("width") == 768
        assert latent.get_input_value("height") == 512
        assert prompt_node.get_input_value("text") == "test prompt"
