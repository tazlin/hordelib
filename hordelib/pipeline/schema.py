"""Pydantic schemas for ComfyUI pipeline definitions.

This module provides type-safe models for defining, validating, and manipulating
ComfyUI pipelines. It replaces the dict-based PAYLOAD_SCHEMA system with proper
Pydantic models.
"""

from __future__ import annotations

import sys
from enum import Enum
from typing import Any, Literal, Union

from PIL import Image
from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# ComfyUI Node Structure Models
# =============================================================================


class NodeConnection(BaseModel):
    """Represents a connection to another node's output.

    In ComfyUI, connections are represented as [node_id, output_index].
    For example: ["4", 0] means connect to node "4", output index 0.
    """

    node_id: str = Field(..., description="The ID of the source node")
    output_index: int = Field(default=0, ge=0, description="The output slot index (0-based)")

    def to_comfy_format(self) -> list[str | int]:
        """Convert to ComfyUI's native [node_id, output_index] format."""
        return [self.node_id, self.output_index]

    @classmethod
    def from_comfy_format(cls, data: list[str | int]) -> NodeConnection:
        """Create from ComfyUI's native [node_id, output_index] format."""
        if not isinstance(data, list) or len(data) != 2:
            raise ValueError(f"Invalid connection format: {data}")
        return cls(node_id=str(data[0]), output_index=int(data[1]))


class NodeMeta(BaseModel):
    """Metadata for a ComfyUI node.

    The _meta field contains information about the node that's not part of
    its functional definition.
    """

    title: str = Field(..., description="Human-readable name for the node")


class NodeInputs(BaseModel):
    """Base class for node inputs.

    Each node type should subclass this to define its specific inputs.
    Inputs can be either direct values or connections to other nodes.
    """

    class Config:
        extra = "allow"  # Allow additional fields for extensibility

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Override to handle NodeConnection serialization."""
        data = super().model_dump(**kwargs)
        # Convert NodeConnection objects back to list format
        for key, value in data.items():
            if isinstance(value, dict) and "node_id" in value and "output_index" in value:
                data[key] = [value["node_id"], value["output_index"]]
        return data


class ComfyNode(BaseModel):
    """Represents a single node in a ComfyUI pipeline.

    Each node has:
    - inputs: Parameters and connections to other nodes
    - class_type: The ComfyUI node class name
    - meta: Metadata including the human-readable title (serialized as _meta)
    """

    inputs: dict[str, Any] = Field(default_factory=dict, description="Node input parameters")
    class_type: str = Field(..., description="ComfyUI node class name")
    meta: NodeMeta = Field(..., alias="_meta", description="Node metadata")

    class Config:
        populate_by_name = True  # Allow both 'meta' and '_meta' when parsing

    @field_validator("inputs", mode="before")
    @classmethod
    def parse_connections(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Parse connection arrays into NodeConnection objects."""
        if not isinstance(v, dict):
            return v

        result = {}
        for key, value in v.items():
            # Check if this looks like a connection [node_id, output_index]
            if isinstance(value, list) and len(value) == 2:
                try:
                    result[key] = NodeConnection.from_comfy_format(value)
                except (ValueError, TypeError):
                    # Not a valid connection, keep as-is
                    result[key] = value
            else:
                result[key] = value

        return result

    def get_input_value(self, key: str) -> Any:
        """Get an input value, returning None if not found."""
        return self.inputs.get(key)

    def set_input_value(self, key: str, value: Any) -> None:
        """Set an input value."""
        self.inputs[key] = value

    def connect_input(self, input_name: str, source_node_id: str, output_index: int = 0) -> None:
        """Connect an input to another node's output."""
        self.inputs[input_name] = NodeConnection(node_id=source_node_id, output_index=output_index)

    def get_connection(self, input_name: str) -> NodeConnection | None:
        """Get a node connection if the input is connected to another node."""
        value = self.inputs.get(input_name)
        if isinstance(value, NodeConnection):
            return value
        return None


class Pipeline(BaseModel):
    """Represents a complete ComfyUI pipeline/workflow.

    A pipeline is a dictionary mapping node IDs to node definitions.
    Node IDs can be numeric strings (e.g., "3", "4") or named strings (e.g., "lora_0").
    """

    nodes: dict[str, ComfyNode] = Field(
        default_factory=dict,
        description="Mapping of node IDs to node definitions",
    )

    def get_node(self, node_id: str) -> ComfyNode | None:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_node_by_title(self, title: str) -> ComfyNode | None:
        """Get a node by its metadata title."""
        for node in self.nodes.values():
            if node.meta.title == title:
                return node
        return None

    def add_node(self, node_id: str, node: ComfyNode) -> None:
        """Add a node to the pipeline."""
        self.nodes[node_id] = node

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the pipeline."""
        if node_id in self.nodes:
            del self.nodes[node_id]

    def to_comfy_format(self) -> dict[str, Any]:
        """Export to ComfyUI's native pipeline format."""
        result = {}
        for node_id, node in self.nodes.items():
            node_data = {
                "inputs": {},
                "class_type": node.class_type,
                "_meta": {"title": node.meta.title},
            }

            # Convert inputs, handling NodeConnection objects
            for key, value in node.inputs.items():
                if isinstance(value, NodeConnection):
                    node_data["inputs"][key] = value.to_comfy_format()
                else:
                    node_data["inputs"][key] = value

            result[node_id] = node_data

        return result

    @classmethod
    def from_comfy_format(cls, data: dict[str, Any]) -> Pipeline:
        """Create a Pipeline from ComfyUI's native format."""
        nodes = {}
        for node_id, node_data in data.items():
            nodes[node_id] = ComfyNode(
                inputs=node_data.get("inputs", {}),
                class_type=node_data["class_type"],
                _meta=NodeMeta(title=node_data.get("_meta", {}).get("title", node_id)),
            )

        return cls(nodes=nodes)


# =============================================================================
# Specific Node Type Models
# =============================================================================


class KSamplerInputs(NodeInputs):
    """Inputs for the KSampler node."""

    seed: int = Field(..., ge=0)
    steps: int = Field(..., ge=1, le=500)
    cfg: float = Field(..., ge=1.0, le=100.0)
    sampler_name: str = Field(...)
    scheduler: str = Field(...)
    denoise: float = Field(..., ge=0.01, le=1.0)
    model: NodeConnection = Field(...)
    positive: NodeConnection = Field(...)
    negative: NodeConnection = Field(...)
    latent_image: NodeConnection = Field(...)


class CheckpointLoaderInputs(NodeInputs):
    """Inputs for the CheckpointLoaderSimple node."""

    ckpt_name: str = Field(...)


class EmptyLatentImageInputs(NodeInputs):
    """Inputs for the EmptyLatentImage node."""

    width: int = Field(..., ge=64, le=8192)
    height: int = Field(..., ge=64, le=8192)
    batch_size: int = Field(default=1, ge=1, le=100)

    @field_validator("width", "height")
    @classmethod
    def validate_divisible_by_64(cls, v: int) -> int:
        """Ensure dimensions are divisible by 64."""
        if v % 64 != 0:
            # Round up to nearest multiple of 64
            v = ((v + 63) // 64) * 64
        return v


class CLIPTextEncodeInputs(NodeInputs):
    """Inputs for the CLIPTextEncode node."""

    text: str = Field(...)
    clip: NodeConnection = Field(...)


class LoraLoaderInputs(NodeInputs):
    """Inputs for the LoraLoader node."""

    model: NodeConnection = Field(...)
    clip: NodeConnection = Field(...)
    lora_name: str = Field(...)
    strength_model: float = Field(default=1.0, ge=-10.0, le=10.0)
    strength_clip: float = Field(default=1.0, ge=-10.0, le=10.0)


class VAEDecodeInputs(NodeInputs):
    """Inputs for the VAEDecode node."""

    samples: NodeConnection = Field(...)
    vae: NodeConnection = Field(...)


class VAEEncodeInputs(NodeInputs):
    """Inputs for the VAEEncode node."""

    pixels: NodeConnection = Field(...)
    vae: NodeConnection = Field(...)


class SaveImageInputs(NodeInputs):
    """Inputs for the SaveImage node."""

    filename_prefix: str = Field(...)
    images: NodeConnection = Field(...)


class LoadImageInputs(NodeInputs):
    """Inputs for the LoadImage node."""

    image: str | Image.Image = Field(...)
    upload: str = Field(default="image")


class ControlNetApplyInputs(NodeInputs):
    """Inputs for the ControlNetApply node."""

    conditioning: NodeConnection = Field(...)
    control_net: NodeConnection = Field(...)
    image: NodeConnection = Field(...)
    strength: float = Field(default=1.0, ge=0.01, le=3.0)


# =============================================================================
# Payload Models (replacing PAYLOAD_SCHEMA)
# =============================================================================


class SamplerName(str, Enum):
    """Valid sampler names."""

    K_EULER = "k_euler"
    K_EULER_A = "k_euler_a"
    K_HEUN = "k_heun"
    K_DPM_2 = "k_dpm_2"
    K_DPM_2_A = "k_dpm_2_a"
    K_LMS = "k_lms"
    K_DPM_FAST = "k_dpm_fast"
    K_DPM_ADAPTIVE = "k_dpm_adaptive"
    K_DPMPP_2S_A = "k_dpmpp_2s_a"
    K_DPMPP_SDE = "k_dpmpp_sde"
    K_DPMPP_2M = "k_dpmpp_2m"
    DDIM = "ddim"
    UNI_PC = "uni_pc"
    UNI_PC_BH2 = "uni_pc_bh2"
    PLMS = "plms"
    LCM = "lcm"


class SchedulerType(str, Enum):
    """Valid scheduler types."""

    NORMAL = "normal"
    KARRAS = "karras"
    SIMPLE = "simple"
    DDIM_UNIFORM = "ddim_uniform"
    SGM_UNIFORM = "sgm_uniform"
    EXPONENTIAL = "exponential"


class ControlType(str, Enum):
    """Valid ControlNet types."""

    CANNY = "canny"
    HED = "hed"
    DEPTH = "depth"
    NORMAL = "normal"
    OPENPOSE = "openpose"
    SEG = "seg"
    SCRIBBLE = "scribble"
    FAKESCRIBBLES = "fakescribbles"
    HOUGH = "hough"
    MLSD = "mlsd"


class SourceProcessing(str, Enum):
    """Valid source image processing modes."""

    IMG2IMG = "img2img"
    INPAINTING = "inpainting"
    OUTPAINTING = "outpainting"
    REMIX = "remix"


class LoraPayload(BaseModel):
    """Payload for a single LoRA."""

    name: str = Field(..., description="LoRA model name or filename")
    model: float = Field(default=1.0, ge=-10.0, le=10.0, description="Model strength")
    clip: float = Field(default=1.0, ge=-10.0, le=10.0, description="CLIP strength")
    inject_trigger: str | None = Field(default=None, description="Trigger word injection mode")
    is_version: bool | None = Field(default=None, description="Whether this is a version reference")


class TextualInversionPayload(BaseModel):
    """Payload for a single Textual Inversion embedding."""

    name: str = Field(..., description="TI embedding name")
    inject_ti: str | None = Field(default=None, description="TI injection mode")
    strength: float = Field(default=1.0, ge=-10.0, le=10.0, description="Embedding strength")


class ExtraImagePayload(BaseModel):
    """Payload for extra source images (e.g., Stable Cascade Remix)."""

    image: Image.Image = Field(..., description="The PIL Image object")
    strength: float = Field(default=1.0, ge=0.0, le=5.0, description="Image influence strength")


class ExtraTextPayload(BaseModel):
    """Payload for extra text inputs (e.g., QR codes)."""

    text: str = Field(..., description="The text content")
    reference: str | None = Field(default=None, description="Reference identifier")


class GenerationPayload(BaseModel):
    """Main payload for image generation requests.

    This replaces the dict-based PAYLOAD_SCHEMA with a proper Pydantic model
    that provides validation, type hints, and better error messages.
    """

    # Sampling parameters
    sampler_name: SamplerName = Field(default=SamplerName.K_EULER)
    scheduler: SchedulerType = Field(default=SchedulerType.NORMAL)
    cfg_scale: float = Field(default=8.0, ge=1.0, le=100.0)
    denoising_strength: float = Field(default=1.0, ge=0.01, le=1.0)
    ddim_steps: int = Field(default=30, ge=1, le=500)
    seed: int = Field(default_factory=lambda: __import__("random").randint(0, sys.maxsize))

    # Image dimensions
    width: int = Field(default=512, ge=64, le=8192)
    height: int = Field(default=512, ge=64, le=8192)

    @field_validator("width", "height")
    @classmethod
    def validate_divisible_by_64(cls, v: int) -> int:
        """Ensure dimensions are divisible by 64."""
        if v % 64 != 0:
            v = ((v + 63) // 64) * 64
        return v

    # Prompts
    prompt: str = Field(default="")
    negative_prompt: str = Field(default="")

    # Model and clip
    model: str = Field(default="stable_diffusion")
    model_name: str = Field(default="stable_diffusion", description="Internal model name for hordelib")
    clip_skip: int = Field(default=1, ge=1, le=20)

    # Iterations
    n_iter: int = Field(default=1, ge=1, le=100, description="Number of iterations/batch size")

    # LoRAs and embeddings
    loras: list[LoraPayload] = Field(default_factory=list)
    tis: list[TextualInversionPayload] = Field(default_factory=list)

    # Source image processing
    source_image: Image.Image | None = Field(default=None)
    source_mask: Image.Image | None = Field(default=None)
    source_processing: SourceProcessing | None = Field(default=None)

    # ControlNet
    control_type: ControlType | None = Field(default=None)
    control_strength: float = Field(default=1.0, ge=0.01, le=3.0)
    image_is_control: bool = Field(default=False, description="Whether source_image is already a control map")
    return_control_map: bool = Field(default=False, description="Return the control map instead of final image")

    # Hires fix
    hires_fix: bool = Field(default=False)
    hires_fix_denoising_strength: float = Field(default=0.65, ge=0.01, le=1.0)

    # Stable Cascade specific
    stable_cascade_stage_b: str | None = Field(default=None)
    stable_cascade_stage_c: str | None = Field(default=None)
    extra_source_images: list[ExtraImagePayload] = Field(default_factory=list)

    # QR codes and other extras
    extra_texts: list[ExtraTextPayload] = Field(default_factory=list)

    # Miscellaneous
    tiling: bool = Field(default=False, description="Enable seamless tiling")
    transparent: bool = Field(default=False, description="Enable transparency (layer diffusion)")
    workflow: str = Field(default="auto_detect", description="Pipeline workflow to use")

    class Config:
        arbitrary_types_allowed = True  # Allow PIL.Image.Image
        use_enum_values = True  # Use enum values instead of enum objects


# =============================================================================
# Sampler Mapping
# =============================================================================


SAMPLER_MAP = {
    SamplerName.K_EULER: "euler",
    SamplerName.K_EULER_A: "euler_ancestral",
    SamplerName.K_HEUN: "heun",
    SamplerName.K_DPM_2: "dpm_2",
    SamplerName.K_DPM_2_A: "dpm_2_ancestral",
    SamplerName.K_LMS: "lms",
    SamplerName.K_DPM_FAST: "dpm_fast",
    SamplerName.K_DPM_ADAPTIVE: "dpm_adaptive",
    SamplerName.K_DPMPP_2S_A: "dpmpp_2s_ancestral",
    SamplerName.K_DPMPP_SDE: "dpmpp_sde",
    SamplerName.K_DPMPP_2M: "dpmpp_2m",
    SamplerName.DDIM: "ddim",
    SamplerName.UNI_PC: "uni_pc",
    SamplerName.UNI_PC_BH2: "uni_pc_bh2",
    SamplerName.PLMS: "euler",
    SamplerName.LCM: "lcm",
}


def get_comfy_sampler_name(sampler: SamplerName | str) -> str:
    """Convert a horde sampler name to ComfyUI sampler name."""
    if isinstance(sampler, str):
        try:
            sampler = SamplerName(sampler)
        except ValueError:
            # Fallback to euler if unknown
            return "euler"

    return SAMPLER_MAP.get(sampler, "euler")


# Rebuild models to ensure forward references are resolved
ComfyNode.model_rebuild()
Pipeline.model_rebuild()
