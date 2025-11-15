"""Pipeline builder utilities for type-safe ComfyUI pipeline manipulation.

This module provides builder classes that make it easier to construct and
manipulate ComfyUI pipelines programmatically, with full type safety and
validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from hordelib.pipeline.schema import (
    ComfyNode,
    GenerationPayload,
    NodeConnection,
    NodeMeta,
    Pipeline,
    get_comfy_sampler_name,
)


class PipelineBuilder:
    """Builder for constructing and manipulating ComfyUI pipelines.

    This provides a fluent interface for building pipelines programmatically,
    with type-safe operations and validation.

    Example:
        >>> builder = PipelineBuilder()
        >>> builder.add_checkpoint_loader("model_loader", "Deliberate.ckpt")
        >>> builder.add_empty_latent("latent", width=512, height=512)
        >>> builder.add_ksampler("sampler", seed=12345, steps=20, cfg=8.0)
        >>> pipeline = builder.build()
    """

    def __init__(self, pipeline: Pipeline | None = None):
        """Initialize the builder.

        Args:
            pipeline: Optional existing pipeline to modify. If None, starts with empty pipeline.
        """
        self.pipeline = pipeline or Pipeline()
        self._next_node_id = 1000  # Start high to avoid conflicts with loaded pipelines

    @classmethod
    def from_file(cls, filepath: str | Path) -> PipelineBuilder:
        """Load a pipeline from a JSON file.

        Args:
            filepath: Path to the pipeline JSON file

        Returns:
            PipelineBuilder initialized with the loaded pipeline
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        pipeline = Pipeline.from_comfy_format(data)
        return cls(pipeline=pipeline)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineBuilder:
        """Load a pipeline from a dictionary.

        Args:
            data: Pipeline data in ComfyUI format

        Returns:
            PipelineBuilder initialized with the loaded pipeline
        """
        pipeline = Pipeline.from_comfy_format(data)
        return cls(pipeline=pipeline)

    def _generate_node_id(self) -> str:
        """Generate a unique node ID."""
        node_id = str(self._next_node_id)
        self._next_node_id += 1
        return node_id

    def add_node(
        self,
        title: str,
        class_type: str,
        inputs: dict[str, Any] | None = None,
        node_id: str | None = None,
    ) -> str:
        """Add a generic node to the pipeline.

        Args:
            title: Human-readable name for the node
            class_type: ComfyUI node class name
            inputs: Node input parameters
            node_id: Optional specific node ID. If None, auto-generated

        Returns:
            The node ID of the added node
        """
        if node_id is None:
            node_id = self._generate_node_id()

        node = ComfyNode(
            inputs=inputs or {},
            class_type=class_type,
            meta=NodeMeta(title=title),
        )

        self.pipeline.add_node(node_id, node)
        return node_id

    def add_checkpoint_loader(
        self,
        title: str,
        ckpt_name: str,
        node_id: str | None = None,
    ) -> str:
        """Add a CheckpointLoaderSimple node.

        Args:
            title: Node title
            ckpt_name: Checkpoint filename
            node_id: Optional specific node ID

        Returns:
            The node ID
        """
        return self.add_node(
            title=title,
            class_type="CheckpointLoaderSimple",
            inputs={"ckpt_name": ckpt_name},
            node_id=node_id,
        )

    def add_empty_latent(
        self,
        title: str,
        width: int = 512,
        height: int = 512,
        batch_size: int = 1,
        node_id: str | None = None,
    ) -> str:
        """Add an EmptyLatentImage node.

        Args:
            title: Node title
            width: Image width (will be rounded to multiple of 64)
            height: Image height (will be rounded to multiple of 64)
            batch_size: Batch size
            node_id: Optional specific node ID

        Returns:
            The node ID
        """
        # Ensure dimensions are divisible by 64
        width = ((width + 63) // 64) * 64
        height = ((height + 63) // 64) * 64

        return self.add_node(
            title=title,
            class_type="EmptyLatentImage",
            inputs={
                "width": width,
                "height": height,
                "batch_size": batch_size,
            },
            node_id=node_id,
        )

    def add_ksampler(
        self,
        title: str,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str = "euler",
        scheduler: str = "normal",
        denoise: float = 1.0,
        model_connection: tuple[str, int] | None = None,
        positive_connection: tuple[str, int] | None = None,
        negative_connection: tuple[str, int] | None = None,
        latent_connection: tuple[str, int] | None = None,
        node_id: str | None = None,
    ) -> str:
        """Add a KSampler node.

        Args:
            title: Node title
            seed: Random seed
            steps: Number of sampling steps
            cfg: CFG scale
            sampler_name: Sampler name
            scheduler: Scheduler type
            denoise: Denoising strength
            model_connection: Connection to model (node_id, output_index)
            positive_connection: Connection to positive prompt
            negative_connection: Connection to negative prompt
            latent_connection: Connection to latent image
            node_id: Optional specific node ID

        Returns:
            The node ID
        """
        inputs = {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": denoise,
        }

        # Add connections if provided
        if model_connection:
            inputs["model"] = NodeConnection(node_id=model_connection[0], output_index=model_connection[1])
        if positive_connection:
            inputs["positive"] = NodeConnection(
                node_id=positive_connection[0],
                output_index=positive_connection[1],
            )
        if negative_connection:
            inputs["negative"] = NodeConnection(
                node_id=negative_connection[0],
                output_index=negative_connection[1],
            )
        if latent_connection:
            inputs["latent_image"] = NodeConnection(
                node_id=latent_connection[0],
                output_index=latent_connection[1],
            )

        return self.add_node(
            title=title,
            class_type="KSampler",
            inputs=inputs,
            node_id=node_id,
        )

    def add_clip_text_encode(
        self,
        title: str,
        text: str,
        clip_connection: tuple[str, int] | None = None,
        node_id: str | None = None,
    ) -> str:
        """Add a CLIPTextEncode node.

        Args:
            title: Node title
            text: Prompt text
            clip_connection: Connection to CLIP (node_id, output_index)
            node_id: Optional specific node ID

        Returns:
            The node ID
        """
        inputs = {"text": text}

        if clip_connection:
            inputs["clip"] = NodeConnection(node_id=clip_connection[0], output_index=clip_connection[1])

        return self.add_node(
            title=title,
            class_type="CLIPTextEncode",
            inputs=inputs,
            node_id=node_id,
        )

    def add_lora_loader(
        self,
        title: str,
        lora_name: str,
        strength_model: float = 1.0,
        strength_clip: float = 1.0,
        model_connection: tuple[str, int] | None = None,
        clip_connection: tuple[str, int] | None = None,
        node_id: str | None = None,
    ) -> str:
        """Add a LoraLoader node.

        Args:
            title: Node title
            lora_name: LoRA filename
            strength_model: Model strength
            strength_clip: CLIP strength
            model_connection: Connection to model (node_id, output_index)
            clip_connection: Connection to CLIP (node_id, output_index)
            node_id: Optional specific node ID

        Returns:
            The node ID
        """
        inputs = {
            "lora_name": lora_name,
            "strength_model": strength_model,
            "strength_clip": strength_clip,
        }

        if model_connection:
            inputs["model"] = NodeConnection(node_id=model_connection[0], output_index=model_connection[1])
        if clip_connection:
            inputs["clip"] = NodeConnection(node_id=clip_connection[0], output_index=clip_connection[1])

        return self.add_node(
            title=title,
            class_type="HordeLoraLoader",
            inputs=inputs,
            node_id=node_id,
        )

    def add_vae_decode(
        self,
        title: str,
        samples_connection: tuple[str, int] | None = None,
        vae_connection: tuple[str, int] | None = None,
        node_id: str | None = None,
    ) -> str:
        """Add a VAEDecode node.

        Args:
            title: Node title
            samples_connection: Connection to samples (node_id, output_index)
            vae_connection: Connection to VAE (node_id, output_index)
            node_id: Optional specific node ID

        Returns:
            The node ID
        """
        inputs = {}

        if samples_connection:
            inputs["samples"] = NodeConnection(
                node_id=samples_connection[0],
                output_index=samples_connection[1],
            )
        if vae_connection:
            inputs["vae"] = NodeConnection(node_id=vae_connection[0], output_index=vae_connection[1])

        return self.add_node(
            title=title,
            class_type="VAEDecode",
            inputs=inputs,
            node_id=node_id,
        )

    def add_save_image(
        self,
        title: str,
        filename_prefix: str = "ComfyUI",
        images_connection: tuple[str, int] | None = None,
        node_id: str | None = None,
    ) -> str:
        """Add a SaveImage node.

        Args:
            title: Node title
            filename_prefix: Filename prefix for saved images
            images_connection: Connection to images (node_id, output_index)
            node_id: Optional specific node ID

        Returns:
            The node ID
        """
        inputs = {"filename_prefix": filename_prefix}

        if images_connection:
            inputs["images"] = NodeConnection(
                node_id=images_connection[0],
                output_index=images_connection[1],
            )

        return self.add_node(
            title=title,
            class_type="SaveImage",
            inputs=inputs,
            node_id=node_id,
        )

    def connect(self, target_node_id: str, input_name: str, source_node_id: str, output_index: int = 0) -> None:
        """Connect a node input to another node's output.

        Args:
            target_node_id: The node receiving the connection
            input_name: The input parameter to connect
            source_node_id: The node providing the output
            output_index: The output slot index (default: 0)
        """
        target_node = self.pipeline.get_node(target_node_id)
        if target_node is None:
            raise ValueError(f"Target node '{target_node_id}' not found")

        target_node.connect_input(input_name, source_node_id, output_index)

    def connect_by_title(
        self,
        target_title: str,
        input_name: str,
        source_title: str,
        output_index: int = 0,
    ) -> None:
        """Connect nodes by their titles instead of IDs.

        Args:
            target_title: The title of the node receiving the connection
            input_name: The input parameter to connect
            source_title: The title of the node providing the output
            output_index: The output slot index (default: 0)
        """
        target_node = self.pipeline.get_node_by_title(target_title)
        source_node = self.pipeline.get_node_by_title(source_title)

        if target_node is None:
            raise ValueError(f"Target node with title '{target_title}' not found")
        if source_node is None:
            raise ValueError(f"Source node with title '{source_title}' not found")

        # Find the node ID for the source node
        source_node_id = None
        for node_id, node in self.pipeline.nodes.items():
            if node is source_node:
                source_node_id = node_id
                break

        if source_node_id is None:
            raise ValueError(f"Could not find node ID for '{source_title}'")

        target_node.connect_input(input_name, source_node_id, output_index)

    def set_input(self, node_id: str, input_name: str, value: Any) -> None:
        """Set an input value on a node.

        Args:
            node_id: The node ID
            input_name: The input parameter name
            value: The value to set
        """
        node = self.pipeline.get_node(node_id)
        if node is None:
            raise ValueError(f"Node '{node_id}' not found")

        node.set_input_value(input_name, value)

    def set_input_by_title(self, node_title: str, input_name: str, value: Any) -> None:
        """Set an input value on a node by its title.

        Args:
            node_title: The node title
            input_name: The input parameter name
            value: The value to set
        """
        node = self.pipeline.get_node_by_title(node_title)
        if node is None:
            raise ValueError(f"Node with title '{node_title}' not found")

        node.set_input_value(input_name, value)

    def set_inputs_by_dot_notation(self, **kwargs) -> None:
        """Set multiple inputs using dot notation (compatible with old _set() method).

        Args:
            **kwargs: Key-value pairs where keys use double underscore as separator.
                     Format: "node_title__input_name" (e.g., sampler__cfg=8.0)
                     You can also pass a dict with dot notation keys using **dict syntax.

        Example:
            >>> # Using double underscore (recommended for kwargs)
            >>> builder.set_inputs_by_dot_notation(
            ...     sampler__cfg=8.0,
            ...     sampler__steps=20,
            ...     prompt__text="a beautiful landscape"
            ... )
            >>> # Using dict unpacking (for keys with dots)
            >>> builder.set_inputs_by_dot_notation(**{
            ...     "sampler.cfg": 8.0,
            ...     "sampler.steps": 20
            ... })
        """
        for key, value in kwargs.items():
            # Try splitting on both '__' and '.' for flexibility
            if "__" in key:
                parts = key.split("__", 1)
            elif "." in key:
                parts = key.split(".", 1)
            else:
                logger.warning(f"Invalid notation key (use '__' or '.'): {key}")
                continue

            if len(parts) != 2:
                logger.warning(f"Invalid notation key: {key}")
                continue

            node_title, input_name = parts

            try:
                self.set_input_by_title(node_title, input_name, value)
            except ValueError as e:
                logger.debug(f"Could not set {key}: {e}")

    def build(self) -> Pipeline:
        """Build and return the pipeline.

        Returns:
            The constructed Pipeline object
        """
        return self.pipeline

    def to_comfy_format(self) -> dict[str, Any]:
        """Export the pipeline to ComfyUI's native format.

        Returns:
            Dictionary in ComfyUI's pipeline format
        """
        return self.pipeline.to_comfy_format()

    def save(self, filepath: str | Path) -> None:
        """Save the pipeline to a JSON file.

        Args:
            filepath: Path where to save the pipeline
        """
        with open(filepath, "w") as f:
            json.dump(self.to_comfy_format(), f, indent=2)

        logger.info(f"Pipeline saved to {filepath}")


class LoraChainBuilder:
    """Builder for creating chains of LoRA nodes.

    This replicates the dynamic LoRA injection behavior from horde.py,
    creating a chain of LoRA loaders and properly connecting them.

    Example:
        >>> builder = PipelineBuilder.from_file("pipeline.json")
        >>> lora_chain = LoraChainBuilder(builder)
        >>> lora_chain.add_lora("lora1.safetensors", model_strength=0.8, clip_strength=0.8)
        >>> lora_chain.add_lora("lora2.safetensors", model_strength=1.0, clip_strength=1.0)
        >>> lora_chain.connect_to_pipeline("model_loader", "sampler", "clip_skip")
    """

    def __init__(self, pipeline_builder: PipelineBuilder):
        """Initialize the LoRA chain builder.

        Args:
            pipeline_builder: The pipeline builder to add LoRA nodes to
        """
        self.pipeline_builder = pipeline_builder
        self.lora_nodes: list[str] = []

    def add_lora(
        self,
        lora_name: str,
        model_strength: float = 1.0,
        clip_strength: float = 1.0,
    ) -> str:
        """Add a LoRA to the chain.

        Args:
            lora_name: LoRA filename
            model_strength: Model strength multiplier
            clip_strength: CLIP strength multiplier

        Returns:
            The node ID of the created LoRA node
        """
        lora_index = len(self.lora_nodes)
        node_id = f"lora_{lora_index}"

        if lora_index == 0:
            # First LoRA - will be connected to model_loader later
            self.pipeline_builder.add_lora_loader(
                title=node_id,
                lora_name=lora_name,
                strength_model=model_strength,
                strength_clip=clip_strength,
                node_id=node_id,
            )
        else:
            # Subsequent LoRAs - connect to previous LoRA
            prev_lora_id = self.lora_nodes[lora_index - 1]
            self.pipeline_builder.add_lora_loader(
                title=node_id,
                lora_name=lora_name,
                strength_model=model_strength,
                strength_clip=clip_strength,
                model_connection=(prev_lora_id, 0),
                clip_connection=(prev_lora_id, 1),
                node_id=node_id,
            )

        self.lora_nodes.append(node_id)
        return node_id

    def connect_to_pipeline(
        self,
        model_loader_title: str,
        sampler_title: str | None = None,
        clip_skip_title: str | None = None,
        upscale_sampler_title: str | None = None,
        cfg_guider_title: str | None = None,
        basic_scheduler_title: str | None = None,
    ) -> None:
        """Connect the LoRA chain to the pipeline.

        This connects:
        - First LoRA inputs to the model loader
        - Last LoRA outputs to sampler and clip_skip

        Args:
            model_loader_title: Title of the model loader node
            sampler_title: Title of the main sampler node (for SD models)
            clip_skip_title: Title of the clip skip node (for SD models)
            upscale_sampler_title: Title of the upscale sampler (for hires fix)
            cfg_guider_title: Title of the CFG guider (for Flux models)
            basic_scheduler_title: Title of the basic scheduler (for Flux models)
        """
        if not self.lora_nodes:
            logger.warning("No LoRA nodes to connect")
            return

        # Connect first LoRA to model loader
        first_lora_id = self.lora_nodes[0]
        self.pipeline_builder.connect_by_title(
            target_title=first_lora_id,
            input_name="model",
            source_title=model_loader_title,
            output_index=0,
        )
        self.pipeline_builder.connect_by_title(
            target_title=first_lora_id,
            input_name="clip",
            source_title=model_loader_title,
            output_index=1,
        )

        # Connect last LoRA to samplers and clip_skip
        last_lora_id = self.lora_nodes[-1]

        # For Flux models
        if cfg_guider_title:
            self.pipeline_builder.connect_by_title(
                target_title=cfg_guider_title,
                input_name="model",
                source_title=last_lora_id,
                output_index=0,
            )

        if basic_scheduler_title:
            self.pipeline_builder.connect_by_title(
                target_title=basic_scheduler_title,
                input_name="model",
                source_title=last_lora_id,
                output_index=0,
            )

        # For SD models
        if sampler_title:
            self.pipeline_builder.connect_by_title(
                target_title=sampler_title,
                input_name="model",
                source_title=last_lora_id,
                output_index=0,
            )

        if upscale_sampler_title:
            self.pipeline_builder.connect_by_title(
                target_title=upscale_sampler_title,
                input_name="model",
                source_title=last_lora_id,
                output_index=0,
            )

        if clip_skip_title:
            self.pipeline_builder.connect_by_title(
                target_title=clip_skip_title,
                input_name="clip",
                source_title=last_lora_id,
                output_index=1,
            )

        logger.debug(f"Connected LoRA chain of {len(self.lora_nodes)} nodes")


class PayloadToPipelineMapper:
    """Maps a GenerationPayload to pipeline parameters.

    This provides a type-safe way to apply payload values to a pipeline,
    replacing the PAYLOAD_TO_PIPELINE_PARAMETER_MAPPING dictionary.
    """

    def __init__(self, payload: GenerationPayload):
        """Initialize the mapper.

        Args:
            payload: The validated generation payload
        """
        self.payload = payload

    def apply_to_pipeline(self, builder: PipelineBuilder) -> None:
        """Apply payload values to a pipeline.

        Args:
            builder: The pipeline builder to apply values to
        """
        # Convert sampler name
        comfy_sampler = get_comfy_sampler_name(self.payload.sampler_name)

        # Build mapping of node titles to input values
        mappings = {
            # Sampler
            "sampler.sampler_name": comfy_sampler,
            "sampler.cfg": self.payload.cfg_scale,
            "sampler.denoise": self.payload.denoising_strength,
            "sampler.seed": self.payload.seed,
            "sampler.noise_seed": self.payload.seed,
            "sampler.steps": self.payload.ddim_steps,
            "sampler.scheduler": self.payload.scheduler,
            # Latent image
            "empty_latent_image.height": self.payload.height,
            "empty_latent_image.width": self.payload.width,
            "empty_latent_image.batch_size": self.payload.n_iter,
            # Model loader
            "model_loader.ckpt_name": self.payload.model,
            "model_loader.model_name": self.payload.model,
            "model_loader.horde_model_name": self.payload.model_name,
            "model_loader.seamless_tiling_enabled": self.payload.tiling,
            # CLIP
            "clip_skip.stop_at_clip_layer": -self.payload.clip_skip,
            # Prompts
            "prompt.text": self.payload.prompt,
            "negative_prompt.text": self.payload.negative_prompt,
            # Image loader
            "image_loader.image": self.payload.source_image,
            # Hires fix
            "upscale_sampler.denoise": self.payload.hires_fix_denoising_strength,
            "upscale_sampler.seed": self.payload.seed,
            "upscale_sampler.cfg": self.payload.cfg_scale,
            "upscale_sampler.sampler_name": comfy_sampler,
            # ControlNet
            "controlnet_apply.strength": self.payload.control_strength,
            # Batch
            "repeat_image_batch.amount": self.payload.n_iter,
            # Flux
            "cfg_guider.cfg": self.payload.cfg_scale,
            "random_noise.noise_seed": self.payload.seed,
            "k_sampler_select.sampler_name": comfy_sampler,
            "basic_scheduler.denoise": self.payload.denoising_strength,
            "basic_scheduler.steps": self.payload.ddim_steps,
            # Stable Cascade
            "stable_cascade_empty_latent_image.width": self.payload.width,
            "stable_cascade_empty_latent_image.height": self.payload.height,
            "stable_cascade_empty_latent_image.batch_size": self.payload.n_iter,
            "sampler_stage_c.sampler_name": comfy_sampler,
            "sampler_stage_b.sampler_name": comfy_sampler,
            "sampler_stage_c.cfg": self.payload.cfg_scale,
            "sampler_stage_c.denoise": self.payload.denoising_strength,
            "sampler_stage_b.seed": self.payload.seed,
            "sampler_stage_c.seed": self.payload.seed,
            "sampler_stage_b.steps": int(self.payload.ddim_steps * 0.33),
            "sampler_stage_c.steps": int(self.payload.ddim_steps * 0.67),
            "model_loader_stage_c.ckpt_name": self.payload.stable_cascade_stage_c,
            "model_loader_stage_c.model_name": self.payload.stable_cascade_stage_c,
            "model_loader_stage_c.horde_model_name": self.payload.model_name,
            "model_loader_stage_b.ckpt_name": self.payload.stable_cascade_stage_b,
            "model_loader_stage_b.model_name": self.payload.stable_cascade_stage_b,
            "model_loader_stage_b.horde_model_name": self.payload.model_name,
        }

        # Apply all mappings
        builder.set_inputs_by_dot_notation(**mappings)

        logger.debug("Applied payload values to pipeline")
