import argparse
from pathlib import Path

# Remove any command line args passed to pytest. ComfyUI hates
# the pytest args being in argv, we we hack them out purely for testing.
import pyarrow.parquet as pq
from PIL import Image

import hordelib


def main(batch_name: str, seed: str, karras: bool):
    print(f"{batch_name=} {seed=} {karras=}")
    hordelib.initialise()

    from hordelib.horde import HordeLib
    from hordelib.shared_model_manager import SharedModelManager

    table = pq.read_table("examples/diffusiondb_longprompts_sample.parquet")
    df = table.to_pandas()

    horde = HordeLib()
    SharedModelManager.loadModelManagers(compvis=True)

    HORDE_MODEL_NAMES = ["Deliberate", "stable_diffusion"]

    for horde_model_name in HORDE_MODEL_NAMES:
        SharedModelManager.manager.load(horde_model_name)

    HEADERS_LOOKUP = {"prompt": 0, "step": 1, "sampler": 2, "cfg": 3, "width": 4, "height": 5, "image_name": 6}
    SAMPLERS_LOOKUP = {
        1: "ddim",
        2: "plms",
        3: "k_euler",
        4: "k_euler_a",
        5: "k_heun",
        6: "k_dpm_2",
        7: "k_dpm_2_a",
        8: "k_lms",
        9: "others",
        None: "k_euler",
    }

    TARGET_IMAGE_PATH = Path("images/longprompts/")

    # Get confirmation from the user that this is the right BATCH_NAME
    # before we start running inference.
    print(f"Batch name: {batch_name}.")
    print("Press enter to continue, or ctrl+c to cancel.")
    input()

    def do_inference(*, df_row, horde_model_name) -> bool:
        if df_row["sampler"] == 9:
            return False

        diffusion_db_id = df_row["image_name"].split(".")[0]
        target_filepath = TARGET_IMAGE_PATH.joinpath(f"{diffusion_db_id}.{horde_model_name}.{batch_name}.webp")
        if target_filepath.exists():
            return True
        data = {
            "sampler_name": SAMPLERS_LOOKUP[df_row["sampler"]],
            "cfg_scale": df_row["cfg"],
            "denoising_strength": 0.75,
            "seed": seed,
            "height": df_row["height"],
            "width": df_row["width"],
            "karras": karras,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": df_row["prompt"],
            "ddim_steps": df_row["step"],
            "n_iter": 1,
            "model": horde_model_name,
        }
        print(data)
        try:
            pil_image = horde.basic_inference(data)
            pil_image.save(target_filepath, quality=90)
            return True
        except Exception as e:
            with open(TARGET_IMAGE_PATH.joinpath("error.txt"), "a") as f:
                f.write(f"{diffusion_db_id}\n{e}\n\n")
                return False

    length = 5
    num_splits = len(df) // length
    batches_of_dataframes = [df[i * length : (i + 1) * length] for i in range(num_splits)]
    for batch in batches_of_dataframes:
        batch.reset_index()
        for horde_model_name in HORDE_MODEL_NAMES:
            results = batch.apply(
                lambda x: do_inference(df_row=x, horde_model_name=horde_model_name),
                axis=1,
                result_type="expand",
            )
            print(results)


if __name__ == "__main__":
    argParsers = argparse.ArgumentParser()
    argParsers.add_argument("--batch_name", type=str, required=True)
    argParsers.add_argument("--seed", type=str, default="23113")
    argParsers.add_argument("--karras", action="store_true")
    args = argParsers.parse_args()
    main(batch_name=args.batch_name, seed=args.seed, karras=args.karras)
