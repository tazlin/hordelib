import argparse
import datetime
import json
from pathlib import Path

import open_clip

# Remove any command line args passed to pytest. ComfyUI hates
# the pytest args being in argv, we we hack them out purely for testing.
import pyarrow.parquet as pq
from loguru import logger
from PIL import Image

import hordelib


def main(run_batch_name: str, seed: str, karras: bool, total_images: int):
    print(f"{run_batch_name=} {seed=} {karras=}")
    hordelib.initialise()

    from hordelib.horde import HordeLib
    from hordelib.shared_model_manager import SharedModelManager

    path_to_parquet = Path("examples/diffusiondb_longprompts_sample.parquet")
    if not path_to_parquet.exists():
        logger.error(
            "`examples/diffusiondb_longprompts_sample.parquet` does not exist. Please make sure your working directory is in hordelib."
        )
        exit(1)

    table = pq.read_table(path_to_parquet)
    df = table.to_pandas()

    horde = HordeLib()
    SharedModelManager.loadModelManagers(compvis=True)

    HORDE_MODEL_NAMES = ["Deliberate", "stable_diffusion"]

    for horde_model_name in HORDE_MODEL_NAMES:
        SharedModelManager.manager.load(horde_model_name)

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
    TARGET_IMAGE_PATH.mkdir(exist_ok=True)

    # Get confirmation from the user that this is the right BATCH_NAME
    # before we start running inference.
    print()
    print(f"Run batch name: {run_batch_name}.")
    print("This will be included in the names of the files written out this run.")
    print()
    print("Press enter to continue, or ctrl+c to cancel.")
    input()

    def do_inference(*, df_row, horde_model_name) -> bool:
        if df_row["sampler"] == 9:
            return False

        tokens = open_clip.tokenize(df_row["prompt"], context_length=1024)
        tokens_count = len([x for x in tokens[0] if x != 0])

        diffusion_db_id = df_row["image_name"].split(".")[0]
        target_filepath = TARGET_IMAGE_PATH.joinpath(
            f"{diffusion_db_id}.{seed}.{tokens_count}.{horde_model_name}.{run_batch_name}.webp",
        )
        logger.warning("--> Starting inference.")
        print(f"{diffusion_db_id=} {tokens_count=}")
        if target_filepath.exists():
            print(f"Skipping {diffusion_db_id} because it already exists.")
            return False
        if tokens_count <= 77:
            print(f"{diffusion_db_id} has less than 78 tokens.")
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
            with open(TARGET_IMAGE_PATH.joinpath("completed.txt"), "a") as f:
                f.writelines(
                    f"diffusiondb_id= {diffusion_db_id} - batchname= {run_batch_name} - tokens= {tokens_count} - time= {datetime.datetime.now()}\n"
                )
                f.write(json.dumps(data) + "\n\n")
            return True
        except Exception as e:
            with open(TARGET_IMAGE_PATH.joinpath("error.txt"), "a") as f:
                f.write(f"{diffusion_db_id}\n{e}\n\n")
                return False

    # Run inference in batches of 5.
    # This allows us to alternate between models without incurring quite as much load time
    total_images_created = 0
    num_images_per_round = 5
    num_splits = len(df) // num_images_per_round
    batches_of_dataframes = [df[i * num_images_per_round : (i + 1) * num_images_per_round] for i in range(num_splits)]
    for batch in batches_of_dataframes:
        batch.reset_index()
        for horde_model_name in HORDE_MODEL_NAMES:
            results = batch.apply(
                lambda x: do_inference(df_row=x, horde_model_name=horde_model_name),
                axis=1,
                result_type="expand",
            )
            images_made_this_round = len([x for x in results.values if x])
            total_images_created += images_made_this_round
            skipped = num_images_per_round - images_made_this_round

            print("*" * 40)
            logger.info(f"Total images created so far: {total_images_created}")
            logger.info(f"Images skipped this round  : {skipped}")
            print("*" * 40)

        if total_images_created >= total_images:
            print()
            print()
            logger.info("Reached total_images (change with command line arg `--total_images N`).")
            logger.info(f"Total images created: {total_images_created}.")
            break


if __name__ == "__main__":
    argParsers = argparse.ArgumentParser()
    argParsers.add_argument(
        "--run_batch_name",
        type=str,
        default="condpatch",
        help="Name of the batch to insert in the filename.",
    )
    argParsers.add_argument(
        "--total_images",
        type=int,
        default=10,
        help="Total images to create. Should be a multiple of 5.",
    )
    argParsers.add_argument(
        "--seed",
        type=str,
        default="23113",
        help="The seed to use for all generations this run.",
    )
    argParsers.add_argument("--karras", action="store_true")
    args = argParsers.parse_args()
    main(run_batch_name=args.run_batch_name, seed=args.seed, karras=args.karras, total_images=args.total_images)
