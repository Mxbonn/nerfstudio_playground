"""
Benchmarking script for nerfstudio paper.

- nerfacto and instant-ngp methods on mipnerf360 data
- nerfacto ablations
"""

import threading
import time
from pathlib import Path
from typing import Union
from dataclasses import dataclass

import tyro
from typing_extensions import Annotated

import GPUtil

from nerfstudio.configs.base_config import PrintableConfig
from nerfstudio.utils.scripts import run_command

# for the mipnerf360 experiments
mipnerf360_capture_names = [
    "bicycle",
    "garden",
    "stump",
    "room",
    "counter",
    "kitchen",
    "bonsai",
]  # 7 splits
mipnerf360_table_rows = [
    # nerfacto method
    (
        "nerfacto-mipnerf360-test",
        "nerfacto-mip360",
        "mipnerf360-data",
    ),
]


def launch_experiments(
    capture_names,
    table_rows,
    data_path: Path = Path("data/nerfstudio"),
    dry_run: bool = False,
):
    """Launch the experiments."""

    # make a list of all the jobs that need to be fun
    jobs = []
    for capture_name in capture_names:
        for table_row_name, method, table_row_command in table_rows:
            command = " ".join(
                (
                    f"ns-train {method}",
                    "--vis wandb",
                    f"--data { data_path / capture_name}",
                    "--output-dir outputs/mipnerf360data",
                    "--steps-per-eval-batch 0 --steps-per-eval-image 0",
                    "--steps-per-eval-all-images 5000 --max-num-iterations 30001",
                    f"--experiment-name {capture_name}_{table_row_name}",
                    # extra_string,
                    table_row_command,
                )
            )
            jobs.append(command)

    while jobs:
        # get GPUs that capacity to run these jobs
        gpu_devices_available = GPUtil.getAvailable(
            order="first", limit=10, maxMemory=0.1
        )

        print("Available GPUs: ", gpu_devices_available)

        # thread list
        threads = []
        while gpu_devices_available and jobs:
            gpu = gpu_devices_available.pop(0)
            command = f"CUDA_VISIBLE_DEVICES={gpu} " + jobs.pop(0)

            def task():
                print("Starting command:\n", command)
                if not dry_run:
                    _ = run_command(command, verbose=False)
                print("Finished command:\n", command)

            threads.append(threading.Thread(target=task))
            threads[-1].start()

            # NOTE(ethan): here we need a delay, otherwise the wandb/tensorboard naming is messed up... not sure why
            if not dry_run:
                time.sleep(5)

        # wait for all threads to finish
        for t in threads:
            t.join()

        print("Finished all threads")


@dataclass
class Benchmark(PrintableConfig):
    """Benchmark code."""

    dry_run: bool = False

    def main(self, dry_run: bool = False) -> None:
        """Run the code."""
        raise NotImplementedError


@dataclass
class BenchmarkMipNeRF360(Benchmark):
    """Benchmark MipNeRF-360."""

    def main(self, dry_run: bool = False):
        launch_experiments(
            mipnerf360_capture_names,
            mipnerf360_table_rows,
            data_path=Path("data/nerfstudio-data-mipnerf360"),
            dry_run=dry_run,
        )

def main(
    benchmark: Benchmark,
):
    """Script to run the benchmark experiments for the Nerfstudio paper.
    - nerfacto-on-mipnerf360: The MipNeRF-360 experiments on the MipNeRF-360 Dataset.
    - nerfacto-ablations: The Nerfacto ablations on the Nerfstudio Dataset.

    Args:
        benchmark: The benchmark to run.
    """
    benchmark.main(dry_run=benchmark.dry_run)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(BenchmarkMipNeRF360))


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Commands)  # noqa
