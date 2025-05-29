import argparse
from dataclasses import dataclass, field, fields

# Define default hyperparameters
DEFAULT_TAU_ALPHA = 0.4
DEFAULT_TAU_BETA = 0.8
DEFAULT_GUIDANCE_SCALE = 3.0
DEFAULT_ALPHA_NOISE = 0.05
DEFAULT_TIMESTEPS = 50
DEFAULT_LAYERS_FOR_INJECTION = "all"
DEFAULT_INJECT_Q = True
DEFAULT_INJECT_K = True
DEFAULT_INJECT_V = False
DEFAULT_USE_PROMPT = True
DEFAULT_MIN_MASK_AREA_RATIO = 0.1  # Specific to run.py
DEFAULT_SEED = 42


@dataclass
class DitEditConfig:
    seed: int = field(
        default=DEFAULT_SEED,
        metadata={"help": f"Random seed for generation (default: {DEFAULT_SEED})."},
    )

    # Core DitEdit Hyperparameters
    tau_alpha: float = field(
        default=DEFAULT_TAU_ALPHA,
        metadata={"help": f"Value for TAU_ALPHA (default: {DEFAULT_TAU_ALPHA})"},
    )
    tau_beta: float = field(
        default=DEFAULT_TAU_BETA,
        metadata={"help": f"Value for TAU_BETA (default: {DEFAULT_TAU_BETA})"},
    )
    guidance_scale: float = field(
        default=DEFAULT_GUIDANCE_SCALE,
        metadata={"help": f"Guidance scale factor (default: {DEFAULT_GUIDANCE_SCALE})"},
    )
    alpha_noise: float = field(
        default=DEFAULT_ALPHA_NOISE,
        metadata={"help": f"Alpha noise parameter (default: {DEFAULT_ALPHA_NOISE})"},
    )
    timesteps: int = field(
        default=DEFAULT_TIMESTEPS,
        metadata={"help": f"Number of timesteps (default: {DEFAULT_TIMESTEPS})"},
    )
    layers_for_injection: str = field(
        default=DEFAULT_LAYERS_FOR_INJECTION,
        metadata={
            "choices": ["all", "vital"],
            "help": f"Layers for injection (default: {DEFAULT_LAYERS_FOR_INJECTION})",
        },
    )
    inject_k: bool = field(
        default=DEFAULT_INJECT_K,
        metadata={
            "help": f"Enable K injection. To disable, use --no-inject-k (default: {DEFAULT_INJECT_K})"
        },
    )
    inject_q: bool = field(
        default=DEFAULT_INJECT_Q,
        metadata={
            "help": f"Enable Q injection. To disable, use --no-inject-q (default: {DEFAULT_INJECT_Q})"
        },
    )
    inject_v: bool = field(
        default=DEFAULT_INJECT_V,
        metadata={
            "help": f"Enable V injection. To disable, use --no-inject-v (default: {DEFAULT_INJECT_V})"
        },
    )

    # Hyperparameters that might be more specific to `run.py` or have different defaults/handling in `run_on_benchmark.py`
    use_prompt_in_generation: bool = field(
        default=DEFAULT_USE_PROMPT,
        metadata={
            "help": f"Use prompt in generation. To disable, use --no-use-prompt-in-generation (default: {DEFAULT_USE_PROMPT})"
        },
    )
    min_mask_area_ratio: float = field(
        default=DEFAULT_MIN_MASK_AREA_RATIO,
        metadata={
            "help": f"Min FG mask area ratio (default: {DEFAULT_MIN_MASK_AREA_RATIO})"
        },
    )

    @classmethod
    def from_args(
        cls, parser: argparse.ArgumentParser, args: argparse.Namespace | None = None
    ):
        """
        Populates the dataclass fields from parsed argparse arguments.
        If args is None, it will parse arguments from sys.argv.
        The provided parser should already have arguments defined for the config fields.
        """
        if args is None:
            args = parser.parse_args()

        config_kwargs = {}
        for f in fields(cls):
            if hasattr(args, f.name):
                config_kwargs[f.name] = getattr(args, f.name)
        return cls(**config_kwargs)

    @staticmethod
    def add_arguments_to_parser(parser: argparse.ArgumentParser):
        """
        Adds arguments to the provided ArgumentParser instance
        based on the fields of this dataclass.
        """
        for f in fields(DitEditConfig):
            field_metadata = f.metadata.copy()
            arg_name = f'--{f.name.replace("_", "-")}'

            kwargs = {
                "help": field_metadata.pop("help", f.name),
            }

            if f.type == bool:
                # For boolean flags, argparse.BooleanOptionalAction is generally preferred
                # as it creates both --flag and --no-flag automatically.
                # The 'default' value is handled directly by this action.
                kwargs["action"] = argparse.BooleanOptionalAction
                kwargs["default"] = f.default
            else:
                kwargs["type"] = f.type
                kwargs["default"] = f.default

            if "choices" in field_metadata:
                kwargs["choices"] = field_metadata.pop("choices")

            # Handle cases where default is None for non-boolean types
            if f.type != bool and f.default is None:
                # If type hint is Optional[T], then 'default=None' is fine.
                # For other types, ensure 'type' is set.
                if not (
                    f.type.__class__.__name__ == "_UnionGenericAlias"
                    and type(None) in f.type.__args__
                ):
                    # This case might need specific handling if 'type' isn't automatically inferred well by argparse
                    pass

            parser.add_argument(arg_name, **kwargs)
        return parser

    def to_dict(self):
        """
        Converts the dataclass instance to a dictionary.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}


def parse_script_specific_args_run(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--bg_path", type=str, required=True, help="Path to the background image."
    )
    parser.add_argument(
        "--fg_path", type=str, required=True, help="Path to the foreground image."
    )
    parser.add_argument(
        "--bbox_path",
        type=str,
        required=True,
        help="Path to the bounding box mask image (black with a white rectangle).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output_composed_image.png",
        help="Path to save the composed image.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional prompt for the image generation.",
    )
    parser.add_argument(
        "--segm_mask_path",
        type=str,
        default=None,
        help="Optional path to an existing segmentation mask image.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to save intermediate images.",
    )
    return parser


def parse_script_specific_args_benchmark(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--run-on-first",
        type=int,
        default=-1,
        help="Run on the first N images from each category (default: -1 == run on all)",
    )
    parser.add_argument(
        "--random-samples-seed",
        type=int,
        default=42,
        help="Seed for random sampling from benchmark data (default: 42)",
    )
    parser.add_argument(
        "--random-samples",
        action="store_true",
        help="If set together with a positive number of --run-on-first, it will randomly sample that number of images from each category.",
    )
    parser.add_argument(
        "--skip-available",
        action="store_true",
        help="If set, images that were already will not be regenerated.",
        default=False,
    )
    parser.add_argument(
        "--save-output-images",
        action="store_true",
        help="Save the generated output images",
        default=True,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save the output images. If not passed, it will use the default output directory which is data/<domain>/<sample>",
    )
    return parser
