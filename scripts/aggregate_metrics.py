import argparse
import json
import os
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# python run_aggregate_metrics.py  --tau-alpha 0.4  --tau-beta 0.8  --guidance-scale 3.0   --alpha-noise 0.05  --timesteps 50  --inject-k  --inject-v  --inject-q --layers all


# Define the methods and metrics based on your description
SAVE_SUFFIX = ""  # TODO: you might need to adjust this suffix
METHODS = ["Photoshop", "TF-ICON", "KV-EDIT", "ours"]
METRICS = [
    "hpsv2_score",
    "aesthetics_score",
    "background_mse",
    "clip_text_image",
    "dinov2_similarity",
]


def construct_metrics_filename(args):
    """Constructs the metrics filename based on CLI arguments."""
    return (
        f"alphanoise{args.alpha_noise}_timesteps{args.timesteps}"
        f"_Q{args.inject_q}_K{args.inject_k}_V{args.inject_v}"
        f"_taua{args.tau_alpha}_taub{args.tau_beta}"
        f"_guidance{args.guidance_scale}_{args.layers}-layers.json"
    )


def load_and_process_data(args):
    """
    Loads data from JSON files according to the specified structure and arguments.
    Returns a dictionary of aggregated scores and a dictionary of average scores.
    """
    benchmark_root_dir = (
        "benchmark_images_generations"  # TODO: you might need to adjust this path
    )
    aggregated_scores = {}
    # { benchmark_type: { metric: { method: [scores] } } }

    benchmark_types = [
        d
        for d in os.listdir(benchmark_root_dir)
        if os.path.isdir(os.path.join(benchmark_root_dir, d))
    ]

    if not benchmark_types:
        print(f"Error: No benchmark type subdirectories found in {benchmark_root_dir}")
        return {}, {}

    print(f"Found benchmark types: {', '.join(benchmark_types)}")

    for benchmark_type in benchmark_types:
        aggregated_scores[benchmark_type] = {
            metric: {method: [] for method in METHODS} for metric in METRICS
        }
        benchmark_type_path = os.path.join(benchmark_root_dir, benchmark_type)

        sample_folders = [
            d
            for d in os.listdir(benchmark_type_path)
            if os.path.isdir(os.path.join(benchmark_type_path, d))
        ]

        if not sample_folders:
            print(f"Warning: No sample folders found in {benchmark_type_path}")
            continue

        print(f"  Processing {benchmark_type} with {len(sample_folders)} samples...")

        for sample_folder in sample_folders:
            sample_folder_path = os.path.join(benchmark_type_path, sample_folder)
            metrics_file_name = construct_metrics_filename(args)
            json_file_path = os.path.join(sample_folder_path, metrics_file_name)

            if os.path.exists(json_file_path):
                try:
                    with open(json_file_path, "r") as f:
                        score_dict = json.load(f)

                    for method_name, method_scores in score_dict.items():
                        if method_name in METHODS:
                            for metric_name, score_value in method_scores.items():
                                if metric_name in METRICS:
                                    aggregated_scores[benchmark_type][metric_name][
                                        method_name
                                    ].append(score_value)
                                else:
                                    print(
                                        f"Warning: Unknown metric '{metric_name}' in method '{method_name}' in file {json_file_path}"
                                    )
                        # else:
                        #     print(f"Warning: Unknown method '{method_name}' in file {json_file_path}")
                except json.JSONDecodeError:
                    print(f"Error: Could not decode JSON from {json_file_path}")
                except Exception as e:
                    print(f"Error processing file {json_file_path}: {e}")
            else:
                print(f"Warning: Metrics file not found: {json_file_path}")

    print(
        "Converting Scores to DATAFRAME and saving as",
        f"aggregated_scores_{SAVE_SUFFIX}.csv",
    )

    # Convert aggregated_scores to a DataFrame for better visualization
    df = pd.DataFrame.from_dict(
        {
            (i, j): aggregated_scores[i][j]
            for i in aggregated_scores.keys()
            for j in aggregated_scores[i].keys()
        },
        orient="index",
    )
    df = df.explode(column=METHODS)
    df.to_csv(f"aggregated_scores_{SAVE_SUFFIX}.csv", index=True)

    # Calculate averages
    average_scores = {}
    # { benchmark_type: { metric: { method: average_score } } }
    for benchmark_type, metrics_data in aggregated_scores.items():
        average_scores[benchmark_type] = {}
        for metric_name, methods_data in metrics_data.items():
            average_scores[benchmark_type][metric_name] = {}
            for method_name, scores_list in methods_data.items():
                if scores_list:
                    average_scores[benchmark_type][metric_name][
                        method_name
                    ] = statistics.mean(scores_list)
                else:
                    raise ValueError(
                        f"No scores found for {method_name} in {metric_name} for {benchmark_type}"
                    )

    return aggregated_scores, average_scores


def plot_histograms(average_scores, cli_args_for_title):
    """
    Generates and saves bar charts (histograms) based on the average scores.
    """
    if not average_scores:
        print("No average scores to plot.")
        return

    output_dir = "./plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Construct a part of the title from CLI args to identify the plot set
    config_str = (
        f"AN{cli_args_for_title.alpha_noise}_T{cli_args_for_title.timesteps}"
        f"_Q{cli_args_for_title.inject_q}_K{cli_args_for_title.inject_k}_V{cli_args_for_title.inject_v}"
        f"_Ta{cli_args_for_title.tau_alpha}_Tb{cli_args_for_title.tau_beta}"
        f"_GS{cli_args_for_title.guidance_scale}_{cli_args_for_title.layers}L"
    )

    for benchmark_type, metrics_data in average_scores.items():
        for metric_name, methods_data in metrics_data.items():
            method_names = list(methods_data.keys())
            avg_values = [methods_data[method] for method in method_names]

            # Ensure we only plot if we have methods with data
            if not method_names or not any(
                avg_values
            ):  # if all avg_values are 0 or method_names is empty
                raise ValueError(
                    f"No valid data to plot for {benchmark_type} - {metric_name}"
                )

            plt.figure(figsize=(10, 6))
            bars = plt.bar(
                method_names,
                avg_values,
                color=["skyblue", "lightcoral", "lightgreen", "gold"],
            )

            plt.xlabel("Method")
            plt.ylabel(f"Average {metric_name}")
            plt.title(
                f"Average {metric_name} for {benchmark_type}\nConfig: {config_str}",
                fontsize=10,
            )
            plt.xticks(rotation=45, ha="right")
            plt.grid(axis="y", linestyle="--")
            plt.tight_layout()  # Adjust layout to make room for rotated x-axis labels

            # Add text labels on top of each bar
            for bar in bars:
                yval = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    yval,
                    f"{yval:.3f}",
                    va="bottom",
                    ha="center",
                )  # Adjust format as needed

            plot_filename = f"{benchmark_type}_{metric_name}_{config_str}.png"
            plot_filepath = os.path.join(output_dir, plot_filename)

            try:
                plt.savefig(plot_filepath)
                print(f"Saved plot: {plot_filepath}")
            except Exception as e:
                print(f"Error saving plot {plot_filepath}: {e}")
            plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate histograms from benchmark metrics."
    )

    # Arguments for constructing the filename (must match those used to generate the files)
    parser.add_argument(
        "--tau-alpha",
        type=float,
        default=0.4,
        help="Value for TAU_ALPHA (default: 0.4)",
    )
    parser.add_argument(
        "--tau-beta", type=float, default=0.8, help="Value for TAU_BETA (default: 0.8)"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.0,
        help="Guidance scale factor (default: 3.0)",
    )
    parser.add_argument(
        "--alpha-noise",
        type=float,
        default=0.05,
        help="Alpha noise parameter (default: 0.05)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=50, help="Number of timesteps (default: 50)"
    )
    # Assuming these were the names used for the boolean args when generating filenames
    # If they were different (e.g. just INJECT_Q), adjust here.
    parser.add_argument(
        "--inject-k", action="store_true", help="K injection was enabled"
    )
    parser.add_argument(
        "--inject-q", action="store_true", help="Q injection was enabled"
    )
    parser.add_argument(
        "--inject-v", action="store_true", help="V injection was enabled"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="vital",
        choices=["all", "vital"],
        help="Layers where injection was performed (all or vital, default: vital)",
    )

    # Add any other args that were part of your filename if needed

    args = parser.parse_args()

    print("Starting data processing...")

    # Construct the expected filename pattern once to show the user
    # Note: This uses the parsed args, which are boolean for inject flags.
    # The construct_metrics_filename function handles converting these to strings "True"/"False".
    example_filename = construct_metrics_filename(args)
    print(f"Expecting JSON filenames like: {example_filename}")

    _, average_scores = load_and_process_data(args)

    if average_scores:
        print("\nPlotting histograms...")
        plot_histograms(average_scores, args)
        print("\nProcessing complete.")
    else:
        print("\nNo data processed or averaged. Exiting.")


if __name__ == "__main__":
    main()
