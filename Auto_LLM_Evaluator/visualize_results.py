import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import sys
import logging # Need logging config if running standalone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Define logger for standalone run

def visualize_scores(results_file_path: str, config: dict = None):
    """
    Loads evaluation results and generates visualizations using Seaborn
    to highlight LLM scoring performance and potential weaknesses.

    Args:
        results_file_path: Path to the saved TSV file containing evaluation results.
        config: Optional dictionary containing the evaluator configuration,
                used to get max scores if available.
    """
    logger.info(f"Loading results for visualization from: {results_file_path}")
    if not os.path.exists(results_file_path):
        logger.error(f"Results file not found: {results_file_path}")
        print(f"Error: Results file not found at {results_file_path}", file=sys.stderr)
        return

    try:
        df = pd.read_csv(results_file_path, sep='\t')
        logger.info(f"Loaded results DataFrame shape: {df.shape}")
        if df.empty:
            logger.warning("Results DataFrame is empty. No visualizations generated.")
            print("Warning: Results file is empty. Cannot generate visualizations.", file=sys.stderr)
            return
    except Exception as e:
        logger.error(f"Failed to load results file '{results_file_path}': {e}", exc_info=True)
        print(f"Error: Failed to load results file: {e}", file=sys.stderr)
        return

    # --- Identify Score Columns ---
    # Define expected score columns
    component_scores = [
        'rephraser_score',
        'source_score',
        'sentences_score_ingredient',
        'sentences_score_instruction'
    ]
    final_score_col = 'final_score'

    # Filter to columns actually present in the DataFrame
    present_component_scores = [col for col in component_scores if col in df.columns]
    present_final_score = final_score_col if final_score_col in df.columns else None

    if not present_component_scores:
         logger.warning("No component score columns found in the results file.")
         print("Warning: No component score columns found. Cannot generate component comparison plot.", file=sys.stderr)
    else:
         # --- Plot 1: Compare Component Score Distributions (Normalized) ---
        logger.info("Generating component score distribution plot...")
        df_normalized = df[present_component_scores].copy()

        # Normalize scores to be out of 1.0 for better comparison
        # Get max scores from config if available, otherwise use defaults
        max_rephraser = config['scoring']['max_points']['rephraser'] if config else 1.0
        max_source = config['scoring']['max_points']['source'] if config else 1.0
        max_ingredients = config['scoring']['max_points']['ingredients'] if config else 2.0
        max_instructions = config['scoring']['max_points']['instructions'] if config else 2.0

        if 'rephraser_score' in df_normalized:
            df_normalized['rephraser_score'] /= max_rephraser
        if 'source_score' in df_normalized:
            df_normalized['source_score'] /= max_source
        if 'sentences_score_ingredient' in df_normalized and max_ingredients > 0:
            df_normalized['sentences_score_ingredient'] /= max_ingredients
        if 'sentences_score_instruction' in df_normalized and max_instructions > 0:
             df_normalized['sentences_score_instruction'] /= max_instructions

        # Rename columns for clarity in the plot
        df_normalized.rename(columns={
             'rephraser_score': 'Rephraser (Norm)',
             'source_score': 'Source (Norm)',
             'sentences_score_ingredient': 'Ingredients (Norm)',
             'sentences_score_instruction': 'Instructions (Norm)'
        }, inplace=True)

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_normalized)
        # Or use violin plot: sns.violinplot(data=df_normalized, inner="quartile")
        plt.title('Distribution of Normalized Component Scores (0.0 to 1.0)')
        plt.ylabel('Normalized Score')
        plt.ylim(-0.05, 1.05) # Set Y limits from 0 to 1
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout() # Adjust layout
        # Optionally save the figure
        # plt.savefig("component_scores_boxplot.png")
        logger.info("Component score plot created.")


    if not present_final_score:
        logger.warning("Final score column not found in the results file.")
        print("Warning: Final score column not found. Cannot generate final score distribution plot.", file=sys.stderr)
    else:
         # --- Plot 2: Distribution of Final Scores ---
        logger.info("Generating final score distribution plot...")
        max_final = config['scoring']['max_points']['total'] if config else 6.0 # Get total from config or default

        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=present_final_score, kde=True, bins=min(20, len(df)//2 + 1) if len(df) > 1 else 1) # Adjust bins based on data size
        plt.title(f'Distribution of Final Scores (Max: {max_final:.1f})')
        plt.xlabel('Final Score')
        plt.ylabel('Count / Density')
        plt.xlim(-0.1, max_final + 0.1) # Set X limits
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        # Optionally save the figure
        # plt.savefig("final_score_histogram.png")
        logger.info("Final score plot created.")


    # Show plots if any were generated
    if present_component_scores or present_final_score:
        logger.info("Displaying plots...")
        plt.show()
    else:
         logger.info("No data to plot.")


# --- Add main block to make script runnable ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize LLM evaluation scores from a results file.")
    parser.add_argument(
        "results_file", # Make results file a positional argument
        type=str,
        help="Path to the TSV results file generated by the evaluator."
    )
    parser.add_argument(
         "-c", "--config-path", # Optional config for max scores
         type=str,
         default=None,
         help="Optional: Path to the YAML config file used during evaluation (for max score values)."
    )

    args = parser.parse_args()

    # Load config if provided
    eval_config = None
    if args.config_path:
        if os.path.exists(args.config_path):
            try:
                # Need yaml import
                import yaml
                with open(args.config_path, 'r') as f:
                     eval_config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {args.config_path} for score scaling.")
            except Exception as e:
                 logger.warning(f"Could not load or parse config file {args.config_path}: {e}. Using default max scores.")
                 print(f"Warning: Could not load config file {args.config_path}. Using default max scores.", file=sys.stderr)
        else:
            logger.warning(f"Config file specified but not found: {args.config_path}. Using default max scores.")
            print(f"Warning: Config file {args.config_path} not found. Using default max scores.", file=sys.stderr)


    visualize_scores(args.results_file, eval_config)
    logger.info("Visualization script finished.")

