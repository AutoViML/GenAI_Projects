##############################################################################
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
# Auto Alignment for LLM Output
# Developed by Jennifer Liang and Ram Seshadri
# Last Updated: April 2025
#
# Note: This is not an officially supported Google product.
##############################################################################
"""
Provides a configurable framework for evaluating Large Language Model (LLM)
responses against ground truth data, focusing on line-by-line comparisons without using LLM's.

This module implements a multi-stage evaluation process using:
1. Rephraser Evaluation: Compares semantic similarity of rephrased queries.
2. Final Answer Evaluation:
   - Source Score: Compares identified sources (e.g., recipe numbers).
   - Ingredient Score: Compares semantic similarity of ingredient lists.
   - Instruction Score: Compares semantic similarity of instruction steps.

Scores are calculated based on matches, with penalties for incorrect or missing
elements from the LLM response compared to the ground truth. Configuration is
loaded from a YAML file.
"""

import os
import sys
import logging
import warnings
from typing import List, Dict, Any, Tuple, Optional, Union

import pandas as pd
import numpy as np
import regex as re
import yaml
from sentence_transformers import util  # For cosine similarity calculation
import argparse

# Suppress specific warnings from SentenceTransformer or dependencies if needed
warnings.filterwarnings("ignore", category=FutureWarning)

# Attempt to import Vertex AI libraries, providing user guidance if unavailable
try:
    import vertexai
    from google.cloud import aiplatform
    from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
    from google.api_core import exceptions as google_exceptions
except ImportError:
    print(
        "ERROR: google-cloud-aiplatform or vertexai not found. "
        "Please install using: pip install google-cloud-aiplatform"
    )
    sys.exit(1)


# --- Module Level Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default "No Answer" phrase for semantic comparison
DEFAULT_NO_ANSWER_PHRASE = "No answer to this question"

# --- Helper Functions ---

def _load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from {config_path}")
        # Basic validation (can be expanded)
        required_sections = ['embedding_model_id', 'column_names', 'regex_patterns', 'thresholds', 'scoring', 'no_answer_keywords']
        if not all(section in config for section in required_sections):
            raise ValueError(f"Config file {config_path} is missing one or more required sections: {required_sections}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {config_path}: {e}")
        raise
    except ValueError as e:
        logger.error(f"Configuration validation error: {e}")
        raise

def _find_identifiers(text: Optional[str], pattern: str) -> List[str]:
    """
    Finds and extracts unique identifiers from text using a regex pattern.

    Args:
        text: The input string to search. Handles None or pd.NA.
        pattern: The regex pattern to use. Must contain one capturing group for the identifier.

    Returns:
        A sorted list of unique identifiers found. Returns an empty list if
        text is None/NA or no identifiers are found.
    """
    if pd.isna(text):
        return []
    try:
        # Ensure pattern has exactly one capturing group
        if re.compile(pattern).groups != 1:
             logger.warning(f"Regex pattern '{pattern}' should have exactly one capturing group for identifier extraction.")
             # Attempt to proceed, but might not capture correctly
        matches = re.findall(pattern, str(text))
        return sorted(list(set(matches))) if matches else []
    except re.error as e:
        logger.error(f"Invalid regex pattern '{pattern}': {e}")
        return [] # Return empty on regex error to avoid crashing apply()
    except Exception as e:
        logger.error(f"Unexpected error in _find_identifiers for text '{str(text)[:50]}...': {e}")
        return []


def _extract_section_regex(text: Optional[str], start_pattern: str, end_pattern: Optional[str] = None, cleanup_pattern: str = '') -> str:
    """
    Extracts a section of text defined by start and optional end patterns using regex.

    Args:
        text: The input string. Handles None or pd.NA.
        start_pattern: Regex pattern marking the beginning of the section (case-insensitive).
                       The content *after* this pattern is captured.
        end_pattern: Optional regex pattern marking the end of the section (case-insensitive).
                     If provided, text between start_pattern and end_pattern is captured.
        cleanup_pattern: Regex pattern for characters/text to remove from the extracted section.

    Returns:
        The extracted and cleaned section as a string, or "" if not found or input is invalid.
    """
    if pd.isna(text):
        return ""

    text_str = str(text)
    try:
        if end_pattern:
            # Regex to find text between start_pattern and end_pattern
            full_pattern = rf"{start_pattern}\s*(.*?)(?:\s*[\"'\n]*\s*{end_pattern})"
            match = re.search(full_pattern, text_str, re.IGNORECASE | re.DOTALL)
        else:
            # Regex to find text after start_pattern to the end
            full_pattern = rf"{start_pattern}\s*(.*)"
            match = re.search(full_pattern, text_str, re.IGNORECASE | re.DOTALL)

        if match:
            extracted_text = match.group(1).strip()
            if cleanup_pattern:
                # Perform cleanup using the provided pattern
                 extracted_text = re.sub(cleanup_pattern, "", extracted_text)
            return extracted_text.strip() # Strip again after potential cleanup
        else:
            # Log if start pattern was found but not the full section (useful for debugging patterns)
            if re.search(start_pattern, text_str, re.IGNORECASE | re.DOTALL):
                 logger.debug(f"Found start pattern '{start_pattern}' but couldn't extract section. End pattern: '{end_pattern}'.")
            return ""
    except re.error as e:
        logger.error(f"Regex error during section extraction. Start:'{start_pattern}', End:'{end_pattern}'. Error: {e}")
        return ""
    except Exception as e:
        logger.error(f"Unexpected error in _extract_section_regex: {e}")
        return ""


# --- Evaluator Class ---

class LLMEvaluator:
    """
    Orchestrates the evaluation of LLM responses based on provided configuration.
    """

    def __init__(self, config_path: str):
        """
        Initializes the LLMEvaluator.

        Args:
            config_path: Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If the config file is invalid YAML.
            ValueError: If the configuration is missing required sections or values.
            google_exceptions.NotFound: If the specified project/location is invalid.
            google_exceptions.PermissionDenied: If authentication fails or lacks permissions.
            ImportError: If required Google Cloud libraries are not installed.
        """
        self.config = _load_config(config_path)
        self._validate_config() # Perform more detailed validation

        self.col_names = self.config['column_names']
        self.regex_patterns = self.config['regex_patterns']
        self.thresholds = self.config['thresholds']
        self.scoring_params = self.config['scoring']
        self.no_answer_kws = [kw.lower() for kw in self.config['no_answer_keywords']]
        self.embedding_model_id = self.config['embedding_model_id']

        self._initialize_vertexai()
        self._load_embedding_model()


    def _validate_config(self):
        """Performs detailed validation of the loaded configuration."""
        # Example validation: Check if necessary keys exist within nested dicts
        if not all(k in self.config['column_names'] for k in ['case_id', 'ground_truth_rephrased', 'llm_rephrased', 'ground_truth_final_answer', 'llm_final_answer']):
             raise ValueError("Config missing required keys under 'column_names'.")
        if not all(k in self.config['regex_patterns'] for k in ['source_identifier', 'ingredients_start', 'instructions_start', 'text_cleanup', 'ignorable_sentence']):
            raise ValueError("Config missing required keys under 'regex_patterns'.")
        if not all(k in self.config['thresholds'] for k in ['rephraser_similarity', 'ingredient_similarity', 'instruction_similarity', 'no_answer_similarity']):
             raise ValueError("Config missing required keys under 'thresholds'.")
        # Add more checks for scoring parameters, types, ranges etc. as needed
        logger.info("Configuration validated successfully.")


    def _initialize_vertexai(self):
        """Initializes Vertex AI connection using environment variables or config."""
        try:
            project_id = self.config.get('project_id') or os.environ.get("GOOGLE_CLOUD_PROJECT")
            location = self.config.get('location') or os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

            if not project_id:
                raise ValueError(
                    "Google Cloud Project ID not found. Please set the GOOGLE_CLOUD_PROJECT "
                    "environment variable or add 'project_id' to the config file."
                )

            logger.info(f"Initializing Vertex AI for Project ID: {project_id}, Location: {location}")
            vertexai.init(project=project_id, location=location)
            # Optionally create a GenAI client if needed for other tasks later
            # self.genai_client = genai.Client(vertexai=True, project=project_id, location=location)
            logger.info("Vertex AI initialized successfully.")
        except (ValueError, google_exceptions.NotFound, google_exceptions.PermissionDenied) as e:
            logger.error(f"Failed to initialize Vertex AI: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during Vertex AI initialization: {e}", exc_info=True)
            raise

    def _load_embedding_model(self):
        """Loads the specified Vertex AI text embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_id}")
            self.embedding_model = TextEmbeddingModel.from_pretrained(self.embedding_model_id)
            logger.info(f"Embedding model {self.embedding_model_id} loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{self.embedding_model_id}': {e}", exc_info=True)
            # Decide if this is fatal or if execution can continue without embeddings
            raise ValueError(f"Could not load embedding model: {self.embedding_model_id}") from e

    def _get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Gets embeddings for a list of texts using the loaded Vertex AI model."""
        if not texts:
            return []
        try:
            # Handle potential API limits by batching if necessary (simple example)
            # Vertex AI embeddings API usually handles batching well internally
            # Max 250 texts per call for text-embedding-004 as of late 2023
            BATCH_SIZE = 250
            all_embeddings = []
            for i in range(0, len(texts), BATCH_SIZE):
                 batch = texts[i:i + BATCH_SIZE]
                 # Check for empty strings in batch, replace with a placeholder if model requires non-empty
                 processed_batch = [t if t else " " for t in batch] # Model might handle empty, check docs
                 embeddings = self.embedding_model.get_embeddings(processed_batch)
                 all_embeddings.extend([np.array(e.values) for e in embeddings])
            return all_embeddings

        except google_exceptions.ApiException as e:
            logger.error(f"Vertex AI API error during embedding: {e}", exc_info=True)
            raise # Re-raise to signal failure
        except Exception as e:
            logger.error(f"Unexpected error getting embeddings: {e}", exc_info=True)
            raise

    def _calculate_similarity(self, text1: Optional[str], text2: Optional[str]) -> float:
        """Calculates cosine similarity between two texts using the Vertex AI model."""
        if pd.isna(text1) or pd.isna(text2) or not text1 or not text2:
            return 0.0  # Return 0 similarity for empty or invalid inputs

        try:
            embeddings = self._get_embeddings([str(text1), str(text2)])
            if len(embeddings) != 2:
                 logger.warning(f"Could not get embeddings for similarity calculation. Texts: '{str(text1)[:50]}...', '{str(text2)[:50]}...'")
                 return 0.0

            # util.cos_sim expects tensors or numpy arrays
            cosine_sim = util.cos_sim(embeddings[0], embeddings[1])
            return max(0.0, float(cosine_sim.item())) # Ensure value is non-negative float
        except Exception as e:
             logger.warning(f"Failed similarity calculation between '{str(text1)[:50]}...' and '{str(text2)[:50]}...': {e}", exc_info=False) # Log less verbosely for apply()
             return 0.0 # Return 0 on error during apply

    def _source_score(self, llm_sources: List[str], gt_sources: List[str]) -> Tuple[float, float, float, float, float]:
        """Calculates the source score based on matches and penalties."""
        if not gt_sources: # Handle case where ground truth has no sources
             # If LLM also has no sources, score is 1 (perfect match for 'no sources')
             # If LLM provides sources incorrectly, score should be 0.
            return (1.0, 0.0, 0.0, 0.0, 0.0) if not llm_sources else (0.0, 0.0, 0.0, 0.0, 0.0)
            # Original logic returned 0,0,0,0,0 - changed to handle no-GT-sources case slightly differently.

        if not llm_sources: # Handle case where LLM provided no sources but GT expected some
            llm_sources = [] # Ensure it's an empty list for set operations

        points_per_source = round(self.scoring_params['max_points']['source'] / len(gt_sources), 4) # Use configured max points

        gt_set = set(gt_sources)
        llm_set = set(llm_sources)

        correct_sources_count = len(gt_set & llm_set)
        incorrect_llm_sources_count = len(llm_set - gt_set)
        missed_gt_sources_count = len(gt_set - llm_set)

        # Use penalty multipliers from config
        penalty_incorrect = self.scoring_params['penalties']['incorrect_source']
        penalty_missed = self.scoring_params['penalties']['missed_source']

        correct_sources_score = correct_sources_count * points_per_source
        incorrect_source_penalty = penalty_incorrect * points_per_source * incorrect_llm_sources_count
        missed_source_penalty = penalty_missed * points_per_source * missed_gt_sources_count

        final_score = round(max(0, (correct_sources_score - incorrect_source_penalty - missed_source_penalty)), 4)

        return final_score, points_per_source, correct_sources_score, incorrect_source_penalty, missed_source_penalty


    def _extract_ingredients_and_instructions(self, llm_text: Optional[str], gt_text: Optional[str]) -> Tuple[str, str, str, str]:
        """Extracts ingredients and instructions using configured regex patterns."""

        ingred_start_pattern = self.regex_patterns['ingredients_start']
        instr_start_pattern = self.regex_patterns['instructions_start']
        cleanup_pattern = self.regex_patterns['text_cleanup']

        # --- Process LLM Text ---
        llm_ingredients = _extract_section_regex(llm_text, ingred_start_pattern, instr_start_pattern, cleanup_pattern)
        llm_instructions = _extract_section_regex(llm_text, instr_start_pattern, None, cleanup_pattern) # Instructions go to the end

        # --- Process GT Text ---
        gt_ingredients = _extract_section_regex(gt_text, ingred_start_pattern, instr_start_pattern, cleanup_pattern)
        gt_instructions = _extract_section_regex(gt_text, instr_start_pattern, None, cleanup_pattern)

        return llm_ingredients, llm_instructions, gt_ingredients, gt_instructions


    def _evaluate_content_similarity(self, llm_text: str, gt_text: str, threshold: float, max_score: float, penalty_extra: float, penalty_missed: float, ignore_pattern_str: str) -> Tuple[float, int, float, int, float, float, float, int, int, int, int, int, int]:
        """
        Evaluates content similarity between LLM and GT text (for ingredients/instructions).
        Uses the configured Vertex AI embedding model.

        Returns:
            Tuple containing detailed scoring metrics (score, counts, penalties etc.).
        """
        if not llm_text or not gt_text or llm_text.lower() == "none" or gt_text.lower() == "none":
             # Return default zero/empty values if inputs are invalid
            return (0.0, 0, 0.0, 0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0)

        try:
            # Split texts into sentences
            gt_sentences_raw = gt_text.split("\n")
            gt_sentences = [sent.strip() for sent in gt_sentences_raw if sent.strip()]
            llm_sentences_raw = llm_text.split("\n")
            llm_sentences = [sent.strip() for sent in llm_sentences_raw if sent.strip()]

            num_gt_sentences = len(gt_sentences)
            num_llm_sentences = len(llm_sentences)

            if num_gt_sentences == 0 and num_llm_sentences == 0:
                 return (max_score, 0, 0.0, 0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0) # Perfect score if both empty
            if num_gt_sentences == 0 or num_llm_sentences == 0:
                 # If one is empty and the other isn't, score is 0 (unless all sentences are ignorable)
                 # This logic is handled below by penalties. Set base score to 0.
                 pass # Proceed to calculate penalties

            # Get embeddings only if there are sentences to compare
            gt_embeddings = []
            llm_embeddings = []
            similarities = np.array([[]]) # Default empty
            if num_gt_sentences > 0 and num_llm_sentences > 0:
                gt_embeddings = self._get_embeddings(gt_sentences)
                llm_embeddings = self._get_embeddings(llm_sentences)
                # Ensure we got embeddings before calculating similarity
                if len(gt_embeddings) == num_gt_sentences and len(llm_embeddings) == num_llm_sentences:
                    similarities = util.cos_sim(np.array(llm_embeddings), np.array(gt_embeddings)).numpy() # Convert tensor to numpy
                else:
                    logger.warning("Mismatch in embedding count, cannot calculate similarities.")
                    # Handle error state - perhaps return 0 score or raise
                    return (0.0, num_gt_sentences, 0.0, num_llm_sentences, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0)

            # Compile ignore pattern once
            try:
                ignore_re = re.compile(ignore_pattern_str, re.IGNORECASE) if ignore_pattern_str else None
            except re.error:
                logger.warning(f"Invalid regex pattern for ignorable sentences: '{ignore_pattern_str}'. Ignoring pattern.")
                ignore_re = None

            # --- Sentence Matching and Counting ---
            matched_gt_indices = set()
            matched_llm_indices = set()
            gt_not_in_llm_count = 0
            extra_sentences_count = 0
            gt_transition_sentence_count = 0
            llm_transition_sentence_count = 0
            matched_sentences_count = 0 # Count of GT sentences matched

            # Iterate through GT sentences to find best match in LLM
            for gt_idx, gt_sent in enumerate(gt_sentences):
                is_gt_ignorable = bool(ignore_re and ignore_re.fullmatch(gt_sent)) # Use fullmatch for patterns like ^[,"]$
                if is_gt_ignorable:
                    gt_transition_sentence_count += 1
                    continue # Skip matching for ignorable GT sentences

                best_llm_match_idx = -1
                max_sim = -1.0
                if num_llm_sentences > 0 and similarities.size > 0: # Check if similarities were calculated
                     # Find the LLM sentence with the highest similarity to this GT sentence
                    best_llm_match_idx = np.argmax(similarities[:, gt_idx])
                    max_sim = similarities[best_llm_match_idx, gt_idx]

                # Check if match is above threshold and the LLM sentence hasn't been matched yet
                if max_sim >= threshold and best_llm_match_idx not in matched_llm_indices:
                     matched_sentences_count += 1
                     matched_gt_indices.add(gt_idx)
                     matched_llm_indices.add(best_llm_match_idx)
                else:
                     # This GT sentence (non-ignorable) was not matched sufficiently
                    gt_not_in_llm_count += 1

            # Iterate through LLM sentences to find those not matched to any GT sentence (extras)
            for llm_idx, llm_sent in enumerate(llm_sentences):
                 is_llm_ignorable = bool(ignore_re and ignore_re.fullmatch(llm_sent))
                 if is_llm_ignorable:
                      llm_transition_sentence_count += 1
                      continue # Ignore transition LLM sentences for penalty

                 if llm_idx not in matched_llm_indices:
                      # This LLM sentence (non-ignorable) did not match any GT sentence sufficiently
                      extra_sentences_count += 1


            # --- Scoring Calculation ---
            num_scorable_gt_sentences = num_gt_sentences - gt_transition_sentence_count
            if num_scorable_gt_sentences <= 0:
                 # If all GT sentences were ignorable, score depends only on extra LLM sentences
                points_per_sentence = 0.00
                correct_sentences_score = max_score if extra_sentences_count == 0 else 0.0 # Full points if no extra LLM sentences
            else:
                 points_per_sentence = round(max_score / num_scorable_gt_sentences, 4)
                 correct_sentences_score = matched_sentences_count * points_per_sentence

            extra_sentences_penalty = extra_sentences_count * penalty_extra * points_per_sentence
            gt_not_in_llm_penalty = gt_not_in_llm_count * penalty_missed * points_per_sentence

            # Final score calculation
            final_score = round(max(0, min(max_score, correct_sentences_score - extra_sentences_penalty - gt_not_in_llm_penalty)), 4)

            # Count exact matches (optional metric)
            num_equal_sentences = 0
            if num_gt_sentences > 0 and num_llm_sentences > 0: # Requires both lists non-empty
                # Simple exact match check - could be refined
                gt_set = set(gt_sentences)
                llm_set = set(llm_sentences)
                num_equal_sentences = len(gt_set.intersection(llm_set))


            return (
                final_score, num_gt_sentences, points_per_sentence, num_llm_sentences,
                correct_sentences_score, extra_sentences_penalty, gt_not_in_llm_penalty,
                num_equal_sentences, matched_sentences_count, gt_not_in_llm_count,
                extra_sentences_count, gt_transition_sentence_count, llm_transition_sentence_count
            )

        except Exception as e:
            logger.error(f"Error during content similarity evaluation: {e}", exc_info=True)
            # Return default zero/empty values on unexpected error
            return (0.0, 0, 0.0, 0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0)


    def _apply_content_evaluation(self, df: pd.DataFrame, text_type: str) -> pd.DataFrame:
        """Applies content evaluation (ingredients or instructions) to the DataFrame."""
        logger.info(f"Starting content evaluation for: {text_type}")
        if text_type not in ['ingredient', 'instruction']:
            raise ValueError("text_type must be 'ingredient' or 'instruction'")

        llm_col = f'llm_{text_type}s_extracted' # e.g., llm_ingredients_extracted
        gt_col = f'gt_{text_type}s_extracted'
        threshold = self.thresholds[f'{text_type}_similarity']
        max_score = self.scoring_params['max_points'][f'{text_type}s'] # e.g., ingredients
        penalty_extra = self.scoring_params['penalties']['extra_sentence']
        penalty_missed = self.scoring_params['penalties']['missed_sentence']
        ignore_pattern = self.regex_patterns['ignorable_sentence']
        case_id_col = self.col_names['case_id']

        results = []
        for _, row in df.iterrows():
            score_tuple = self._evaluate_content_similarity(
                row[llm_col],
                row[gt_col],
                threshold,
                max_score,
                penalty_extra,
                penalty_missed,
                ignore_pattern
            )
            # Create dict for this row's results
            result_dict = {
                case_id_col: row[case_id_col], # Ensure case_id is included for merging
                f'sentences_score_{text_type}': score_tuple[0],
                f'num_gt_sentences_{text_type}': score_tuple[1],
                f'points_per_sentence_{text_type}': score_tuple[2],
                f'num_llm_sentences_{text_type}': score_tuple[3],
                f'correct_sentences_score_{text_type}': score_tuple[4],
                f'extra_sentences_penalty_{text_type}': score_tuple[5],
                f'gt_not_in_llm_penalty_{text_type}': score_tuple[6],
                f'num_equal_sentences_{text_type}': score_tuple[7],
                f'matched_sentences_count_{text_type}': score_tuple[8],
                f'gt_not_in_llm_count_{text_type}': score_tuple[9],
                f'extra_sentences_count_{text_type}': score_tuple[10],
                f'gt_transition_sentence_count_{text_type}': score_tuple[11],
                f'llm_transition_sentence_count_{text_type}': score_tuple[12],
            }
            results.append(result_dict)

        # Convert list of dicts to DataFrame
        results_df = pd.DataFrame(results)
        logger.info(f"Finished content evaluation for: {text_type}")
        return results_df

    def _no_answer_case(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies and applies special scoring for "No Answer" cases based on config.
        Modifies the DataFrame in place for rows identified as "No Answer".
        """
        logger.info("Applying 'No Answer' case logic...")
        gt_col = self.col_names['ground_truth_final_answer']
        llm_col = self.col_names['llm_final_answer']
        similarity_threshold = self.thresholds['no_answer_similarity']
        no_answer_scores = self.scoring_params['no_answer_scores']

        modified_count = 0
        for index, row in df.iterrows():
            try:
                gt_answer = str(row[gt_col]).strip().lower() if pd.notna(row[gt_col]) else ""
                llm_answer = str(row[llm_col]).strip().lower() if pd.notna(row[llm_col]) else ""

                # Skip if either answer is empty
                if not gt_answer or not llm_answer:
                    continue

                # Check 1: Keyword match
                gt_has_keyword = any(keyword in gt_answer for keyword in self.no_answer_kws)
                llm_has_keyword = any(keyword in llm_answer for keyword in self.no_answer_kws)

                # Check 2: Semantic similarity match
                gt_similarity = self._calculate_similarity(gt_answer, DEFAULT_NO_ANSWER_PHRASE)
                llm_similarity = self._calculate_similarity(llm_answer, DEFAULT_NO_ANSWER_PHRASE)

                is_no_answer_case = (gt_has_keyword and llm_has_keyword) or \
                                    (gt_similarity >= similarity_threshold and llm_similarity >= similarity_threshold)

                if is_no_answer_case:
                    # Apply predefined scores for this specific case
                    df.loc[index, 'source_score'] = no_answer_scores['source']
                    df.loc[index, 'sentences_score_ingredient'] = no_answer_scores['ingredients']
                    df.loc[index, 'sentences_score_instruction'] = no_answer_scores['instructions']
                    # Assuming 'final_score' column exists or will be calculated later
                    df.loc[index, 'final_score_override'] = no_answer_scores['final'] # Use a flag/override column
                    modified_count += 1
                    logger.debug(f"Case {row.get(self.col_names['case_id'], index)} identified as 'No Answer'. Applying override scores.")

            except KeyError as e:
                logger.error(f"Column '{e}' not found while processing 'No Answer' logic for index {index}. Skipping row.", exc_info=False)
            except Exception as e:
                logger.error(f"Unexpected error processing 'No Answer' logic for index {index}: {e}", exc_info=True) # Log full trace for unexpected

        logger.info(f"Finished 'No Answer' case logic. Applied override scores to {modified_count} rows.")
        return df


    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the full LLM evaluation pipeline on the input DataFrame.

        Args:
            df: The input Pandas DataFrame containing the data to evaluate.
                Must include columns specified in the 'column_names' section
                of the configuration file.

        Returns:
            A Pandas DataFrame containing the original data plus appended columns
            with evaluation scores and intermediate results.

        Raises:
            ValueError: If the input DataFrame is missing required columns.
            Exception: If errors occur during the evaluation pipeline.
        """
        logger.info("Starting LLM evaluation pipeline...")

        # --- Input Validation ---
        required_cols = list(self.col_names.values())
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}")

        # Create a copy to avoid modifying the original DataFrame
        eval_df = df.copy()
        case_id_col = self.col_names['case_id']

        # --- Phase 1: Rephraser Evaluation ---
        logger.info("Phase 1: Evaluating Rephraser Similarity...")
        gt_rephrased_col = self.col_names['ground_truth_rephrased']
        llm_rephrased_col = self.col_names['llm_rephrased']
        rephraser_thresh = self.thresholds['rephraser_similarity']
        rephraser_max_score = self.scoring_params['max_points']['rephraser']

        eval_df['rephraser_semantic_similarity'] = eval_df.apply(
            lambda row: self._calculate_similarity(row[llm_rephrased_col], row[gt_rephrased_col]),
            axis=1
        )
        # Simple binary score based on threshold for rephraser
        eval_df['rephraser_score'] = np.where(eval_df['rephraser_semantic_similarity'] >= rephraser_thresh, rephraser_max_score, 0.0)
        logger.info("Rephraser evaluation complete.")


        # --- Phase 2: Final Answer Evaluation ---
        logger.info("Phase 2: Evaluating Final Answer...")
        gt_final_answer_col = self.col_names['ground_truth_final_answer']
        llm_final_answer_col = self.col_names['llm_final_answer']
        source_pattern = self.regex_patterns['source_identifier']

        # 2a: Source Extraction & Scoring
        logger.info("Step 2a: Extracting and Scoring Sources...")
        eval_df['gt_sources_extracted'] = eval_df[gt_final_answer_col].apply(
            lambda x: _find_identifiers(x, source_pattern)
        )
        eval_df['llm_sources_extracted'] = eval_df[llm_final_answer_col].apply(
             lambda x: _find_identifiers(x, source_pattern)
        )
        source_scores = eval_df.apply(
            lambda row: self._source_score(row['llm_sources_extracted'], row['gt_sources_extracted']),
            axis=1,
            result_type="expand"
        )
        source_scores.columns = ['source_score', 'points_per_source', 'correct_sources_score', 'incorrect_source_penalty', 'missed_source_penalty']
        eval_df = pd.concat([eval_df, source_scores], axis=1)
        logger.info("Source scoring complete.")

        # 2b: Ingredient/Instruction Extraction
        logger.info("Step 2b: Extracting Ingredients and Instructions...")
        extracted_content = eval_df.apply(
             lambda row: self._extract_ingredients_and_instructions(row[llm_final_answer_col], row[gt_final_answer_col]),
             axis=1,
             result_type="expand"
        )
        extracted_content.columns = ['llm_ingredients_extracted', 'llm_instructions_extracted', 'gt_ingredients_extracted', 'gt_instructions_extracted']
        eval_df = pd.concat([eval_df, extracted_content], axis=1)
        logger.info("Ingredient/Instruction extraction complete.")


        # 2c: Ingredient/Instruction Scoring (Content Similarity)
        logger.info("Step 2c: Scoring Ingredient Content...")
        ingredients_results_df = self._apply_content_evaluation(eval_df[[case_id_col, 'llm_ingredients_extracted', 'gt_ingredients_extracted']], 'ingredient')

        logger.info("Step 2c: Scoring Instruction Content...")
        instructions_results_df = self._apply_content_evaluation(eval_df[[case_id_col, 'llm_instructions_extracted', 'gt_instructions_extracted']], 'instruction')

        # Merge results back - ensure unique column names handled by _apply_content_evaluation
        eval_df = eval_df.merge(ingredients_results_df, on=case_id_col, how='left')
        eval_df = eval_df.merge(instructions_results_df, on=case_id_col, how='left')
        logger.info("Content scoring complete.")

        # --- Final Score Calculation ---
        logger.info("Calculating Final Score...")
        # Sum the component scores
        eval_df['final_score_calculated'] = (
            eval_df['rephraser_score'] +
            eval_df['source_score'] +
            eval_df['sentences_score_ingredient'] +
            eval_df['sentences_score_instruction']
        ).round(4) # Summing individual max scores: 1+1+2+2 = 6

        # --- Apply "No Answer" Override ---
        logger.info("Applying 'No Answer' case overrides...")
        eval_df['final_score_override'] = np.nan # Initialize override column
        eval_df = self._no_answer_case(eval_df)

        # Create the final score, using the override if present, otherwise the calculated score
        eval_df['final_score'] = eval_df['final_score_override'].fillna(eval_df['final_score_calculated'])
        # Optionally drop the intermediate calculation/override columns
        eval_df = eval_df.drop(columns=['final_score_calculated', 'final_score_override'], errors='ignore')

        logger.info("LLM evaluation pipeline finished successfully.")
        return eval_df

# --- Main Usage ---
if __name__ == "__main__":
    # Ensure necessary imports for this block are present at the top
    import argparse
    import os
    import sys
    import pandas as pd
    import yaml
    # Assuming google_exceptions might be caught here too
    try:
        from google.api_core import exceptions as google_exceptions
    except ImportError:
        # Define a dummy exception if google cloud libraries weren't installed
        class GoogleAPICallError(Exception): pass
        google_exceptions = MagicMock() # Use MagicMock if available from testing, else None
        google_exceptions.GoogleAPICallError = GoogleAPICallError


    print("Running Auto Alignment of LLM Evaluator Script...")

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Run the LLM Evaluator pipeline on input data using a specified configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Improves help message format
    )
    parser.add_argument(
        '-c',
        '--config-file',   # More conventional name (lowercase, hyphen)
        type=str,
        required=True,     # Make explicitly required
        help="Path to the required YAML configuration file."
    )
    parser.add_argument(
        '-d',              # Changed short option to '-d'
        '--data-file',     # More conventional name
        type=str,
        required=True,     # Make explicitly required
        help="Path to the required input data file (TSV format expected)."
    )
    parser.add_argument(
        '-o',
        '--output-file',   # Added optional output path argument
        type=str,
        default=None,      # Default to None (don't save unless specified)
        help="Optional: Path to save the evaluation results DataFrame (TSV format)."
    )
    parser.add_argument('-v', '--verbose',
                        action='store_true')  # on/off flag

    args = parser.parse_args() # Parse arguments from command line

    # Use more descriptive variable names from args
    config_file_path = args.config_file
    data_file_path = args.data_file
    output_file_path = args.output_file
    verbose = args.verbose
    
    # Log the paths being used
    logger.info(f"Using Configuration File: {config_file_path}")
    logger.info(f"Using Data File: {data_file_path}")
    if output_file_path:
        logger.info(f"Will save results to: {output_file_path}")
    else:
        ## set up a default file name 
        output_file_path = 'llm_eval_results.tsv'

    # Basic file existence check early on (though handled by LLMEvaluator too)
    if not os.path.exists(config_file_path):
        logger.error(f"Configuration file not found at: {config_file_path}")
        print(f"\nError: Configuration file not found at {config_file_path}", file=sys.stderr)
        sys.exit(1) # Exit with error code

    if not os.path.exists(data_file_path):
        logger.error(f"Data file not found at: {data_file_path}")
        print(f"\nError: Data file not found at {data_file_path}", file=sys.stderr)
        sys.exit(1) # Exit with error code

    try:
        # 1. Initialize the evaluator with the config file
        logger.info("Initializing evaluator...")
        evaluator = LLMEvaluator(config_path=config_file_path)

        # 2. Load the data
        logger.info(f"Loading data from {data_file_path}...")
        try:
            # Specify dtype=str initially to prevent pandas type inference issues
            input_df = pd.read_csv(data_file_path, sep='\t', dtype=str, keep_default_na=False)
            # Replace empty strings potentially read as "" with None or np.nan if necessary downstream
            # input_df = input_df.replace({'' : None}) # Uncomment if functions expect None/NaN vs ""
            logger.info(f"Input DataFrame shape: {input_df.shape}")
        except pd.errors.EmptyDataError:
            logger.error(f"Data file '{data_file_path}' is empty.")
            print(f"\nError: Data file '{data_file_path}' is empty.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to read or parse data file '{data_file_path}': {e}", exc_info=True)
            print(f"\nError reading data file: {e}", file=sys.stderr)
            sys.exit(1)


        # 3. Run the evaluation
        logger.info("Starting evaluation pipeline...")
        results_df = evaluator.evaluate(input_df)
        logger.info("Evaluation pipeline finished.")

        # 4. Display results summary
        if verbose:
            print(f"\nOutput DataFrame shape: {results_df.shape}")
            print("\nEvaluation Results Summary (Selected Columns):")
        # Safely access case_id using .get with a default fallback name
        case_id_col_name = evaluator.col_names.get('case_id', 'case_id') # Use 'case_id' if not in config map
        display_cols = [
            case_id_col_name,
            'rephraser_score',
            'source_score',
            'sentences_score_ingredient',
            'sentences_score_instruction',
            'final_score'
        ]
        # Filter display_cols to only those actually present in results_df
        display_cols = [col for col in display_cols if col in results_df.columns]
        if not display_cols and verbose:
             print("(No key score columns found to display)")
        # Check if DataFrame is empty before trying to display
        elif results_df.empty and verbose:
            print("(Result DataFrame is empty)")
        else:
            # Display head() for brevity, or use to_string() for full output
            if verbose:
                print(results_df[display_cols].head().to_string())
                # print(results_df[display_cols].to_string()) # Uncomment for full output
            pass

        # 5. Optionally save the full results
        if output_file_path:
            logger.info(f"Saving evaluation results to {output_file_path}...")
            try:
                results_df.to_csv(output_file_path, sep='\t', index=False)
                logger.info(f"Results successfully saved.")
                print(f"\nFull evaluation results saved to {output_file_path}")
            except Exception as e:
                logger.error(f"Failed to save results to {output_file_path}: {e}", exc_info=True)
                print(f"\nError saving results: {e}", file=sys.stderr)
                # Decide if this should cause script failure
                # sys.exit(1)

    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        # Errors during config loading, validation, or non-existent files passed initial checks
        logger.error(f"Setup Error: {e}", exc_info=True)
        print(f"\nSetup Error: {e}", file=sys.stderr)
        sys.exit(1)
    except google_exceptions.GoogleAPICallError as e:
        # Errors during Vertex AI API calls (authentication, permissions, model not found, quota)
        logger.error(f"Google Cloud API Error: {e}", exc_info=True)
        print(f"\nGoogle Cloud API Error: {e}", file=sys.stderr)
        print("Please check your authentication, permissions, project ID, location, and API quotas.", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
         # Catch missing dependencies if initial check failed
         logger.error(f"Import Error: {e}. Please ensure all required libraries are installed.", exc_info=True)
         print(f"\nImport Error: {e}. Please ensure all required libraries are installed (e.g., pip install google-cloud-aiplatform pandas PyYAML sentence-transformers regex).", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
         # Catch-all for any other unexpected errors during execution
        logger.error(f"An unexpected error occurred during evaluation: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1) # Exit with error code

    logger.info("LLM Evaluator script finished successfully.")
    print("\nEvaluation script finished successfully.")
    sys.exit(0) # Explicitly exit with success code
