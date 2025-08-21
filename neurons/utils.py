import json
import random
import re
from typing import Dict, List, Any
import bittensor as bt
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types
import requests
from neurons.rule_applier import generate_variation_by_rule
from MIID.validator.reward import calculate_phonetic_similarity, calculate_orthographic_similarity
from MIID.validator.rule_evaluator import RULE_EVALUATORS
from neurons.similarity_adjust import generate_ortho_variation

def calculate_variation_scores(data: dict, variation_config: dict) -> dict:
        """
        Calculate scores for name variations and filter based on quality.
        
        Args:
            data: Dictionary of names and their variations
            
        Returns:
            Filtered dictionary containing only valid variations with good scores
        """
        from MIID.validator.reward import calculate_variation_quality, get_name_variation_rewards
        filtered_data = {}
        scores_data = {}
        final_scores = []
        rule_score = []

        # Validate input structure
        if not isinstance(data, dict) or not data:
            raise ValueError("Invalid or empty data structure")

        # Process each name and variations
        for name, variations in data.items():
            if not name or not isinstance(name, str):
                bt.logging.warning(f"Skipping invalid name: {name}")
                continue

            if variations is None or not isinstance(variations, (list, str)):
                bt.logging.warning(f"Skipping invalid variations for name: {name}")
                continue

            # Convert single string variation to list
            if isinstance(variations, str):
                variations = [variations]

            if variations:
                try:
                    # Get similarity configurations
                    phonetic_similarity = {
                        k.capitalize(): v['percentage'] 
                        for k, v in variation_config['phonetic_similarity_distribution'].items()
                    }
                    orthographic_similarity = {
                        k.capitalize(): v['percentage'] 
                        for k, v in variation_config['orthographic_similarity_distribution'].items()
                    }

                    # Calculate quality metrics
                    final_score, metrics = calculate_variation_quality(
                        name,
                        variations,
                        phonetic_similarity,
                        orthographic_similarity,
                        variation_config['variation_per_seed_name'],
                        variation_config['rule_transformation']
                    )

                    if final_score > 0.0 and metrics:
                        filtered_data[name] = variations
                        scores_data[name] = metrics
                        final_scores.append(final_score)
                        rule_score.append(metrics['rule_compliance']['score'])

                        # Log metrics
                        bt.logging.info(f"\nScoring results for '{name}':")
                        bt.logging.info(f"Final Score: {metrics.get('final_score', 0):.3f}")
                        bt.logging.info(f"Variation Count: {metrics.get('variation_count', 0)}")
                        bt.logging.info(f"Base Score: {metrics.get('base_score', 0):.3f}")
                        bt.logging.info("===============================================")

                except Exception as e:
                    bt.logging.error(f"Error calculating scores for {name}: {str(e)}")
                    continue
        bt.logging.info(f"Average final score across all names: {np.mean(final_scores) if final_scores else 0:.3f}")
        bt.logging.info(f"Average rule complience score: {np.mean(rule_score) if rule_score else 0:.3f}")
        bt.logging.info(f"Final score: {(np.mean(final_scores) * 0.6) + (np.mean(rule_score) * 0.2)}")

def clean_response(response) -> dict:
        """
        Clean and parse the raw response into a Python dictionary.
        
        Args:
            response: Raw response text
            
        Returns:
            Parsed dictionary
            
        Raises:
            ValueError: If response cannot be parsed into valid JSON
        """
        # Handle if already a dict
        if isinstance(response, dict):
            bt.logging.info("Processing input as dictionary")
            return response

        # Try parsing as JSON directly
        try:
            data = json.loads(response)
            bt.logging.info("Successfully parsed raw JSON input")
            return data
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code blocks
        code_block_markers = ["```json", "```python"]
        
        for marker in code_block_markers:
            if marker in response:
                blocks = response.split(marker)
                if len(blocks) > 1:
                    code_content = blocks[1].split("```")[0].strip()
                    
                    # If Python code block, look for JSON-like dict
                    if marker == "```python":
                        # Try to find a dictionary assignment
                        dict_match = re.search(r'(?:^|\n)\s*[\w_]+\s*=\s*(\{.*?\})', code_content, re.DOTALL)
                        if dict_match:
                            code_content = dict_match.group(1)
                        else:
                            # Look for just a dictionary
                            dict_match = re.search(r'\{.*\}', code_content, re.DOTALL)
                            if dict_match:
                                code_content = dict_match.group()
                    
                    try:
                        data = json.loads(code_content)
                        bt.logging.info(f"Successfully parsed content from {marker} block")
                        return data
                    except json.JSONDecodeError:
                        continue
                        
        # Try finding JSON-like content
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                data = json.loads(json_str)
                bt.logging.info("Successfully parsed JSON from content")
                return data
            except json.JSONDecodeError:
                raise ValueError("Found JSON-like content but it's invalid")

        raise ValueError("No valid JSON content found in response")

def send_to_dashboard(uid, config, data):
    url = 'http://127.0.0.1:5000/api/yanez/score'
    datas = {
        'uid' : uid,
        'variation_config' : config,
        'variation_result' : data
    }
    try:
        response = requests.post(url, headers = {"X-API-KEY" : os.getenv('X-API-KEY'), "Content-Type": "application/json"}, json=datas)
        response.raise_for_status()
        print(response.json())
    except Exception as e:
        print(f"Can't get data from API, Error: {e}")

def merge_response(data_openai: dict, data_rule: dict) -> Dict[str, list]:
    response = {}
    for key in set(data_openai) | set(data_rule):
        merged_list = data_openai.get(key, []) + data_rule.get(key, [])
        response[key] = list(dict.fromkeys(merged_list))
    return response

def filter_and_check(data: dict, var_config: dict) -> Dict[str, list]:
    result = {}
    rule_comp = {}
    similarity = {}
    
    # Safe access to config with defaults
    selected_rules = var_config.get('rule_transformation', {}).get('selected_rules', [])
    total_variations = var_config.get('variation_per_seed_name', 10)
    rule_percentage = var_config.get('rule_transformation', {}).get('percentage', 0)
    
    phonetic_boundaries = {"light": (0.80, 1.00), "medium": (0.60, 0.79), "far": (0.30, 0.59)}
    orthographic_boundaries = {"light": (0.70, 1.00), "medium": (0.50, 0.69), "far": (0.20, 0.49)}
    
    def get_level(score, boundaries):
        for level, (low, high) in boundaries.items():
            if low <= score <= high:
                return level
        return None
    
    # First pass: identify rule transformations
    rule_variations = set()  # Track all variations that match rules
    
    for seed, variations in data.items():
        for variation in variations:
            # Check rule transformations
            is_rule_variation = False
            for rule in selected_rules:
                if RULE_EVALUATORS[rule](seed, variation):
                    if seed not in rule_comp:
                        rule_comp[seed] = {}
                    if rule not in rule_comp[seed]:
                        rule_comp[seed][rule] = []
                    rule_comp[seed][rule].append(variation)
                    rule_variations.add(variation)
                    is_rule_variation = True
            
            # Only calculate similarity for variations NOT in rule bucket
            if not is_rule_variation:
                if seed not in similarity:
                    similarity[seed] = {"phonetic": {"light": [], "medium": [], "far": [], "other": []},
                                        "orthographic": {"light": [], "medium": [], "far": [], "other": []}}
                
                phone = calculate_phonetic_similarity(seed, variation)
                ortho = calculate_orthographic_similarity(seed, variation)
                phone_level = get_level(phone, phonetic_boundaries)
                ortho_level = get_level(ortho, orthographic_boundaries)
                
                if phone_level is not None:
                    similarity[seed]["phonetic"][phone_level].append(variation)
                else:
                    similarity[seed]["phonetic"]["other"].append(variation)
                
                if ortho_level is not None:
                    similarity[seed]["orthographic"][ortho_level].append(variation)
                else:
                    similarity[seed]["orthographic"]["other"].append(variation)
    
    # Calculate expected counts with safety checks
    expected_rule_count = round((rule_percentage / 100) * total_variations)
    expected_per_rule_count = round(expected_rule_count / len(selected_rules)) if selected_rules else 0
    
    # Similarity calculations should be based on remaining variations (not in rule bucket)
    remain_variations = total_variations - expected_rule_count
    
    # Safe access to phonetic distribution config
    phonetic_dist = var_config.get('phonetic_similarity_distribution', {})
    expected_phonetic = {
        'light': round((phonetic_dist.get('light', {}).get('percentage', 0) / 100) * remain_variations),
        'medium': round((phonetic_dist.get('medium', {}).get('percentage', 0) / 100) * remain_variations),
        'far': round((phonetic_dist.get('far', {}).get('percentage', 0) / 100) * remain_variations)
    }
    
    # Safe access to orthographic distribution config
    orthographic_dist = var_config.get('orthographic_similarity_distribution', {})
    expected_orthographic = {
        'light': round((orthographic_dist.get('light', {}).get('percentage', 0) / 100) * remain_variations),
        'medium': round((orthographic_dist.get('medium', {}).get('percentage', 0) / 100) * remain_variations),
        'far': round((orthographic_dist.get('far', {}).get('percentage', 0) / 100) * remain_variations)
    }
    
    # Log detailed statistics for each seed
    for seed in data.keys():
        bt.logging.info(f"\n=== Statistics for seed: '{seed}' ===")

        # Initialize similarity data if not exists
        if seed not in similarity:
            similarity[seed] = {"phonetic": {"light": [], "medium": [], "far": [], "other": []},
                               "orthographic": {"light": [], "medium": [], "far": [], "other": []}}

        # Orthographic similarity
        bt.logging.info("Orthographic Similarity:")
        orthographic_data = similarity[seed]['orthographic']
        for category in ['light', 'medium', 'far']:
            current_list = orthographic_data.get(category, [])
            need_count = expected_orthographic.get(category, 0)
            # Trim excess
            if len(current_list) > need_count:
                del current_list[need_count:]
            # Pad shortage
            elif len(current_list) < need_count:
                to_add = need_count - len(current_list)
                if category == 'light':
                    new_vars = [generate_light_orthographic_variation(seed) for _ in range(to_add)]
                elif category == 'medium':
                    new_vars = [generate_medium_orthographic_variation(seed) for _ in range(to_add)]
                else:
                    new_vars = [generate_far_orthographic_variation(seed) for _ in range(to_add)]
                current_list.extend(new_vars)
            bt.logging.info(f"  {category}: {len(current_list)} / {need_count}")

        # Phonetic similarity
        bt.logging.info("Phonetic Similarity:")
        phonetic_data = similarity[seed]['phonetic']
        for category in ['light', 'medium', 'far']:
            current_list = phonetic_data.get(category, [])
            need_count = expected_phonetic.get(category, 0)
            # Trim excess
            if len(current_list) > need_count:
                current_list[:] =  current_list[need_count:]
            # Pad shortage
            elif len(current_list) < need_count:
                to_add = need_count - len(current_list)
                if category == 'light':
                    new_vars = [generate_light_phonetic_variation(seed) for _ in range(to_add)]
                elif category == 'medium':
                    new_vars = [generate_medium_phonetic_variation(seed) for _ in range(to_add)]
                else:
                    new_vars = [generate_far_phonetic_variation(seed) for _ in range(to_add)]
                current_list.extend(new_vars)
            bt.logging.info(f"  {category}: {len(current_list)} / {need_count}")
        
        # Rule Transformation
        bt.logging.info('Rule Transformation:')
        for rule in selected_rules:
            if seed not in rule_comp:
                rule_comp[seed] = {}
            if rule not in rule_comp[seed]:
                rule_comp[seed][rule] = []
            
            current_list = rule_comp[seed][rule]
            if len(current_list) > expected_per_rule_count:
                del current_list[expected_per_rule_count:]
            elif len(current_list) < expected_per_rule_count:
                to_add = expected_per_rule_count - len(current_list)  # Fixed: was 'need_count'
                variations = [generate_variation_by_rule(seed, rule) for _ in range(to_add)]
                current_list.extend(variations)
            
            bt.logging.info(f"  {rule}: {len(current_list)} / {expected_per_rule_count}")

    # Build final result
    for seed in data.keys():
        all_vars = []
        
        # Rule variations first
        for rule in selected_rules:
            rule_vars = rule_comp.get(seed, {}).get(rule, [])
            for v in rule_vars:
                if v not in all_vars:
                    all_vars.append(v)
        
        # Add similarity variations if seed has them
        if seed in similarity:
            # Orthographic variations
            orthographic_data = similarity[seed]['orthographic']
            for category in ['light', 'medium', 'far', 'other']:
                for v in orthographic_data.get(category, []):
                    if v not in all_vars:
                        all_vars.append(v)
            
            # Phonetic variations
            phonetic_data = similarity[seed]['phonetic']
            for category in ['light', 'medium', 'far', 'other']:
                for v in phonetic_data.get(category, []):
                    if v not in all_vars:
                        all_vars.append(v)
        
        # Remove self/seed if present
        all_vars = [v for v in all_vars if v != seed]
        result[seed] = all_vars
    
    return result