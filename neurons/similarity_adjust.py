import random
import jellyfish
from typing import List, Dict, Tuple
import re
import Levenshtein
from MIID.validator.reward import calculate_orthographic_similarity, calculate_phonetic_similarity

def generate_light_phonetic_variation(name: str) -> str:
    """Generate light phonetic variation (0.80-1.00 similarity)."""
    name_lower = name.lower()
    
    # Light transformations
    transformations = [
        lambda x: x + 'e' if not x.endswith('e') else x + 'h',
        lambda x: x + 'h' if not x.endswith('h') else x + 'e',
        lambda x: x.replace('c', 'k') if 'c' in x else x + 'e',
        lambda x: x.replace('k', 'c') if 'k' in x else x + 'e',
        lambda x: x.replace('s', 'c') if 's' in x else x + 'e',
        lambda x: x.replace('z', 's') if 'z' in x else x + 'e',
        lambda x: x.replace('f', 'ph') if 'f' in x else x + 'e',
        lambda x: x.replace('ph', 'f') if 'ph' in x else x + 'e',
        lambda x: x + 'e',
        lambda x: x + 'h',
        lambda x: x + 'y',
    ]
    
    transform = random.choice(transformations)
    variation = transform(name_lower)
    
    # Ensure we always get a different variation
    if variation == name_lower:
        variation = name_lower + random.choice(['e', 'h', 'y'])
    
    return variation.capitalize()

def generate_medium_phonetic_variation(name: str) -> str:
    """Generate medium phonetic variation (0.60-0.79 similarity)."""
    name_lower = name.lower()
    
    # Medium transformations
    transformations = [
        lambda x: x.replace('a', 'e') if 'a' in x else x + 'ie',
        lambda x: x.replace('e', 'a') if 'e' in x else x + 'ie',
        lambda x: x.replace('i', 'y') if 'i' in x else x + 'ie',
        lambda x: x.replace('o', 'u') if 'o' in x else x + 'ie',
        lambda x: x.replace('u', 'o') if 'u' in x else x + 'ie',
        lambda x: x.replace('t', 'tt') if 't' in x else x + 'ie',
        lambda x: x.replace('d', 'dd') if 'd' in x else x + 'ie',
        lambda x: x.replace('l', 'll') if 'l' in x else x + 'ie',
        lambda x: x + 'ie',
        lambda x: x + 'ey',
        lambda x: x + 'ay',
    ]
    
    transform = random.choice(transformations)
    variation = transform(name_lower)
    
    # Ensure we always get a different variation
    if variation == name_lower:
        variation = name_lower + random.choice(['ie', 'ey', 'ay'])
    
    return variation.capitalize()

def generate_far_phonetic_variation(name: str) -> str:
    """Generate far phonetic variation (0.30-0.59 similarity)."""
    name_lower = name.lower()
    
    # Far transformations
    transformations = [
        lambda x: x.replace('a', 'o') if 'a' in x else x + 'x',
        lambda x: x.replace('e', 'u') if 'e' in x else x + 'x',
        lambda x: x.replace('i', 'a') if 'i' in x else x + 'x',
        lambda x: x.replace('o', 'a') if 'o' in x else x + 'x',
        lambda x: x.replace('u', 'e') if 'u' in x else x + 'x',
        lambda x: x.replace('c', 'x') if 'c' in x else x + 'x',
        lambda x: x.replace('k', 'q') if 'k' in x else x + 'x',
        lambda x: x.replace('s', 'z') if 's' in x else x + 'x',
        lambda x: x + 'x',
        lambda x: x + 'q',
        lambda x: x + 'z',
    ]
    
    transform = random.choice(transformations)
    variation = transform(name_lower)
    
    # Ensure we always get a different variation
    if variation == name_lower:
        variation = name_lower + random.choice(['x', 'q', 'z'])
    
    return variation.capitalize()

def generate_light_orthographic_variation(name: str) -> str:
    """Generate light orthographic variation (0.70-1.00 similarity)."""
    name_lower = name.lower()
    
    # Light orthographic changes - very subtle changes to maintain high similarity
    transformations = [
        lambda x: x + 'e' if not x.endswith('e') else x + 'h',
        lambda x: x + 'h' if not x.endswith('h') else x + 'e',
        lambda x: x.replace('c', 'k') if 'c' in x else x + 'e',
        lambda x: x.replace('k', 'c') if 'k' in x else x + 'e',
        lambda x: x.replace('s', 'c') if 's' in x else x + 'e',
        lambda x: x.replace('z', 's') if 'z' in x else x + 'e',
        lambda x: x.replace('f', 'ph') if 'f' in x else x + 'e',
        lambda x: x.replace('ph', 'f') if 'ph' in x else x + 'e',
        lambda x: x + 'e',
        lambda x: x + 'h',
        lambda x: x + 'y',
        lambda x: x + 'a',
        lambda x: x[:-1] + 'e' if len(x) > 1 and x[-1] != 'e' else x + 'e',
        lambda x: x[:-1] + 'h' if len(x) > 1 and x[-1] != 'h' else x + 'h',
    ]
    
    # Try multiple transformations to ensure we get a good variation
    for _ in range(10):
        transform = random.choice(transformations)
        variation = transform(name_lower)
        
        # Ensure we always get a different variation
        if variation == name_lower:
            variation = name_lower + random.choice(['e', 'h', 'y', 'a'])
        
        # Check if this variation would be in the light range
        orthographic_score = calculate_orthographic_similarity(name, variation.capitalize())
        if 0.70 <= orthographic_score <= 1.00:
            return variation.capitalize()
    
    # Fallback: just add a suffix
    return (name_lower + random.choice(['e', 'h', 'y'])).capitalize()

def generate_medium_orthographic_variation(name: str) -> str:
    """Generate medium orthographic variation (0.50-0.69 similarity)."""
    name_lower = name.lower()
    
    # Medium orthographic changes - more aggressive to ensure medium similarity
    transformations = [
        lambda x: x.replace('a', 'e') if 'a' in x else x + 'ie',
        lambda x: x.replace('e', 'a') if 'e' in x else x + 'ie',
        lambda x: x.replace('i', 'y') if 'i' in x else x + 'ie',
        lambda x: x.replace('o', 'u') if 'o' in x else x + 'ie',
        lambda x: x.replace('u', 'o') if 'u' in x else x + 'ie',
        lambda x: x.replace('t', 'tt') if 't' in x else x + 'ie',
        lambda x: x.replace('d', 'dd') if 'd' in x else x + 'ie',
        lambda x: x.replace('l', 'll') if 'l' in x else x + 'ie',
        lambda x: x.replace('n', 'nn') if 'n' in x else x + 'ie',
        lambda x: x.replace('r', 'rr') if 'r' in x else x + 'ie',
        lambda x: x.replace('s', 'ss') if 's' in x else x + 'ie',
        lambda x: x + 'ie',
        lambda x: x + 'ey',
        lambda x: x + 'ay',
        lambda x: x + 'e' + 'y',
        lambda x: x + 'i' + 'e',
        lambda x: x[:-1] + 'ie' if len(x) > 1 else x + 'ie',
        lambda x: x[:-1] + 'ey' if len(x) > 1 else x + 'ey',
    ]
    
    # Try multiple transformations to ensure we get a good variation
    for _ in range(10):
        transform = random.choice(transformations)
        variation = transform(name_lower)
        
        # Ensure we always get a different variation
        if variation == name_lower:
            variation = name_lower + random.choice(['ie', 'ey', 'ay', 'e'])
        
        # Check if this variation would be in the medium range
        orthographic_score = calculate_orthographic_similarity(name, variation.capitalize())
        if 0.50 <= orthographic_score < 0.70:
            return variation.capitalize()
    
    # Fallback: just add a suffix
    return (name_lower + random.choice(['ie', 'ey', 'ay'])).capitalize()

def generate_far_orthographic_variation(name: str) -> str:
    """Generate far orthographic variation (0.20-0.49 similarity)."""
    name_lower = name.lower()
    
    # Far orthographic changes - more aggressive changes
    transformations = [
        lambda x: x.replace('a', 'o') if 'a' in x else x + 'x',
        lambda x: x.replace('e', 'u') if 'e' in x else x + 'x',
        lambda x: x.replace('i', 'a') if 'i' in x else x + 'x',
        lambda x: x.replace('o', 'a') if 'o' in x else x + 'x',
        lambda x: x.replace('u', 'e') if 'u' in x else x + 'x',
        lambda x: x.replace('c', 'x') if 'c' in x else x + 'x',
        lambda x: x.replace('k', 'q') if 'k' in x else x + 'x',
        lambda x: x.replace('s', 'z') if 's' in x else x + 'x',
        lambda x: x + 'x',
        lambda x: x + 'q',
        lambda x: x + 'z',
        lambda x: x.replace('a', 'o').replace('e', 'u') if 'a' in x or 'e' in x else x + 'x',
        lambda x: x.replace('i', 'a').replace('o', 'u') if 'i' in x or 'o' in x else x + 'x',
        lambda x: x[:-1] + 'x' if len(x) > 1 else x + 'x',
        lambda x: x[:-1] + 'q' if len(x) > 1 else x + 'q',
    ]
    
    # Try multiple transformations to ensure we get a good variation
    for _ in range(10):
        transform = random.choice(transformations)
        variation = transform(name_lower)
        
        # Ensure we always get a different variation
        if variation == name_lower:
            variation = name_lower + random.choice(['x', 'q', 'z'])
        
        # Check if this variation would be in the far range
        orthographic_score = calculate_orthographic_similarity(name, variation.capitalize())
        if 0.20 <= orthographic_score < 0.50:
            return variation.capitalize()
    
    # Fallback: just add a suffix
    return (name_lower + random.choice(['x', 'q', 'z'])).capitalize()
def generate_ortho_variation(name_original : str,variation_total: int, names: list, ortho_dist: dict):
    variations = []
    boundaries = {
        'light': (0.70, 1.00),
        'medium': (0.50, 0.69),
        'far': (0.20, 0.49)
    }

    for name in names:
        for level, cfg in ortho_dist.items():
            level = level.lower()
            percentage = cfg.get('percentage', 0)
            cnt = round((percentage / 100) * variation_total)

            if cnt <= 0:
                continue  # skip if no variation to generate

            min_sim, max_sim = boundaries.get(level, (0.0, 1.0))
            cnt_to_up = 1 if level == 'light' else 2

            generated = 0
            attempt = 0
            max_attempt = cnt * 20

            while generated < cnt and attempt < max_attempt:
                name_var = to_uppercase(name, cnt_to_up)
                score = calculate_orthographic_similarity(name_original, name_var)
                score_sim = calculate_phonetic_similarity(name_original, name)
                print(f" {attempt}.{name_original} : variasi {name_var}, score ortho {score}, score phon {score_sim}")
                if (
                    name_var != name_original and
                    name_var != name and
                    min_sim <= score <= max_sim and
                    name_var not in variations
                ):
                    variations.append(name_var)
                    generated += 1
                else:
                    if score > max_sim:
                        cnt_to_up += 1
                    elif score < min_sim and cnt_to_up > 1:
                        cnt_to_up -= 1
                attempt += 1

    return variations

def to_uppercase(name: str, count: int):
    if not name or count <= 0:
        return name
    
    # Find alphabetic characters and their positions
    alpha_positions = [(i, c) for i, c in enumerate(name) if c.isalpha()]
    
    if not alpha_positions:
        return name
    
    # Sample positions to uppercase
    positions_to_change = random.sample(
        alpha_positions, 
        min(count, len(alpha_positions))
    )
    
    # Build result string
    chars = list(name)
    for pos, _ in positions_to_change:
        chars[pos] = chars[pos].upper()
    
    return ''.join(chars)