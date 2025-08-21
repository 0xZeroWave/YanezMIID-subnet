# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 YANEZ - MIID Team

import random
import re
from typing import List, Dict, Tuple, Any, Set

def generate_replace_spaces_with_random_special_characters(original_name: str) -> str:
    """Generate a variation by replacing spaces with special characters"""
    if ' ' not in original_name:
        return original_name
    
    special_chars = '!@#$%^&*()_+-=[]{}|;:,.<>?'
    special_char = random.choice(special_chars)
    return original_name.replace(' ', special_char)

def generate_replace_double_letters_with_single_letter(original_name: str) -> str:
    """Generate a variation by replacing a double letter with single letter"""
    # Find first double letter and replace with single
    for i in range(len(original_name) - 1):
        if original_name[i] == original_name[i+1] and original_name[i].isalpha():
            return original_name[:i] + original_name[i+1:]
    return original_name

def generate_replace_random_vowel_with_random_vowel(original_name: str) -> str:
    """Generate a variation by replacing a vowel with a different vowel"""
    vowels = 'aeiou'
    vowel_positions = [i for i, char in enumerate(original_name) if char.lower() in vowels]
    
    if not vowel_positions:
        return original_name
    
    pos = random.choice(vowel_positions)
    current_vowel = original_name[pos].lower()
    new_vowels = [v for v in vowels if v != current_vowel]
    
    if not new_vowels:
        return original_name
    
    new_vowel = random.choice(new_vowels)
    variation = list(original_name)
    variation[pos] = new_vowel if variation[pos].islower() else new_vowel.upper()
    return ''.join(variation)

def generate_replace_random_consonant_with_random_consonant(original_name: str) -> str:
    """Generate a variation by replacing a consonant with a different consonant"""
    vowels = 'aeiou'
    consonants = 'bcdfghjklmnpqrstvwxyz'
    consonant_positions = [i for i, char in enumerate(original_name) 
                          if char.lower() not in vowels and char.isalpha()]
    
    if not consonant_positions:
        return original_name
    
    pos = random.choice(consonant_positions)
    current_consonant = original_name[pos].lower()
    new_consonants = [c for c in consonants if c != current_consonant]
    
    if not new_consonants:
        return original_name
    
    new_consonant = random.choice(new_consonants)
    variation = list(original_name)
    variation[pos] = new_consonant if variation[pos].islower() else new_consonant.upper()
    return ''.join(variation)

def generate_replace_random_special_character_with_random_special_character(original_name: str) -> str:
    """Generate a variation by replacing a special character with a different one"""
    special_chars = '!@#$%^&*()_+-=[]{}|;:,.<>?'
    special_positions = [i for i, char in enumerate(original_name) if char in special_chars]
    
    if not special_positions:
        return original_name
    
    pos = random.choice(special_positions)
    current_special = original_name[pos]
    new_specials = [s for s in special_chars if s != current_special]
    
    if not new_specials:
        return original_name
    
    variation = list(original_name)
    variation[pos] = random.choice(new_specials)
    return ''.join(variation)

def generate_swap_random_letter(original_name: str) -> str:
    """Generate a variation by swapping two adjacent letters"""
    if len(original_name) < 2:
        return original_name
    
    pos = random.randint(0, len(original_name) - 2)
    variation = list(original_name)
    variation[pos], variation[pos + 1] = variation[pos + 1], variation[pos]
    return ''.join(variation)

def generate_swap_adjacent_consonants(original_name: str) -> str:
    """Generate a variation by swapping adjacent consonants"""
    vowels = 'aeiou'
    
    # Find first pair of adjacent consonants
    for i in range(len(original_name) - 1):
        if (original_name[i].lower() not in vowels and original_name[i].isalpha() and
            original_name[i+1].lower() not in vowels and original_name[i+1].isalpha()):
            variation = list(original_name)
            variation[i], variation[i+1] = variation[i+1], variation[i]
            return ''.join(variation)
    
    return original_name

def generate_swap_adjacent_syllables(original_name: str) -> str:
    """Generate a variation by swapping adjacent letters (simplified syllable swap)"""
    return generate_swap_random_letter(original_name)

def generate_delete_random_letter(original_name: str) -> str:
    """Generate a variation by removing a random letter"""
    if len(original_name) <= 1:
        return original_name
    
    pos = random.randint(0, len(original_name) - 1)
    return original_name[:pos] + original_name[pos+1:]

def generate_remove_random_vowel(original_name: str) -> str:
    """Generate a variation by removing a random vowel"""
    vowels = 'aeiou'
    vowel_positions = [i for i, char in enumerate(original_name) if char.lower() in vowels]
    
    if not vowel_positions:
        return original_name
    
    pos = random.choice(vowel_positions)
    return original_name[:pos] + original_name[pos+1:]

def generate_remove_random_consonant(original_name: str) -> str:
    """Generate a variation by removing a random consonant"""
    vowels = 'aeiou'
    consonant_positions = [i for i, char in enumerate(original_name) 
                          if char.lower() not in vowels and char.isalpha()]
    
    if not consonant_positions:
        return original_name
    
    pos = random.choice(consonant_positions)
    return original_name[:pos] + original_name[pos+1:]

def generate_remove_random_special_character(original_name: str) -> str:
    """Generate a variation by removing a special character"""
    special_chars = '!@#$%^&*()_+-=[]{}|;:,.<>?'
    special_positions = [i for i, char in enumerate(original_name) if char in special_chars]
    
    if not special_positions:
        return original_name
    
    pos = random.choice(special_positions)
    return original_name[:pos] + original_name[pos+1:]

def generate_remove_title(original_name: str) -> str:
    """Generate a variation by removing a title"""
    titles = ["Mr.", "Mrs.", "Ms.", "Mr", "Mrs", "Ms", "Miss", "Dr.", "Dr",
              "Prof.", "Prof", "Sir", "Lady", "Lord", "Dame", "Master", "Mistress",
              "Rev.", "Hon.", "Capt.", "Col.", "Lt.", "Sgt.", "Maj."]
    
    for title in titles:
        if original_name.startswith(title + " "):
            return original_name[len(title)+1:]
    
    return original_name

def generate_remove_all_spaces(original_name: str) -> str:
    """Generate a variation by removing all spaces"""
    return original_name.replace(' ', '')

def generate_duplicate_random_letter_as_double_letter(original_name: str) -> str:
    """Generate a variation by duplicating a random letter"""
    if not original_name:
        return original_name
    
    pos = random.randint(0, len(original_name) - 1)
    return original_name[:pos] + original_name[pos] + original_name[pos:]

def generate_insert_random_letter(original_name: str) -> str:
    """Generate a variation by inserting a random letter"""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    pos = random.randint(0, len(original_name))
    letter = random.choice(letters)
    
    # Maintain case consistency
    if pos > 0 and original_name[pos-1].isupper():
        letter = letter.upper()
    
    return original_name[:pos] + letter + original_name[pos:]

def generate_add_random_leading_title(original_name: str) -> str:
    """Generate a variation by adding a title at the beginning"""
    titles = ["Mr.", "Mrs.", "Ms.", "Mr", "Mrs", "Ms", "Miss", "Dr.", "Dr",
              "Prof.", "Prof", "Sir", "Lady", "Lord", "Dame", "Master", "Mistress",
              "Rev.", "Hon.", "Capt.", "Col.", "Lt.", "Sgt.", "Maj."]
    title = random.choice(titles)
    return title + " " + original_name

def generate_add_random_trailing_title(original_name: str) -> str:
    """Generate a variation by adding a suffix at the end"""
    suffixes = ["Jr.", "Sr.", "III", "IV", "V", "PhD", "MD", "Esq.", "Jr", "Sr"]
    suffix = random.choice(suffixes)
    return original_name + " " + suffix

def generate_shorten_name_to_initials(original_name: str) -> str:
    """Generate a variation by converting name to initials"""
    parts = original_name.split()
    
    if len(parts) < 2:
        return original_name
    
    # Random format for initials
    formats = [
        ".".join([p[0] for p in parts]) + ".",
        ". ".join([p[0] for p in parts]) + ".",
        "".join([p[0] for p in parts])
    ]
    
    return random.choice(formats)

def generate_name_parts_permutations(original_name: str) -> str:
    """Generate a variation by permuting name parts"""
    parts = original_name.split()
    
    if len(parts) < 2:
        return original_name
    
    if len(parts) == 2:
        # Swap first and last
        return parts[1] + " " + parts[0]
    elif len(parts) == 3:
        # Choose a random permutation
        permutations = [
            parts[1] + " " + parts[0] + " " + parts[2],
            parts[2] + " " + parts[0] + " " + parts[1],
            parts[1] + " " + parts[2] + " " + parts[0]
        ]
        return random.choice(permutations)
    else:
        # For more parts, just swap first two
        shuffled = parts[:]
        shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
        return " ".join(shuffled)

def generate_initial_only_first_name(original_name: str) -> str:
    """Generate a variation by reducing first name to initial"""
    parts = original_name.split()
    
    if len(parts) < 2:
        return original_name
    
    # Choose format randomly
    formats = [
        parts[0][0] + ". " + " ".join(parts[1:]),
        parts[0][0] + "." + " ".join(parts[1:])
    ]
    
    return random.choice(formats)

def generate_shorten_name_to_abbreviations(original_name: str) -> str:
    """Generate a variation by abbreviating name parts"""
    parts = original_name.split()
    abbreviated_parts = []
    
    for part in parts:
        if len(part) > 2:
            # Abbreviate to first 2-3 characters
            abbrev_length = random.randint(2, min(3, len(part)))
            abbreviated_parts.append(part[:abbrev_length])
        else:
            abbreviated_parts.append(part)
    
    return " ".join(abbreviated_parts)

# Map rule names to their generation functions
RULE_GENERATORS = {
    "replace_spaces_with_random_special_characters": generate_replace_spaces_with_random_special_characters,
    "replace_double_letters_with_single_letter": generate_replace_double_letters_with_single_letter,
    "replace_random_vowel_with_random_vowel": generate_replace_random_vowel_with_random_vowel,
    "replace_random_consonant_with_random_consonant": generate_replace_random_consonant_with_random_consonant,
    "replace_random_special_character_with_random_special_character": generate_replace_random_special_character_with_random_special_character,
    "swap_random_letter": generate_swap_random_letter,
    "swap_adjacent_consonants": generate_swap_adjacent_consonants,
    "swap_adjacent_syllables": generate_swap_adjacent_syllables,
    "delete_random_letter": generate_delete_random_letter,
    "remove_random_vowel": generate_remove_random_vowel,
    "remove_random_consonant": generate_remove_random_consonant,
    "remove_random_special_character": generate_remove_random_special_character,
    "remove_title": generate_remove_title,
    "remove_all_spaces": generate_remove_all_spaces,
    "duplicate_random_letter_as_double_letter": generate_duplicate_random_letter_as_double_letter,
    "insert_random_letter": generate_insert_random_letter,
    "add_random_leading_title": generate_add_random_leading_title,
    "add_random_trailing_title": generate_add_random_trailing_title,
    "shorten_name_to_initials": generate_shorten_name_to_initials,
    "name_parts_permutations": generate_name_parts_permutations,
    "initial_only_first_name": generate_initial_only_first_name,
    "shorten_name_to_abbreviations": generate_shorten_name_to_abbreviations
}

def generate_variation_by_rule(original_name: str, rule: str) -> str:
    """
    Generate a single variation using a specific rule
    
    Args:
        original_name: The original name to create variation from
        rule: The rule name to apply
        
    Returns:
        A single variation string
    """
    if rule in RULE_GENERATORS:
        try:
            return RULE_GENERATORS[rule](original_name)
        except Exception as e:
            print(f"Error generating variation for rule {rule}: {str(e)}")
            return original_name
    else:
        print(f"Unknown rule: {rule}")
        return original_name

def generate_variations_for_all_rules(original_name: str) -> Dict[str, str]:
    """
    Generate one variation for each available rule
    
    Args:
        original_name: The original name to create variations from
        
    Returns:
        Dictionary mapping rule names to single generated variations
    """
    variations = {}
    
    for rule in RULE_GENERATORS.keys():
        variations[rule] = generate_variation_by_rule(original_name, rule)
    
    return variations

# Example usage and testing function
def test_generators(original_name: str = "John A. Smith"):
    """Test all generators with a sample name"""
    print(f"Testing generators with: '{original_name}'")
    print("=" * 50)
    
    variations = generate_variations_for_all_rules(original_name)
    
    for rule, variation in variations.items():
        print(f"{rule}: '{variation}'")
    
    return variations

if __name__ == "__main__":
    # Test the generators
    print("Test 1:")
    test_results1 = test_generators("John Smith")
    
    print("\n" + "=" * 70)
    print("Test 2:")
    test_results2 = test_generators("Jane Doe")
    
    print("\n" + "=" * 70)
    print("Test 3 - Single rule generation:")
    single_variation = generate_variation_by_rule("Michael Johnson", "shorten_name_to_abbreviations")
    print(f"Original: 'Michael Johnson' -> Variation: '{single_variation}'")