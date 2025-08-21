GENERATE_SYSTEM_VARIATION = """
You are a Synthetic Data Generation Specialist for Security System Stress-Testing.  
Your job is to generate unique synthetic name variations for adversarial identity testing.  
All outputs must be synthetic, non-identifying, and used strictly for defensive security research.  

EVALUATION FUNCTIONS (use them conceptually when generating):
--------------------------------------------------------------
def calculate_phonetic_similarity(original_name: str, variation: str) -> float:
    algorithms = {
        "soundex": lambda x, y: jellyfish.soundex(x) == jellyfish.soundex(y),
        "metaphone": lambda x, y: jellyfish.metaphone(x) == jellyfish.metaphone(y),
        "nysiis": lambda x, y: jellyfish.nysiis(x) == jellyfish.nysiis(y),
    }
    # deterministically sample and weight
    score = weighted combination of equality checks
    return float(score)

def calculate_orthographic_similarity(original_name: str, variation: str) -> float:
    distance = Levenshtein.distance(original_name, variation)
    max_len = max(len(original_name), len(variation))
    return 1.0 - (distance / max_len)

BOUNDARIES:
Phonetic → Light: 0.80–1.00, Medium: 0.60–0.79, Far: 0.30–0.59  
Orthographic → Light: 0.70–1.00, Medium: 0.50–0.69, Far: 0.20–0.49  

GENERATION RULES:
1. Generate phonetic variations first. Keep only those whose similarity score falls into the specified boundary range.  
2. After phonetic is completed, generate orthographic variations:  
   - Light: uppercase 1–2 letters  
   - Medium: uppercase 3–4 letters  
   - Far: uppercase 5-6 letters  
   Evaluate with orthographic similarity and keep only results inside boundary.  
3. Return output as **comma-separated values only**.  
   - Example: variation1, variation2, variation3  
   - No extra text, no explanation, no labels.
"""
SYSTEM_INSTRUCTION_EXTRACT_INFO = """
You will be given an instruction for generating adversarial identity name variations.

TASK:
Return a single raw JSON object in the following format, with no explanations or comments.  
**Output must match the style and structure of the example below.**  

EXAMPLE OUTPUT FORMAT:
{{
  "seed_name": "allison, brian, john, donna howell, michelle schneider, jennifer jackson, thomas alexander, abigail garcia, christine taylor, stephanie andrews, joshua, jeffrey, brittany morrison, linda, isaiah whitaker",
  "variation_per_seed_name": 5,
  "phonetic_similarity_distribution": {{
    "light": {{"percentage": 70}},
    "medium": {{"percentage": 30}},
    "far": {{"percentage": 0}}
  }},
  "orthographic_similarity_distribution": {{
    "light": {{"percentage": 0}},
    "medium": {{"percentage": 70}},
    "far": {{"percentage": 30}}
  }},
  "rule_transformation": {{
    "percentage" : 30
    "selected_rules": ["swap_adjacent_consonants", "swap_adjacent_vowels", "replace_spaces_with_random_special_characters"]
  }}
}}

RULE_DESCRIPTIONS = {{
    "replace_spaces_with_random_special_characters": "Replace spaces with special characters",
    "replace_double_letters_with_single_letter": "Replace double letters with a single letter",
    "replace_random_vowel_with_random_vowel": "Replace random vowels with different vowels",
    "replace_random_consonant_with_random_consonant": "Replace random consonants with different consonants",
    "replace_random_special_character_with_random_special_character": "Replace special characters with different ones",
    "swap_adjacent_consonants": "Swap adjacent consonants",
    "swap_adjacent_syllables": "Swap adjacent syllables",
    "swap_random_letter": "Swap random adjacent letters",
    "delete_random_letter": "Delete a random letter",
    "remove_random_vowel": "Remove a random vowel",
    "remove_random_consonant": "Remove a random consonant",
    "remove_random_special_character": "Remove a random special character",
    "remove_title": "Remove title (Mr., Dr., etc.)",
    "remove_all_spaces": "Remove all spaces",
    "duplicate_random_letter_as_double_letter": "Duplicate a random letter",
    "insert_random_letter": "Insert a random letter",
    "add_random_leading_title": "Add a title prefix (Mr., Dr., etc.)",
    "add_random_trailing_title": "Add a title suffix (Jr., PhD, etc.)",
    "initial_only_first_name": "Use first name initial with last name",
    "shorten_name_to_initials": "Convert name to initials",
    "shorten_name_to_abbreviations": "Abbreviate name parts",
    "name_parts_permutations": "Reorder name parts"
}}
NOTE: 
 - for selected_rules, use the keys from RULE_DESCRIPTIONS
 - Default percentages for phonetic and orthographic similarity are 100% Medium
 - Default number of variations per seed name is 15
 - Default rule transformations are 30%

YOUR RESPONSE:
- Only output the single raw JSON object, nothing else.
- Use the same structure, field order, and style as the example above.
"""
def get_variation_analysis(prompt) -> Dict:
        """
        Request LLM to analyze names and generate variations with similarity metrics.
        
        Args:
            names: List of names to analyze
            
        Returns:
            Dictionary containing variations and their similarity metrics
        """
        try:
            # Create Ollama client with configured URL
            client = genai.Client(
               api_key=os.getenv('GEMINI_API_KEY')
            )
            response = client.models.generate_content(
               model="gemini-2.5-flash",
               contents=prompt,
               config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION_EXTRACT_INFO,
                thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
                top_k=20,        # More restricted vocabulary for consistency
                top_p=0.8,       # More focused sampling
                ),
            )
            
            # Extract and return the content of the response
            return response.text.strip()
        except Exception as e:
            bt.logging.error(f"LLM query failed: {str(e)}")
            raise

def generate_similarity_variation(name: str, count: int, ortho_dist: dict, phon_dist: dict):
    LEVELS = ["light", "medium", "far"]

    def _extract_percentage(value) -> float:
        if isinstance(value, dict) and "percentage" in value:
            return float(value["percentage"])
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    def _normalize_to_proportions(percentages):
        s = sum(percentages)
        if s <= 0:
            n = len(percentages)
            return [1.0 / n] * n
        return [p / s for p in percentages]

    def _largest_remainder_allocation(total, weights):
        raw = [w * total for w in weights]
        floors = [int(x) for x in raw]
        remainder = total - sum(floors)
        remainders = [(i, raw[i] - floors[i]) for i in range(len(raw))]
        remainders.sort(key=lambda x: x[1], reverse=True)
        for i in range(remainder):
            floors[remainders[i][0]] += 1
        return floors

    # --- Build combined distribution (phonetic × orthographic) ---
    phon_perc = [_extract_percentage(phon_dist.get(l, {})) for l in LEVELS]
    ortho_perc = [_extract_percentage(ortho_dist.get(l, {})) for l in LEVELS]

    phon_prop = _normalize_to_proportions(phon_perc)
    ortho_prop = _normalize_to_proportions(ortho_perc)

    combo_keys = []
    combo_weights = []
    for i, p in enumerate(LEVELS):
        for j, o in enumerate(LEVELS):
            combo_keys.append((p, o))
            combo_weights.append(phon_prop[i] * ortho_prop[j])

    combo_alloc = _largest_remainder_allocation(count, combo_weights)

    combined = {}
    for k, v in zip(combo_keys, combo_alloc):
        if v > 0:
            combined[k] = v

    # --- Build readable text for prompt & subtotals (guardrails) ---
    parts = []
    for p in LEVELS:
        for o in LEVELS:
            n = combined.get((p, o), 0)
            if n > 0:
                parts.append(f"- {n} variations: othographic {o}, phonetic {p}\n")
    combined_text = "\n".join(parts) if parts else "- 0 variations"

    prompt = f"""
Generate exactly {count} variations for {name}.

Distribute the {count} variations using buckets as follows:
{combined_text}

NOTES:
- Keep multi-part name as multi-part; vary within parts but do not reduce the number of parts.
- Do NOT include the seed name itself.
- Each variation must be unique.
- Return as comma-separated values only. No metadata, numbering, or extra text.
    """
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=GENERATE_SYSTEM_VARIATION,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                top_k=15,  # More restrictive for better rule following
                top_p=0.7,  # More focused
                temperature=0.8
            ),
        )
        return response.text
    except Exception as e:
        bt.logging.error(f"LLM query failed: {str(e)}")
        return {}  # Return empty dict instead of raising

def generate_variation_for_rule(name: str, rule: str, rule_count: int):
    # Create detailed rule descriptions to ensure proper following
    RULE_DESCRIPTIONS = {
        "replace_spaces_with_random_special_characters": "Replace all spaces with random special characters like -, _, ., etc. (e.g., 'John Smith' -> 'John-Smith' or 'John_Smith')",
        "replace_double_letters_with_single_letter": "Replace double letters in the name with a single letter (e.g., 'Lloyd' -> 'Loyd', 'Hannah' -> 'Hana')",
        "replace_random_vowel_with_random_vowel": "Replace a random vowel in the name with a different vowel (e.g., 'John' -> 'Jeen', 'Maria' -> 'Moria')",
        "replace_random_consonant_with_random_consonant": "Replace a random consonant in the name with a different consonant (e.g., 'Brian' -> 'Bzian', 'Diana' -> 'Diana' or 'Diala')",
        "swap_adjacent_consonants": "Swap two adjacent consonants in the name (e.g., 'John' -> 'Jhon', 'Brian' -> 'Biarn')",
        "swap_adjacent_syllables": "Swap two adjacent syllables in the name (e.g., 'Diana' -> 'Nadia', 'Maria' -> 'Riam')",
        "swap_random_letter": "Swap two random adjacent letters in the name (e.g., 'Maria' -> 'Maira', 'Brian' -> 'Brina')",
        "delete_random_letter": "Delete a random letter from the name (e.g., 'Brian' -> 'Bran', 'Maria' -> 'Mria')",
        "remove_random_vowel": "Remove a random vowel from the name (e.g., 'Diana' -> 'Dina', 'Maria' -> 'Mria')",
        "remove_random_consonant": "Remove a random consonant from the name (e.g., 'Brian' -> 'Bian', 'Maria' -> 'Mara')",
        "remove_all_spaces": "Remove all spaces from the name (e.g., 'John Smith' -> 'JohnSmith', 'Mary Jane' -> 'MaryJane')",
        "duplicate_random_letter_as_double_letter": "Duplicate a random letter in the name (e.g., 'Brian' -> 'Briaan', 'Maria' -> 'Marria')",
        "insert_random_letter": "Insert a random letter into the name (e.g., 'John' -> 'Johna', 'Maria' -> 'Marvia')",
        "add_random_leading_title": "Add a professional title before the name (e.g., 'John' -> 'Dr. John', 'Mary' -> 'Ms. Mary', 'Robert' -> 'Mr. Robert')",
        "add_random_trailing_title": "Add a professional title after the name (e.g., 'John' -> 'John Jr.', 'Mary' -> 'Mary PhD', 'Robert' -> 'Robert Sr.')",
        "initial_only_first_name": "Replace first name with initial only (e.g., 'John Smith' -> 'J. Smith', 'Mary Johnson' -> 'M. Johnson')",
        "shorten_name_to_initials": "Convert entire name to initials (e.g., 'John Smith' -> 'J.S.', 'Mary Jane Doe' -> 'M.J.D.')",
        "shorten_name_to_abbreviations": "Abbreviate each part of the name (e.g., 'John Smith' -> 'Jn Sth', 'Mary Jane Doe' -> 'My Jn D')",
        "name_parts_permutations": "Rearrange the order of name parts (e.g., 'John Smith' -> 'Smith John', 'Mary Jane Doe' -> 'Jane Mary Doe')"
    }

    rule_description = RULE_DESCRIPTIONS.get(rule, f"Apply the transformation rule: {rule}")
    
    # Create more specific prompt with examples
    prompt = f"""
    CRITICAL: You must generate exactly {rule_count} variations for EACH name using the specific transformation rule.

    RULE: {rule}
    DESCRIPTION: {rule_description}
    
    NAMES TO TRANSFORM: {name}
    
    REQUIREMENTS:
    1. Each variation MUST clearly show the {rule} transformation applied
    2. Generate exactly {rule_count} variations per name
    3. Each variation must be different from the original name
    4. Each variation must be recognizable as following the {rule} rule
    5. each variation must be unique
    
    Return ONLY as comma-seperated only format. exactly {rule_count} rule-transformed variations.
    """
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=GENERATE_SYSTEM_VARIATION,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                top_k=15,  # More restrictive for better rule following
                top_p=0.7,  # More focused
                temperature=0.9  # Lower temperature for more consistent rule following
            ),
        )
        # cleaned_response = clean_response(response.text
            
        # Validate that the response follows the rule (basic validation)
        return response.text
                    
    except Exception as e:
        bt.logging.error(f"LLM query failed: {str(e)}")
        return {}  # Return empty dict instead of response.text which might not exist

def generate_variation(variation_config: dict, name: str):
    # jumlah total variasi yang diminta
    variations_per_name = variation_config['variation_per_seed_name']
    # phonetic distribusi
    phonetic_dist = variation_config['phonetic_similarity_distribution']
    # orthographic distribusi
    orthographic_dist = variation_config['orthographic_similarity_distribution']
    # yang tersedia
    rules_selected = variation_config['rule_transformation']['selected_rules']
    # rule yang hanya untuk nama dengan 2 suku kata
    rule_for_multi_part_name = ['name_parts_permutations', 'initial_only_first_name', 'shorten_name_to_initials', 'replace_spaces_with_random_special_characters', 'remove_all_spaces']
    # rule percentase
    percentage = variation_config['rule_transformation']['percentage']
    # jumlah rule yang harus diapply
    expected_rule_count = round(variations_per_name * (percentage / 100))
    if isinstance(rules_selected, dict):
        rules_list = list(rules_selected.keys())
    else:
        rules_list = rules_selected
    # jumlah variasi yang sisa yang sudah di kurangi dengan dengan rule transformasi
    remain_variation_per_name = variations_per_name - expected_rule_count
    # rule yang hanya bisa di apply pada nama dengan single part-only
    rule_for_single_part_name = [rule for rule in rules_list if rule not in rule_for_multi_part_name]
    # hitung distribusi rulenya
    expected_rule_per_seed = expected_rule_count // len(rules_list)
    remaining_rule_per_seed = expected_rule_count % len(rules_list)
    # distribusi rule
    dist = {r: expected_rule_per_seed for r in rules_selected}
    for i in range(remaining_rule_per_seed):
        dist[rules_selected[i]] += 1
    # variable untuk menyimpan hasil
    name_variation = []
    # check namanya single atau multi
    if len(name.split()) < 2:
        if rule_for_single_part_name:
            expected_single_rule = expected_rule_count // len(rule_for_single_part_name)
            remain_single_rule = expected_rule_count % len(rule_for_single_part_name)
            dist_single = {r: expected_single_rule for r in rule_for_single_part_name}
            for i in range(remain_single_rule):
                dist_single[rule_for_single_part_name[i]] += 1
            for rule, cnt in dist_single.items():
                if rule in rule_for_single_part_name:
                    for _ in range(cnt):
                        result_var_rule = generate_variation_by_rule(name, rule)
                        name_variation.append(result_var_rule)
                else:
                    remain_variation_per_name += cnt
        else:
            remain_variation_per_name += expected_rule_count
        
        # Generate similarity variations for single part names
        result_var_sim = generate_similarity_variation(name, remain_variation_per_name, orthographic_dist, phonetic_dist)
        if isinstance(result_var_sim, str):
            result_var_sim = [s.strip() for s in result_var_sim.split(",") if s.strip()]
        # result_var_sim = generate_ortho_variation(name,remain_variation_per_name, result_var_sim, orthographic_dist)
        name_variation.extend(result_var_sim)
        return ', '.join(name_variation)
    else:
        # Handle multi-part names
        for rule, cnt in dist.items():
            for _ in range(cnt):
                result_rule = generate_variation_by_rule(name, rule)
                name_variation.append(result_rule)
        
        # Generate similarity variations for multi-part names
        result_sim = generate_similarity_variation(name, remain_variation_per_name, orthographic_dist, phonetic_dist)
        if isinstance(result_sim, str):
            result_sim = [s.strip() for s in result_sim.split(",") if s.strip()]
        # result_sim = generate_ortho_variation(name,remain_variation_per_name, result_sim, orthographic_dist)
        name_variation.extend(result_sim)
        return ', '.join(name_variation)