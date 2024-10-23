import json
import re


def _parse_generated_pairs(response_text):
    lines = _remove_instructions(response_text).splitlines()
    new_pairs = []

    for line in lines:
        # Try to parse each line as a JSON object
        try:
            pair = json.loads(line)
            input_text = pair.get("input", "").strip()
            output_text = pair.get("output", "").strip()

            # Add the parsed input-output pair to the list
            if input_text and output_text:
                new_pairs.append((input_text, output_text))
        except json.JSONDecodeError:
            # Skip lines that don't match the JSON format
            continue

    return new_pairs


def _remove_instructions(response_text):
    """
    Remove everything between [INST] and [/INST] tags.
    """
    # Remove anything between [INST] and [/INST]
    cleaned_text = re.sub(r"\[INST\](.*?)\[/INST\] ", "", response_text, flags=re.DOTALL)
    return cleaned_text
