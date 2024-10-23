import json
import re
from dataclasses import dataclass
import numpy as np  # Assuming embeddings are stored as numpy arrays or similar

format_string = """{"input":"<input_example>","output":"<output_example>"}"""


@dataclass
class DataPoint:
    input: str
    output: str
    embedding: np.ndarray = None  # or list, depending on how embeddings are represented


def _parse_generated_pairs(response_text):
    """
    Parses the generated text and returns a list of DataPoint objects.
    Handles cases where lines have numbering or irregular formatting.
    """
    lines = _remove_instructions(response_text).splitlines()
    data_points = []

    for line in lines:
        # Remove any leading numbering like "1.", "2.", etc.
        line = re.sub(r"^\d+\.\s*", "", line).strip()

        # Try to parse the line as JSON
        try:
            pair = json.loads(line)
            input_text = pair.get("input", "").strip()
            output_text = pair.get("output", "").strip()

            # Add the parsed input-output pair to the list as a DataPoint object
            if input_text and output_text:
                data_points.append(DataPoint(input=input_text, output=output_text))
        except json.JSONDecodeError:
            # Log a warning and continue if the line is not valid JSON
            print(f"Warning: Skipping invalid line - {line}")
            continue

    return data_points


def _remove_instructions(response_text):
    """
    Remove everything between [INST] and [/INST] tags.
    """
    # Remove anything between [INST] and [/INST]
    cleaned_text = re.sub(r"\[INST\](.*?)\[/INST\]", "", response_text, flags=re.DOTALL).strip()
    return cleaned_text


def _datapoint_to_string(datapoint: DataPoint) -> str:
    """
    Converts a DataPoint object to a string in the format {"input": ..., "output": ...}.
    """
    return f'{{"input":"{datapoint.input}","output":"{datapoint.output}"}}'
