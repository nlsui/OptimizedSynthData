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
    lines = _remove_instructions(response_text).splitlines()  # Assuming _remove_instructions is defined elsewhere
    data_points = []

    for line in lines:
        # Try to parse each line as a JSON object
        try:
            pair = json.loads(line)
            input_text = pair.get("input", "").strip()
            output_text = pair.get("output", "").strip()

            # Add the parsed input-output pair to the list as a DataPoint object
            if input_text and output_text:
                data_points.append(DataPoint(input=input_text, output=output_text))
        except json.JSONDecodeError:
            # Skip lines that don't match the JSON format
            continue

    return data_points


def _remove_instructions(response_text):
    """
    Remove everything between [INST] and [/INST] tags.
    """
    # Remove anything between [INST] and [/INST]
    cleaned_text = re.sub(r"\[INST\](.*?)\[/INST\] ", "", response_text, flags=re.DOTALL)
    return cleaned_text


def _datapoint_to_string(datapoint: DataPoint) -> str:
    """
    Converts a DataPoint object to a string in the format {"input": ..., "output": ...}.
    """
    return f'{{"input":"{datapoint.input}","output":"{datapoint.output}"}}'
