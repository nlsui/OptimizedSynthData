import random
import numpy as np
import torch
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain

from data_format import _parse_generated_pairs, _datapoint_to_string


class Generator:
    def __init__(self, model_pipeline, block_size):
        # Set the model pipeline (from main.py)
        self.llm = HuggingFacePipeline(pipeline=model_pipeline)

        # Define the template for generating similar data points
        self.template = """[INST] You are a data generation assistant. Your task is to generate new data points 
        that are similar to the given input-output pairs. Use the input-output format provided and generate 
        new examples that follow the same structure. The format is:
        Input: <input_example>
        Output: <output_example>

        Here are the input-output pairs:
        {pairs}

        Now generate {block_size} new input-output pairs following the format. Ensure the new pairs are realistic and parsable.
        Only output the new data points in the same format, do not include any extra text. [/INST]"""

        # Set up the PromptTemplate object
        self.prompt = PromptTemplate(template=self.template, input_variables=["pairs", "block_size"])
        self.block_size = block_size

    def generate(self, input_output_pairs):
        # Convert list of input-output pairs to string format
        pairs_str = "\n".join([_datapoint_to_string(dp) for dp in input_output_pairs])

        random.seed(None)
        torch.manual_seed(torch.seed())
        np.random.seed(None)

        # Create an LLMChain with the model and the prompt
        llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)

        # Generate a response using the string format of pairs
        response = llm_chain.run({"pairs": pairs_str, "block_size": self.block_size})

        print(response)

        # Split the response into new input-output pairs
        generated_pairs = _parse_generated_pairs(response)

        return generated_pairs
