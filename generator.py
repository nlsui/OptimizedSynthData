from langchain import HuggingFacePipeline, PromptTemplate, LLMChain


def _parse_generated_pairs(response_text):
    lines = response_text.splitlines()
    new_pairs = []
    input_text, output_text = None, None

    for line in lines:
        if line.startswith("Input:"):
            if input_text and output_text:
                new_pairs.append((input_text, output_text))
            input_text = line.replace("Input:", "").strip()
            output_text = None  # Reset output
        elif line.startswith("Output:"):
            output_text = line.replace("Output:", "").strip()

    if input_text and output_text:
        new_pairs.append((input_text, output_text))

    return new_pairs


class Generator:
    def __init__(self, model_pipeline):
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

        Now generate new input-output pairs following the format. Ensure the new pairs are realistic and parsable.
        Only output the new data points in the same format, do not include any extra text. [/INST]"""

        # Set up the PromptTemplate object
        self.prompt = PromptTemplate(template=self.template, input_variables=["pairs"])

    def generate(self, input_output_pairs):
        # Convert list of input-output pairs to string format
        pairs_str = "\n".join([f"Input: {pair[0]}\nOutput: {pair[1]}" for pair in input_output_pairs])

        # Create an LLMChain with the model and the prompt
        llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)

        # Generate a response using the string format of pairs
        response = llm_chain.run({"pairs": pairs_str})

        # Split the response into new input-output pairs
        generated_pairs = _parse_generated_pairs(response)

        return generated_pairs
