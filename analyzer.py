from langchain import HuggingFacePipeline, PromptTemplate, LLMChain


class Analyzer:
    def __init__(self, model_pipeline):
        # Set the model pipeline (from main.py)
        self.llm = HuggingFacePipeline(pipeline=model_pipeline)

        # Define the prompt template for analysis
        self.template = """[INST] You are an expert data analyst. Your task is to review a dataset
        containing input-output pairs and their nearest neighbor distances. Based on the distances,
        assess whether the dataset has too much diversity, too little diversity, or whether certain 
        datapoints should be removed because they are either too close or too far from other points.

        For each data point, you will be provided:
        - The input.
        - The output.
        - The nearest neighbor distance to other datapoints.

        After reviewing the data, write a report that:
        - Recommends whether the dataset has appropriate diversity.
        - Identifies any datapoints that are too similar (too close).
        - Identifies any datapoints that are too far from the others.
        - Suggests if any datapoints should be cut due to these issues.

        Here are the input-output pairs and their distances:
        {data}

        Please generate your report based on this information. [/INST]"""

        # Set up the PromptTemplate object
        self.prompt = PromptTemplate(template=self.template, input_variables=["data"])

    def analyze(self, generated_data, distances):
        """
        Analyze the generated data and distances, and produce a report.

        Parameters:
        - generated_data: List of input-output pairs.
        - distances: List of nearest neighbor distances corresponding to the input-output pairs.

        Returns:
        - A report generated by the model.
        """

        # Format the input-output pairs and distances for the template
        data_str = "\n".join([f"Input: {pair[0]}\nOutput: {pair[1]}\nNearest Neighbor Distance: {distance}"
                              for pair, distance in zip(generated_data, distances)])

        # Create an LLMChain with the model and the prompt
        llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)

        # Generate the report using the formatted data
        report = llm_chain.run({"data": data_str})

        return report