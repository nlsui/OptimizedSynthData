from absl import logging
import tensorflow_hub as hub


class SentenceEncoder:
    def __init__(self, module_url="https://tfhub.dev/google/universal-sentence-encoder/4"):
        # Load the model from the provided URL
        self.model = hub.load(module_url)
        print("Module {} loaded".format(module_url))
        # Reduce logging output.
        logging.set_verbosity(logging.ERROR)

    def __call__(self, data_points):
        """
        Embeds the input-output pairs in the given DataPoint objects and stores the embedding in the 'embedding' attribute.

        Parameters:
        data_points (list of DataPoint): A list of DataPoint objects.

        Returns:
        List[DataPoint]: The same list of DataPoint objects with the 'embedding' attribute populated.
        """
        # Prepare input text by combining 'input' and 'output' from each DataPoint
        input_text = [f"{dp.input} {dp.output}" for dp in data_points]

        # Generate embeddings by calling the model
        embeddings = self.model(input_text)

        # Populate the 'embedding' attribute for each DataPoint
        for dp, embedding in zip(data_points, embeddings):
            dp.embedding = embedding

        return data_points

