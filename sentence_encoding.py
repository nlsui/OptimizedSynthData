from absl import logging
import tensorflow_hub as hub


class SentenceEncoder:
    def __init__(self, module_url="https://tfhub.dev/google/universal-sentence-encoder/4"):
        # Load the model from the provided URL
        self.model = hub.load(module_url)
        print("Module {} loaded".format(module_url))
        # Reduce logging output.
        logging.set_verbosity(logging.ERROR)

    def __call__(self, input_text):
        # This allows the instance to be called like a function
        return self.model(input_text)

