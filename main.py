import preprocessing
import sentence_encoding
from absl import logging
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from metrics import nearest_neighbor_distances
from analyzer import Analyzer
from generator import Generator

# Instantiate the encoder
encoder = sentence_encoding.SentenceEncoder()

# Reduce logging output.
logging.set_verbosity(logging.ERROR)


# Example usage of the encoder
def embed(input_text):
    return encoder(input_text)


def initialize_model():
    # Set up quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Load the model and tokenizer
    model_4bit = preprocessing.MistralForCausalLMWithSkip.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        device_map="auto",
        quantization_config=quantization_config,
    )

    return model_4bit


def setup_pipeline(model_4bit, tokenizer):
    # Create a pipeline
    pipeline_inst = pipeline(
        "text-generation",
        model=model_4bit,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=2500,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    return pipeline_inst


def synthesize_data(input_output_pairs, model=None):
    # If no model is supplied, initialize one
    if model is None:
        model = initialize_model()

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

    model_pipeline = setup_pipeline(model, tokenizer)

    # Initialize the generator and analyzer on the same model
    generator = Generator(model_pipeline)
    analyzer = Analyzer(model_pipeline)  # Assuming Analyzer handles embeddings and analysis
    task2vec = preprocessing.Task2Vec(model, tokenizer)

    # Generate new data using the input-output pairs
    generated_data = generator.generate(input_output_pairs)

    # Extract inputs from the generated pairs for embedding calculation
    generated_texts = [f"{pair[0]} {pair[1]}" for pair in generated_data]

    # Calculate embeddings for the generated data using the analyzer (or any embedding function)
    embeddings = embed(generated_texts)

    # Calculate nearest neighbor distances using the metrics function
    distances = nearest_neighbor_distances(embeddings)

    # Output the generated data, embeddings, and distances
    print("Generated Data:")
    for i, pair in enumerate(generated_data):
        print(f"Input: {pair[0]}")
        print(f"Output: {pair[1]}")
        print(f"Nearest Neighbor Distance: {distances[i]}\n")

    task2vec.embed(generated_data)

    # Feed the generated pairs along with their respective distances to the analyzer's analyze function
    report = analyzer.analyze(generated_data, distances)

    print(report)

    # Return the generated data, embeddings, and distances for further use
    return generated_data, embeddings, distances


def main():
    # Define 10 input-output pairs as a list of tuples
    input_output_pairs = [
        ("What is the capital of France?", "Paris"),
        ("Who was the first president of the United States?", "George Washington"),
        ("What is the largest planet in our solar system?", "Jupiter"),
        ("What is the boiling point of water?", "100°C"),
        ("Which country is known as the Land of the Rising Sun?", "Japan"),
        ("What is the chemical symbol for water?", "H2O"),
        ("How many continents are there on Earth?", "7"),
        ("What is the currency used in Japan?", "Yen"),
        ("Which animal is known as the King of the Jungle?", "Lion"),
        ("What is the smallest prime number?", "2")
    ]

    # Call the synthesize_data function with the 10 input-output pairs
    synthesize_data(input_output_pairs)


if __name__ == "__main__":
    main()
