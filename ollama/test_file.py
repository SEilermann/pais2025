import ollama
import argparse
import os
import sys


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run an experiment with an Ollama model and optionally save the response.")
    parser.add_argument('--model', type=str, default='llama3.1', help='Model to use (e.g., llama3.1, mistral, gemma3)')
    parser.add_argument('--experiment', type=str, required=True, help='Name of the experiment folder')
    parser.add_argument('--output', type=str, default=None, help='Name of the output text file (optional, skip to avoid saving)')
    parser.add_argument('--prompt', type=str, default='Do you think King Kong is a King or a Kong and is python sexy?', help='Prompt to send to the model')
    args = parser.parse_args()

    # Set experiment name
    experiment_name = args.experiment

    # Initialize Ollama client and generate response
    client = ollama.Client()
    response = client.generate(model=args.model, prompt=args.prompt)

    # Print the output
    print('Generated output:')
    print(response.response)

    # Handle optional saving
    if args.output is not None:
        # Create experiment folder if it doesn't exist
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)
            print(f"Created experiment folder: {experiment_name}")
        else:
            print(f"Experiment folder already exists: {experiment_name}")

        # Define full output path
        output_path = os.path.join(experiment_name, args.output)

        # Check if output file already exists
        if os.path.exists(output_path):
            print(f"File already exists and will not be overwritten: {output_path}")
            sys.exit(0)

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(response.response)
        print(f"Response saved to {output_path}")
    else:
        print("No output file specified — skipping save.")

if __name__ == "__main__":
    main()

