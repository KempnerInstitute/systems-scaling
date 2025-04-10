from transformers import AutoTokenizer

# The identifier for the tokenizer on Hugging Face Hub
tokenizer_id = "google-t5/t5-base" #"KoboldAI/llama2-tokenizer"

try:
    # Load the tokenizer
    # trust_remote_code=True might be needed for some custom tokenizers,
    # but usually not for standard ones like this. Including it just in case.
    # Use legacy=False if you encounter warnings/issues with SentencePiece.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True, legacy=False)

    # Get the vocabulary size
    vocab_size = tokenizer.vocab_size

    # Print the result
    print(f"Successfully loaded tokenizer: '{tokenizer_id}'")
    print(f"Vocabulary size: {vocab_size}")

except Exception as e:
    print(f"An error occurred while loading the tokenizer '{tokenizer_id}':")
    print(e)
    print("\nPlease ensure the tokenizer identifier is correct and you have an internet connection.")
    print("You might need to install dependencies: pip install transformers sentencepiece")