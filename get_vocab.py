from transformers import AutoTokenizer
import json
import sys

print("Downloading tokenizer...")
try:
    # This automatically downloads and caches the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    
    # Get the vocab
    vocab = tokenizer.get_vocab()
    
    # Save it
    with open("vocab.json", "w") as f:
        json.dump(vocab, f)
    
    print(f"✅ Success! Created vocab.json with {len(vocab)} tokens")
    print("📁 File size:", len(json.dumps(vocab)), "bytes")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
