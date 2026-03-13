import struct
import sys

def create_gguf_wrapper(input_file, output_file):
    print(f"Creating GGUF wrapper for {input_file}...")
    
    # Read the raw weights
    with open(input_file, 'rb') as f:
        weights = f.read()
    
    # Create GGUF file
    with open(output_file, 'wb') as f:
        # Header
        f.write(b'GGUF')  # magic
        f.write(struct.pack('<I', 3))  # version
        f.write(struct.pack('<Q', 1))  # tensor count (simplified)
        f.write(struct.pack('<Q', 13))  # metadata count
        
        # Metadata
        meta = [
            ('general.architecture', 'llama'),
            ('general.name', 'Britix 8B'),
            ('llama.context_length', '8192'),
            ('llama.embedding_length', '4096'),
            ('llama.block_count', '32'),
            ('llama.feed_forward_length', '14336'),
            ('llama.attention.head_count', '32'),
            ('llama.attention.head_count_kv', '8'),
            ('llama.rope.dimension_count', '128'),
            ('llama.attention.layer_norm_rms_epsilon', '1e-5'),
            ('tokenizer.ggml.model', 'llama'),
            ('tokenizer.ggml.pre', 'default'),
            ('general.file_type', '2'),
        ]
        
        for key, value in meta:
            f.write(struct.pack('<I', len(key)))
            f.write(key.encode())
            f.write(struct.pack('<I', 8))  # string type
            f.write(struct.pack('<Q', len(value)))
            f.write(value.encode())
        
        # Tensor info
        name = b'weights\0' + b'\0' * 57
        f.write(name)
        f.write(struct.pack('<I', 1))  # n_dims
        f.write(struct.pack('<I', len(weights) // 4))  # dim0 (number of floats)
        f.write(struct.pack('<I', 0))
        f.write(struct.pack('<I', 0))
        f.write(struct.pack('<I', 0))
        f.write(struct.pack('<I', 2))  # type Q4_0
        f.write(struct.pack('<Q', 1024))  # offset
        
        # Write weights
        f.write(weights)
    
    print(f"✅ Created {output_file}")
    print(f"   Size: {len(weights)/1e9:.2f} GB")

if __name__ == '__main__':
    create_gguf_wrapper('weights.bin', 'britix-wrapped.gguf')
