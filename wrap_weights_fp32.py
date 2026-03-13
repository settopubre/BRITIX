import struct
import sys

def create_gguf_wrapper(input_file, output_file):
    print(f"Creating FP32 GGUF wrapper for {input_file}...")
    
    with open(input_file, 'rb') as f:
        weights = f.read()
    
    with open(output_file, 'wb') as f:
        # Header
        f.write(b'GGUF')
        f.write(struct.pack('<I', 3))
        f.write(struct.pack('<Q', 1))
        f.write(struct.pack('<Q', 13))
        
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
            ('general.file_type', '0'),  # 0 = FP32, not 2!
        ]
        
        for key, value in meta:
            f.write(struct.pack('<I', len(key)))
            f.write(key.encode())
            f.write(struct.pack('<I', 8))
            f.write(struct.pack('<Q', len(value)))
            f.write(value.encode())
        
        # Tensor info for FP32 (type 0)
        name = b'weights\0' + b'\0' * 57
        f.write(name)
        f.write(struct.pack('<I', 1))
        f.write(struct.pack('<I', len(weights) // 4))
        f.write(struct.pack('<I', 0))
        f.write(struct.pack('<I', 0))
        f.write(struct.pack('<I', 0))
        f.write(struct.pack('<I', 0))  # 0 = FP32
        f.write(struct.pack('<Q', 1024))
        
        # Write weights
        f.write(weights)
    
    print(f"✅ Created {output_file}")
    print(f"   Type: FP32")
    print(f"   Size: {len(weights)/1e9:.2f} GB")

if __name__ == '__main__':
    create_gguf_wrapper('weights.bin', 'britix-fp32.gguf')
