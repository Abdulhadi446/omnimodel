"""
Configuration for the main router model
"""

# Model architecture
MODEL_CONFIG = {
    "vocab_size": 59496,  # From unified tokenizer
    "hidden_dim": 384,  # Reduced from 512 to keep total params <50M
    "num_layers": 6,
    "num_heads": 8,
    "ffn_dim": 1536,  # Reduced proportionally
    "max_seq_length": 2048,
    "dropout": 0.1,
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 16,
    "learning_rate": 1e-4,
    "warmup_steps": 500,
    "num_epochs": 10,
    "save_every_n_steps": 500,
    "eval_every_n_steps": 100,
    "device": "cpu",  # CPU-only for compatibility
    "mixed_precision": False,  # FP32 only
}

# Inference configuration
INFERENCE_CONFIG = {
    "max_new_tokens": 50,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
}

# Supported modalities and styles
MODALITIES = {
    "text": ["human_like", "creative", "professional", "simple", "code", "mcp_handler", "general"],
    "image": ["realism", "cartoon", "professional", "simple", "general"],
    "audio": ["realism", "professional", "simple", "general"],
    "video": ["realism", "cartoon", "simple", "general"],
    "pdf": ["realism", "creative", "professional", "simple", "general"],
}

# Tool registry
TOOLS = {
    "web_search": {"description": "Search the web using DuckDuckGo"},
    "calculator": {"description": "Evaluate mathematical expressions"},
    "code_executor": {"description": "Execute code with timeout"},
    "file_read": {"description": "Read local files"},
    "file_write": {"description": "Write to local files"},
    "image_capture": {"description": "Capture image from webcam"},
}

# Paths
PATHS = {
    "models": "./models",
    "checkpoints": "./checkpoints",
    "data": "./training/data",
    "outputs": "./outputs",
}
