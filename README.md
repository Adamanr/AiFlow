# AiFlow

A simple Elixir library for interacting with the Ollama API.

## Features

- **Model Management**: List, create, copy, delete, pull, and push models
- **Chat Sessions**: Maintain chat history with persistent storage
- **Text Generation**: Send prompts and get completions
- **Embeddings**: Generate embeddings for text
- **Blob Operations**: Upload and manage model files
- **Error Handling**: Comprehensive error handling with bang (!) versions
- **Debugging**: Built-in debugging functions for troubleshooting

## Installation

Add `ai_flow` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:ai_flow, "~> 0.1.0"}
  ]
end
```

## Configuration

You can configure the Ollama client via application environment or by passing options to `start_link/1`:

```elixir
# In config.exs
config :ai_flow, AiFlow.Ollama,
  hostname: "localhost",
  port: 11434,
  timeout: 60_000

# Or when starting
{:ok, pid} = AiFlow.Ollama.start_link(
  hostname: "localhost",
  port: 11434,
  timeout: 60_000
)
```

## Usage

### Basic Setup

```elixir
# Start the client
{:ok, pid} = AiFlow.Ollama.start_link()

# Or with custom configuration
{:ok, pid} = AiFlow.Ollama.start_link(
  hostname: "localhost",
  port: 11434,
  timeout: 60_000
)
```

### Text Generation

```elixir
# Simple query
{:ok, response} = AiFlow.Ollama.query("Why is the sky blue?", "llama3.1")

# With options
{:ok, response} = AiFlow.Ollama.query(
  "Hello, world!", 
  "llama3.1", 
  temperature: 0.7,
  top_p: 0.9
)

# Bang version (raises on error)
response = AiFlow.Ollama.query!("Hello, world!", "llama3.1")
```

### Chat Sessions

```elixir
# Start a chat session
{:ok, response} = AiFlow.Ollama.chat("Hello!", "chat_123", "user_456", "llama3.1")

# Continue the conversation
{:ok, response} = AiFlow.Ollama.chat("How are you?", "chat_123", "user_456", "llama3.1")

# View chat history
messages = AiFlow.Ollama.show_chat_history("chat_123", "user_456")

# Clear chat history
AiFlow.Ollama.clear_chat_history()
```

### Model Management

```elixir
# List available models
{:ok, models} = AiFlow.Ollama.list_models()
{:ok, names} = AiFlow.Ollama.list_models(short: true)

# Create a new model
{:ok, :success} = AiFlow.Ollama.create_model(
  "my-model", 
  "llama3.1", 
  "You are a helpful assistant."
)

# Show model information
{:ok, info} = AiFlow.Ollama.show_model("llama3.1")

# Copy a model
{:ok, :success} = AiFlow.Ollama.copy_model("llama3.1", "llama3.1-backup")

# Delete a model
{:ok, :success} = AiFlow.Ollama.delete_model("old-model")

# Pull a model from the library
{:ok, :success} = AiFlow.Ollama.pull_model("llama3.1")

# Push a model to the library
{:ok, :success} = AiFlow.Ollama.push_model("my-model:latest")
```

### Embeddings

```elixir
# Generate embeddings for a single text
{:ok, embeddings} = AiFlow.Ollama.generate_embeddings("Hello, world!")

# Generate embeddings for multiple texts
{:ok, embeddings} = AiFlow.Ollama.generate_embeddings([
  "Hello, world!",
  "How are you?"
])

# Legacy endpoint
{:ok, embedding} = AiFlow.Ollama.generate_embeddings_legacy("Hello, world!")
```

### Blob Operations

```elixir
# Check if a blob exists
{:ok, :exists} = AiFlow.Ollama.check_blob("digest")

# Create a blob from a file
{:ok, :success} = AiFlow.Ollama.create_blob("digest", "model.bin")
```

### Running Models

```elixir
# List running models
{:ok, models} = AiFlow.Ollama.list_running_models()

# Load a model into memory
{:ok, :success} = AiFlow.Ollama.load_model("llama3.1")
```

### Debugging

```elixir
# Show all chats
all_chats = AiFlow.Ollama.show_all_chats()

# Debug chat data loading
{:ok, chat_data} = AiFlow.Ollama.debug_load_chat_data()

# Check chat file
{:ok, chat_data} = AiFlow.Ollama.check_chat_file()

# Debug chat history
messages = AiFlow.Ollama.debug_show_chat_history("chat_id", "user_id")
```

## Testing

The library includes comprehensive tests covering:

### Unit Tests (`test/ai_flow_test.exs`)

- **Configuration**: Testing default and custom configuration
- **Chat History Management**: Testing chat file operations
- **Debug Functions**: Testing debugging utilities
- **File Operations**: Testing file handling
- **Configuration Functions**: Testing getter functions

### Integration Tests (`test/ai_flow_integration_test.exs`)

- **Network Error Handling**: Testing behavior when server is unavailable
- **Bang Versions**: Testing error-raising versions of functions
- **File Operations**: Testing file operations with real files
- **Configuration**: Testing configuration functions

### Running Tests

```bash
# Run all tests
mix test

# Run with detailed output
mix test --trace

# Run specific test file
mix test test/ai_flow_test.exs

# Run specific test
mix test test/ai_flow_test.exs:89
```

### Test Coverage

The tests cover:

1. **Configuration Management**
   - Default configuration
   - Custom configuration
   - Configuration getters

2. **Chat History**
   - Chat file creation and management
   - History clearing
   - Debug functions

3. **Error Handling**
   - Network errors
   - File errors
   - HTTP errors
   - Bang versions

4. **File Operations**
   - File reading
   - File writing
   - Error handling

5. **Debug Functions**
   - Chat data loading
   - File content inspection
   - History debugging

## Error Handling

All functions return `{:ok, result}` on success or `{:error, reason}` on failure. Bang versions (e.g., `query!/3`) raise a `RuntimeError` on failure.

Common error types:
- `{:http_error, status, body}`: HTTP request failed
- `{:file_error, reason}`: File operation failed
- `{:network_error, reason}`: Network connection failed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License.

