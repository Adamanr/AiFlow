defmodule AiFlow do
  @moduledoc """
  AiFlow - Unified interface for working with various AI models and services.

  This library provides a convenient and consistent API for interacting with
  different AI providers and models. Currently, it supports Ollama API, with
  plans to expand support to include other AI frameworks like Bumblebee,
  Hugging Face, and more.

  ## Current Support

  * **Ollama**: Local LLM inference with support for model loading, generation,
    and embedding operations

  ## Future Plans

  * **Bumblebee Integration**: Support for Hugging Face models through Bumblebee
  * **Additional AI Providers**: Integration with cloud-based AI services
  * **Advanced Features**: Model management, caching, and performance optimization

  ## Usage

  In its current form, the library supports direct function calls without specifying the full module path:

  * `AiFlow.Ollama.list_models` - Get currently installed models in Ollama
  * `AiFlow.Ollama.query` - Ask a question to an Ollama model
  * `AiFlow.Ollama.chat` - Start a chat with an Ollama model

  Example usage:

  ```elixir
  # Load a model
  AiFlow.Ollama.load_model("llama3.1")

  # Generate text
  AiFlow.Ollama.query("Hello, how are you?", model: "llama3.1")
  ```

  For detailed documentation, see the specific modules for each AI service.
  """
end
