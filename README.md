# AiFlow

**Streamline your AI workflow with a unified, elegant interface for multiple AI providers**

[![Hex.pm](https://img.shields.io/hexpm/v/ai_flow.svg)](https://hex.pm/packages/ai_flow)
[![Documentation](https://img.shields.io/badge/documentation-gray)](https://hexdocs.pm/ai_flow)
[![License](https://img.shields.io/hexpm/l/ai_flow.svg)](https://github.com/adamanr/ai_flow/blob/main/LICENSE)

---

## Why AiFlow?

Working with different AI models shouldn't feel like herding cats. AiFlow provides a **consistent, developer-friendly interface** that makes integrating AI into your Elixir applications a breeze. Start with Ollama today, with more providers coming soon.

### 🚀 Simple & Intuitive
```elixir
# Ask any question - it's that simple!
{:ok, response} = AiFlow.Ollama.query("Explain quantum computing in simple terms", "llama3.1")
```

### 🔧 Unified API
One interface, multiple AI providers. Switch between services without rewriting your code.

### 🛠️ Production Ready
Built-in error handling, debugging tools, and comprehensive testing.

---

## 🌟 Key Features

- **🧠 Model Management**: List, create, copy, delete, pull, and push models
- **💬 Smart Chat Sessions**: Persistent chat history with automatic context management  
- **✍️ Text Generation**: Powerful prompt completion with customizable parameters
- **🔍 Embeddings**: Generate vector embeddings for semantic search and ML tasks
- **🔄 Blob Operations**: Efficient model file management
- **🛡️ Robust Error Handling**: Comprehensive error management with bang (!) versions
- **🐛 Advanced Debugging**: Built-in tools for troubleshooting and development

---

## 📦 Installation

Add `ai_flow` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:ai_flow, "~> 0.1.0"}
  ]
end
```

---

## ⚙️ Quick Start

### 1. Start the Client

```elixir
# Quick start with defaults
{:ok, pid} = AiFlow.Ollama.start_link()

# Or customize your setup
{:ok, pid} = AiFlow.Ollama.start_link(
  hostname: "localhost",
  port: 11434,
  timeout: 60_000
)
```

### 2. Start Chatting

```elixir
# Simple question
{:ok, response} = AiFlow.Ollama.query("Why is the sky blue?", "llama3.1")

# Interactive chat
{:ok, response} = AiFlow.Ollama.chat("Hello!", "chat_session_1", "user_123", "llama3.1")
{:ok, response} = AiFlow.Ollama.chat("Tell me more about that", "chat_session_1", "user_123", "llama3.1")
```

### 3. Advanced Usage

```elixir
# Generate embeddings for semantic search
{:ok, embeddings} = AiFlow.Ollama.generate_embeddings([
  "The cat sat on the mat",
  "A feline rested on the rug"
])

# Manage your models
{:ok, models} = AiFlow.Ollama.list_models()
{:ok, :success} = AiFlow.Ollama.create_model("my-custom-model", "llama3.1", "You are a helpful coding assistant.")
```

---

## 🎯 Current Capabilities

### Direct Function Calls
Work with AI models intuitively:

* `AiFlow.Ollama.list_models()` - Discover available models  
* `AiFlow.Ollama.query()` - Ask questions to any model
* `AiFlow.Ollama.chat()` - Engage in persistent conversations

### Comprehensive Model Management
```elixir
# Everything you need to manage AI models
AiFlow.Ollama.list_models()
AiFlow.Ollama.create_model("my-model", "base-model", "system prompt")
AiFlow.Ollama.copy_model("original", "backup")
AiFlow.Ollama.delete_model("old-model")
AiFlow.Ollama.pull_model("new-model")
AiFlow.Ollama.push_model("my-model:latest")
```

---

## 🛠️ Configuration

Flexible configuration for any environment:

```elixir
# Application-wide configuration
config :ai_flow, AiFlow.Ollama,
  hostname: "localhost",
  port: 11434,
  timeout: 60_000

# Or per-instance configuration
{:ok, pid} = AiFlow.Ollama.start_link(
  hostname: "production-ai.internal",
  port: 11434,
  timeout: 120_000
)
```

## 🚀 What's Coming Next?

AiFlow is just getting started! Upcoming integrations include:

- **🐝 Bumblebee Integration**: Hugging Face models support
- **☁️ Cloud AI Providers**: OpenAI, Anthropic, Google AI
- **📦 Model Registry**: Centralized model management
- **⚡ Performance Optimizations**: Caching and batching

---

## 🤝 Contributing

We love contributions! Here's how to get started:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📚 Documentation

Full API documentation is available at [HexDocs](https://hexdocs.pm/ai_flow).

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 💬 Get in Touch

- Found a bug? [Open an issue](https://github.com/yourusername/ai_flow/issues)
- Have a feature request? We'd love to hear it!
- Questions? Check out the documentation or open a discussion

**Made with ❤️ for the Elixir community**