#!/usr/bin/env elixir

# Basic usage examples for AiFlow.Ollama

# Start the client
{:ok, _pid} = AiFlow.Ollama.start_link(
  hostname: "localhost",
  port: 11434,
  timeout: 60_000
)

IO.puts("=== AiFlow.Ollama Basic Usage Examples ===\n")

# Example 1: List available models
IO.puts("1. Listing available models:")
case AiFlow.Ollama.list_models() do
  {:ok, models} ->
    IO.puts("   Available models: #{length(models)}")
    Enum.each(models, fn model ->
      IO.puts("   - #{model["name"]} (#{model["size"]} bytes)")
    end)

  {:error, reason} ->
    IO.puts("   Error listing models: #{inspect(reason)}")
end

IO.puts()

# Example 2: Simple text generation
IO.puts("2. Text generation:")
case AiFlow.Ollama.query("What is Elixir?", "llama3.1") do
  {:ok, response} ->
    IO.puts("   Response: #{response}")

  {:error, reason} ->
    IO.puts("   Error generating text: #{inspect(reason)}")
end

IO.puts()

# Example 3: Chat session
IO.puts("3. Chat session:")
case AiFlow.Ollama.chat("Hello! How are you?", "example_chat", "example_user", "llama3.1") do
  {:ok, response} ->
    IO.puts("   Assistant: #{response}")

  {:error, reason} ->
    IO.puts("   Error in chat: #{inspect(reason)}")
end

IO.puts()

# Example 4: Continue chat
IO.puts("4. Continuing chat:")
case AiFlow.Ollama.chat("Tell me about functional programming", "example_chat", "example_user", "llama3.1") do
  {:ok, response} ->
    IO.puts("   Assistant: #{response}")

  {:error, reason} ->
    IO.puts("   Error in chat: #{inspect(reason)}")
end

IO.puts()

# Example 5: Show chat history
IO.puts("5. Chat history:")
messages = AiFlow.Ollama.show_chat_history("example_chat", "example_user")
IO.puts("   Messages in history: #{length(messages)}")
Enum.each(messages, fn msg ->
  IO.puts("   #{msg.role}: #{msg.content}")
end)

IO.puts()

# Example 6: Configuration
IO.puts("6. Current configuration:")
IO.puts("   Hostname: #{AiFlow.Ollama.get_hostname()}")
IO.puts("   Port: #{AiFlow.Ollama.get_port()}")
IO.puts("   Timeout: #{AiFlow.Ollama.get_timeout()} ms")

IO.puts()

# Example 7: Debug functions
IO.puts("7. Debug information:")
all_chats = AiFlow.Ollama.show_all_chats()
IO.puts("   Total users: #{map_size(all_chats.chats)}")

IO.puts()

# Example 8: File operations
IO.puts("8. File operations:")
case AiFlow.Ollama.create_blob("test-digest", "nonexistent_file.txt") do
  {:error, {:file_error, :enoent}} ->
    IO.puts("   Expected error: File not found")

  {:error, reason} ->
    IO.puts("   Unexpected error: #{inspect(reason)}")

  {:ok, :success} ->
    IO.puts("   File uploaded successfully")
end

IO.puts()

# Example 9: Bang versions (error handling)
IO.puts("9. Bang versions (error handling):")
try do
  AiFlow.Ollama.query!("This will fail with invalid model", "invalid-model")
  IO.puts("   Unexpected success")
rescue
  e in RuntimeError ->
    IO.puts("   Expected error: #{e.message}")
end

IO.puts()

# Example 10: Clear chat history
IO.puts("10. Clearing chat history:")
AiFlow.Ollama.clear_chat_history()
IO.puts("   Chat history cleared")

IO.puts("\n=== Examples completed ===")
