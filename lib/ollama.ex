defmodule AiFlow.Ollama do
  @moduledoc """
  Client for interacting with the Ollama API, providing functionality for text generation,
  chat, model management, embeddings, and blob operations.

  ## Configuration
  Configure via `config.exs` or at startup:
  ```elixir
  config :ai_flow_ollama, AiFlow.Ollama,
    hostname: "localhost",
    port: 11434,
    model: llama3.1
    timeout: 120_000,
    chat_file: "chats.json"
  ```

  ## Usage
  ```elixir
  {:ok, _pid} = AiFlow.Ollama.start_link()
  {:ok, text} = AiFlow.Ollama.query("What is Elixir?", "llama3.1")
  {:ok, answer} = AiFlow.Ollama.chat("Hello!", "chat1", "user1")
  {:ok, models} = AiFlow.Ollama.list_models()
  ```

  ## Common Options
  - `:debug` (boolean): Enable debug logging (default: false)
  - `:short` (boolean): Disable short variant response (default: true)
  - `:field` (atom | string): show specificate field from response
  - `:retries` (integer): Number of retries on network error (default: 0)
  - `:cache_ttl` (integer): Cache time-to-live in ms (default: 30_000)
  """

  use Agent
  require Logger

  alias AiFlow.Ollama.{Config, TextGeneration, Chat, Model, Embeddings, Blob}

  @doc """
  Starts the Ollama client with the given configuration.
  """
  @spec start_link(keyword()) :: {:ok, pid()} | {:error, term()}
  def start_link(opts \\ []) do
    config = Config.load(opts)
    Agent.start_link(fn -> config end, name: __MODULE__)
  end

  @doc """
  Gets the current configuration.
  """
  @spec get_config() :: Config.t()
  def get_config, do: Agent.get(__MODULE__, & &1)

  # Delegate functions to submodules
  defdelegate query(prompt, opts \\ []), to: TextGeneration
  defdelegate query!(prompt, opts \\ []), to: TextGeneration
  defdelegate chat(prompt, chat_id, user_id \\ "default_user", opts \\ []), to: Chat, as: :chat
  defdelegate chat!(prompt, chat_id, user_id \\ "default_user", opts \\ []), to: Chat, as: :chat!
  defdelegate list_models(opts \\ []), to: Model, as: :list_models
  defdelegate list_models!(opts \\ []), to: Model, as: :list_models!
  defdelegate show_model(name, opts \\ []), to: Model
  defdelegate show_model!(name, opts \\ []), to: Model
  defdelegate create_model(name, model, system, opts \\ []), to: Model
  defdelegate create_model!(name, model, system, opts \\ []), to: Model
  defdelegate copy_model(source, destination, opts \\ []), to: Model
  defdelegate copy_model!(source, destination, opts \\ []), to: Model
  defdelegate delete_model(name, opts \\ []), to: Model
  defdelegate delete_model!(name, opts \\ []), to: Model
  defdelegate pull_model(name, opts \\ []), to: Model
  defdelegate pull_model!(name, opts \\ []), to: Model
  defdelegate push_model(name, opts \\ []), to: Model
  defdelegate push_model!(name, opts \\ []), to: Model
  defdelegate generate_embeddings(input, opts \\ []), to: Embeddings
  defdelegate generate_embeddings!(input, opts \\ []), to: Embeddings
  defdelegate generate_embeddings_legacy(prompt, opts \\ []), to: Embeddings
  defdelegate generate_embeddings_legacy!(prompt, opts \\ []), to: Embeddings
  defdelegate list_running_models(opts \\ []), to: Model
  defdelegate list_running_models!(opts \\ []), to: Model
  defdelegate load_model(model, opts \\ []), to: Model
  defdelegate load_model!(model, opts \\ []), to: Model
  defdelegate show_chat_history(opts \\ []), to: Chat
  defdelegate show_chat_history!(opts \\ []), to: Chat
  defdelegate show_all_chats, to: Chat
  defdelegate show_all_chats!, to: Chat
  defdelegate clear_chat_history(opts \\ []), to: Chat
  defdelegate clear_chat_history!(opts \\ []), to: Chat
  defdelegate debug_load_chat_data, to: Chat
  defdelegate debug_show_chat_history(chat_id, user_id \\ "default_user"), to: Chat
  defdelegate check_chat_file, to: Chat
  defdelegate check_blob(digest, opts \\ []), to: Blob
  defdelegate check_blob!(digest, opts \\ []), to: Blob
  defdelegate create_blob(digest, file_path, opts \\ []), to: Blob
  defdelegate create_blob!(digest, file_path, opts \\ []), to: Blob
end
