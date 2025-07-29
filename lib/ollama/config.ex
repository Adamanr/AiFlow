defmodule AiFlow.Ollama.Config do
  @moduledoc """
  Configuration module for the AiFlow Ollama client.

  This module defines the `AiFlow.Ollama.Config` struct and provides functions
  for loading configuration from application environment or runtime options,
  and for building API URLs.

  ## Configuration

  The configuration can be set in your application's `config/config.exs`:

      config :ai_flow_ollama, AiFlow.Ollama,
        hostname: "localhost",
        port: 11434,
        model: "llama3.1",
        timeout: 120_000,
        chat_file: "path/to/chats.json"

  Runtime options passed to `load/1` will override application environment settings.
  """

  @enforce_keys []
  defstruct hostname: "127.0.0.1",
            port: 11_434,
            model: "llama3.1",
            timeout: 60_000,
            chat_file: "chats.json"

  @typedoc """
  The configuration struct for AiFlow Ollama client.

  ## Fields

  - `hostname`: The Ollama server hostname (default: `"127.0.0.1"`).
  - `port`: The Ollama server port (default: `11_434`).
  - `model`: The default model name to use (default: `"llama3.1"`).
  - `timeout`: Request timeout in milliseconds (default: `60_000`).
  - `chat_file`: Path to the JSON file for storing chat history (default: `"chats.json"`).
  """
  @type t :: %__MODULE__{
          hostname: String.t(),
          port: integer(),
          model: String.t(),
          timeout: integer(),
          chat_file: String.t()
        }

  @doc """
  Loads the configuration, merging runtime options with application environment defaults.

  This function retrieves configuration values from the application environment
  (`config/config.exs`) under `:ai_flow_ollama, AiFlow.Ollama`, and allows
  overriding them with runtime options passed as a keyword list.

  If the configured `chat_file` does not exist, it initializes an empty chat data file.

  ## Parameters

  - `opts`: A keyword list of configuration options to override defaults.
    Supported keys: `:hostname`, `:port`, `:model`, `:timeout`, `:chat_file`.

  ## Returns

  - `%AiFlow.Ollama.Config{}`: A struct containing the loaded configuration.

  ## Examples

      # Load with default configuration
      config = AiFlow.Ollama.Config.load([])

      # Override specific options
      config = AiFlow.Ollama.Config.load(hostname: "localhost", port: 8080)

      # With no application environment set, uses built-in defaults
      config = AiFlow.Ollama.Config.load(model: "mistral")
  """
  @spec load(keyword()) :: t()
  def load(opts \\ []) do
    defaults = Application.get_env(:ai_flow_ollama, AiFlow.Ollama, [])

    config = %__MODULE__{
      hostname: Keyword.get(opts, :hostname, Keyword.get(defaults, :hostname, "127.0.0.1")),
      port: Keyword.get(opts, :port, Keyword.get(defaults, :port, 11_434)),
      model: Keyword.get(opts, :model, Keyword.get(defaults, :model, "llama3.1")),
      timeout: Keyword.get(opts, :timeout, Keyword.get(defaults, :timeout, 60_000)),
      chat_file: Keyword.get(opts, :chat_file, Keyword.get(defaults, :chat_file, "chats.json"))
    }

    ensure_chat_file_exists(config.chat_file)
    config
  end

  @doc """
  Builds a full URL for an Ollama API endpoint.

  Combines the configured hostname and port with a given API endpoint path.

  ## Parameters

  - `config`: The `%AiFlow.Ollama.Config{}` struct.
  - `endpoint`: A string representing the API endpoint path (e.g., "/api/generate").

  ## Returns

  - `String.t()`: The complete URL.

  ## Examples

      iex> config = %AiFlow.Ollama.Config{hostname: "localhost", port: 11434}
      iex> AiFlow.Ollama.Config.build_url(config, "/api/generate")
      "http://localhost:11434/api/generate"
  """
  @spec build_url(t(), String.t()) :: String.t()
  def build_url(%__MODULE__{} = config, endpoint) when is_binary(endpoint) do
    "http://#{config.hostname}:#{config.port}#{endpoint}"
  end

  # Ensures the chat file exists, creating it with initial data if it doesn't.
  @spec ensure_chat_file_exists(String.t()) :: :ok
  defp ensure_chat_file_exists(chat_file_path) do
    unless File.exists?(chat_file_path) do
      initial_data = %{
        "chats" => %{},
        "created_at" => DateTime.utc_now() |> DateTime.to_iso8601()
      }

      with {:ok, json_data} <- Jason.encode(initial_data, pretty: true),
           :ok <- File.write(chat_file_path, json_data) do
        :ok
      else
        {:error, reason} ->
          require Logger
          Logger.warning("Failed to create initial chat file '#{chat_file_path}': #{inspect(reason)}")
          :ok
      end
    else
      :ok
    end
  end
end
