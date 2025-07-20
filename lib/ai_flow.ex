defmodule AiFlow.Ollama do
  @moduledoc """
  # AiFlow.Ollama — Ollama API Client

  This module provides a client for the Ollama API: text generation, chat, model management, embeddings, blob upload/download, debugging, and caching.


  ## Quick Start

  ```elixir
  # Start the client
  {:ok, _pid} = AiFlow.Ollama.start_link()

  # Generate text
  {:ok, text} = AiFlow.Ollama.query("What is Elixir?", "llama3.1")

  # Chat
  {:ok, answer} = AiFlow.Ollama.chat("Hello!", "chat1", "user1")

  # List models
  {:ok, models} = AiFlow.Ollama.list_models()

  # Model info
  {:ok, info} = AiFlow.Ollama.show_model("llama3.1")
  ```

  ## Configuration

  - `hostname` — Ollama host (default: "127.0.0.1")
  - `port` — Ollama port (default: 11434)
  - `timeout` — HTTP request timeout (default: 60_000 ms)
  - `chat_file` — path to chat history file (default: "chats.json")

  Set via `config.exs` or at startup:

  ```elixir
  config :ai_flow_ollama, AiFlow.Ollama,
    hostname: "localhost",
    port: 11434,
    timeout: 120_000

  {:ok, _pid} = AiFlow.Ollama.start_link(hostname: "localhost", port: 11434)
  ```

  ## Common options for most functions
  - `:debug` — log requests/responses (default: false)
  - `:retries` — number of retries on network error (default: 0)
  - `:cache_ttl` — cache time-to-live in ms (default: 30_000)

  ---

  ## API Overview

  ### Agent and config
  - `start_link/1` — start the client
  - `init/1` — initialize state
  - `get_hostname/0`, `get_port/0`, `get_timeout/0` — get current config

  ### Text generation
  - `query/3` — generate text for a prompt
  - `query!/3` — same, but raises on error

  **Example:**
  ```elixir
  {:ok, text} = AiFlow.Ollama.query("What is Elixir?", "llama3.1", debug: true)
  ```

  ### Chat
  - `chat/5` — send a message to a chat, history is saved to file
  - `chat!/5` — same, but raises on error

  **Example:**
  ```elixir
  {:ok, answer} = AiFlow.Ollama.chat("Hello!", "chat1", "user1", "llama3.1")
  ```

  ### Chat history and debugging
  - `show_chat_history/2` — get chat message history
  - `show_all_chats/0` — get all chat data
  - `clear_chat_history/0` — clear chat history
  - `show_chat_file_content/0` — show raw chat file content
  - `debug_load_chat_data/0` — debug chat data loading with logs
  - `debug_show_chat_history/2` — debug print chat history
  - `check_chat_file/0` — check chat file validity

  **Example:**
  ```elixir
  AiFlow.Ollama.clear_chat_history()
  AiFlow.Ollama.show_chat_history("chat1", "user1")
  ```

  ### Model management
  - `list_models/1` — list models (option `:short` for names only)
  - `list_models!/1` — same, but raises on error
  - `show_model/2` — get model info
  - `create_model/4` — create a new model
  - `copy_model/3` — copy a model
  - `delete_model/2` — delete a model
  - `pull_model/2` — pull a model from the repository
  - `push_model/2` — push a model to the repository

  **Example:**
  ```elixir
  {:ok, :success} = AiFlow.Ollama.create_model("my-model", "llama3.1", "You are an assistant")
  {:ok, info} = AiFlow.Ollama.show_model("llama3.1")
  ```

  ### Embeddings
  - `generate_embeddings/3` — embeddings for text/list
  - `generate_embeddings_legacy/3` — legacy endpoint

  **Example:**
  ```elixir
  {:ok, emb} = AiFlow.Ollama.generate_embeddings(["Elixir", "Phoenix"])
  ```

  ### Blobs (files)
  - `check_blob/2` — check if a blob exists
  - `create_blob/3` — upload a blob

  **Example:**
  ```elixir
  {:ok, :exists} = AiFlow.Ollama.check_blob("sha256digest")
  {:ok, :success} = AiFlow.Ollama.create_blob("sha256digest", "model.bin")
  ```

  ### Other
  - `list_running_models/1` — list loaded models
  - `load_model/2` — load a model into memory

  ---

  ## Error handling
  - All functions return `{:ok, result}` or `{:error, %AiFlow.Ollama.Error{...}}`
  - Bang versions (`!`) raise `RuntimeError`
  - Error types: network, http, file, unknown

  **Example:**
  ```elixir
  case AiFlow.Ollama.query("bad", "bad-model") do
    {:ok, text} -> IO.puts(text)
    {:error, err} -> IO.inspect(err)
  end
  ```

  ---

  ## Telemetry
  - All public functions emit telemetry events (see README)

  ---

  See each function's docstring below for details and examples.
  """

  use Agent
  require Logger
  alias AiFlow.Ollama.Error

  @default_hostname "127.0.0.1"
  @default_timeout 60_000
  @default_port 11434
  @chat_file "chats.json"
  @encrypted_chat_file "chats.enc"

  # HTTP client for dependency injection
  defp http_client, do: Application.get_env(:ai_flow, :http_client, AiFlow.HTTPClient)

  @type state :: %{
          hostname: String.t(),
          port: integer(),
          timeout: integer()
        }

  @type response_result :: String.t() | :success | map() | list()
  @type error_reason :: {:http_error, integer(), term()} | term()
  @type opts :: keyword()

  # Models cache (Agent)
  @models_cache :ai_flow_ollama_models_cache

  @model_info_cache :ai_flow_ollama_model_info_cache

  @doc """
  Starts the Ollama client with the given configuration.

  ## Parameters
  - `opts`: Keyword list of options, including:
    - `:hostname` (String.t()): The hostname of the Ollama server (default: #{@default_hostname}).
    - `:port` (integer()): The port of the Ollama server (default: #{@default_port}).
    - `:timeout` (integer()): The request timeout in milliseconds (default: #{@default_timeout}).

  ## Returns
  - `{:ok, pid()}`: On successful start.
  - `{:error, term()}`: If the Agent fails to start.

  ## Example
      iex> {:ok, pid} = AiFlow.Ollama.start_link(hostname: "localhost", port: 11434)
      iex> is_pid(pid)
      true
  """
  @spec start_link(opts()) :: {:ok, pid()} | {:error, term()}
  def start_link(opts \\ []) do
    hostname =
      Keyword.get(
        opts,
        :hostname,
        Application.get_env(:ai_flow_ollama, __MODULE__, [])[:hostname] || @default_hostname
      )

    timeout =
      Keyword.get(
        opts,
        :timeout,
        Application.get_env(:ai_flow_ollama, __MODULE__, [])[:timeout] || @default_timeout
      )

    port =
      Keyword.get(
        opts,
        :port,
        Application.get_env(:ai_flow_ollama, __MODULE__, [])[:port] || @default_port
      )

    chat_file =
      Keyword.get(
        opts,
        :chat_file,
        Application.get_env(:ai_flow_ollama, __MODULE__, [])[:chat_file] || @chat_file
      )

    unless File.exists?(chat_file) do
      File.write!(chat_file, Jason.encode!(%{chats: %{}, created_at: DateTime.utc_now()}))
    end

    Agent.start_link(
      fn -> %{hostname: hostname, port: port, timeout: timeout, chat_file: chat_file} end,
      name: __MODULE__
    )
  end

  @doc """
  Initializes the Ollama client with the given configuration.

  ## Parameters
  - `opts`: Keyword list of options, including:
    - `:hostname` (String.t()): The hostname of the Ollama server (default: #{@default_hostname}).
    - `:port` (integer()): The port of the Ollama server (default: #{@default_port}).
    - `:timeout` (integer()): The request timeout in milliseconds (default: #{@default_timeout}).

  ## Returns
  - `state()`: The initialized state of the client.

  ## Example
      iex> state = AiFlow.Ollama.init(hostname: "localhost")
      iex> state.hostname
      "localhost"
  """
  @spec init(opts()) :: state()
  def init(opts \\ []) do
    hostname =
      Keyword.get(
        opts,
        :hostname,
        Application.get_env(:ai_flow_ollama, __MODULE__, [])[:hostname] || @default_hostname
      )

    timeout =
      Keyword.get(
        opts,
        :timeout,
        Application.get_env(:ai_flow_ollama, __MODULE__, [])[:timeout] || @default_timeout
      )

    port =
      Keyword.get(
        opts,
        :port,
        Application.get_env(:ai_flow_ollama, __MODULE__, [])[:port] || @default_port
      )

    Logger.debug(
      "Initializing AiFlow.Ollama with hostname=#{hostname}, timeout=#{timeout}, port=#{port}"
    )

    new_state = %{hostname: hostname, port: port, timeout: timeout}
    Agent.update(__MODULE__, fn _state -> new_state end)
    new_state
  end

  @doc """
  Gets the configured hostname.

  ## Returns
  - `String.t()`: The hostname.

  ## Example
      iex> AiFlow.Ollama.get_hostname()
      "127.0.0.1"
  """
  @spec get_hostname() :: String.t()
  def get_hostname do
    Agent.get(__MODULE__, & &1.hostname)
  end

  @doc """
  Gets the configured timeout.

  ## Returns
  - `integer()`: The timeout in milliseconds.

  ## Example
      iex> AiFlow.Ollama.get_timeout()
      60_000
  """
  @spec get_timeout() :: integer()
  def get_timeout do
    Agent.get(__MODULE__, & &1.timeout)
  end

  @doc """
  Gets the configured port.

  ## Returns
  - `integer()`: The port number.

  ## Example
      iex> AiFlow.Ollama.get_port()
      11434
  """
  @spec get_port() :: integer()
  def get_port do
    Agent.get(__MODULE__, & &1.port)
  end

  @doc """
  Lists models available locally on the Ollama server.

  ## Parameters
  - `opts`: Keyword list of options, including:
    - `:debug` (boolean()): Enable debug logging (default: false).
    - `:short` (boolean()): Return only model names as a list of strings (default: false).
    - `:retries` (integer()): Number of retries on network error (default: 0).
    - `:cache_ttl` (integer()): Cache TTL in milliseconds (default: 30_000).

  ## Returns
  - `{:ok, list()}`: List of model names (if `:short` is true) or model details.
  - `{:error, error_reason()}`: On failure.

  ## Example
      iex> {:ok, models} = AiFlow.Ollama.list_models()
      iex> is_list(models)
      true

      iex> {:ok, names} = AiFlow.Ollama.list_models(short: true)
      iex> Enum.all?(names, &is_binary/1)
      true
  """
  @spec list_models(opts()) :: {:ok, list()} | {:error, Error.t()}
  def list_models(opts \\ []) do
    state = Agent.get(__MODULE__, & &1)
    timeout = state.timeout
    debug = Keyword.get(opts, :debug, false)
    short = Keyword.get(opts, :short, false)
    retries = Keyword.get(opts, :retries, 0)
    cache_ttl = Keyword.get(opts, :cache_ttl, 30_000)

    cache_key = {state.hostname, state.port}
    now = System.system_time(:millisecond)
    models_cache =
      case :ets.whereis(@models_cache) do
        :undefined ->
          :ets.new(@models_cache, [:named_table, :public, :set])
          @models_cache
        _ ->
          @models_cache
      end

    case :ets.lookup(models_cache, cache_key) do
      [{^cache_key, {cached_models, ts}}] when now - ts < cache_ttl ->
        if debug, do: Logger.debug("Returning models from cache")
        result = if short, do: Enum.map(cached_models, &Map.get(&1, "name")), else: cached_models
        :telemetry.execute([:ai_flow, :ollama, :list_models], %{cache: true}, %{result: :ok})
        {:ok, result}
      _ ->
        url = "http://#{state.hostname}:#{state.port}/api/tags"
        if debug, do: Logger.debug("Ollama list models request: URL=#{url}")
        do_list_models(url, timeout, debug, short, retries, models_cache, cache_key, now, cache_ttl)
    end
  end

  defp do_list_models(url, timeout, debug, short, retries, models_cache, cache_key, now, cache_ttl) do
    :telemetry.execute([:ai_flow, :ollama, :list_models, :request], %{retries: retries}, %{url: url})
    case http_client().get(url, receive_timeout: timeout) do
      {:ok, %Req.Response{status: 200, body: %{"models" => models} = body}} ->
        if debug, do: Logger.debug("Ollama list models response: Status=200, Body=#{inspect(body)}")
        :ets.insert(models_cache, {cache_key, {models, now}})
        result = if short, do: Enum.map(models, &Map.get(&1, "name")), else: models
        :telemetry.execute([:ai_flow, :ollama, :list_models], %{cache: false}, %{result: :ok})
        {:ok, result}
      {:ok, %Req.Response{status: status, body: body}} ->
        Logger.error("Ollama list models request failed with status #{status}: #{inspect(body)}")
        if debug, do: Logger.debug("Ollama list models response: Status=#{status}, Body=#{inspect(body)}")
        :telemetry.execute([:ai_flow, :ollama, :list_models], %{cache: false}, %{result: :error, status: status})
        {:error, Error.http(status, body)}
      {:error, %Req.TransportError{reason: reason}} ->
        if retries > 0 do
          :timer.sleep(200)
          do_list_models(url, timeout, debug, short, retries - 1, models_cache, cache_key, now, cache_ttl)
        else
          Logger.error("Ollama list models request failed: #{inspect(reason)}")
          if debug, do: Logger.debug("Ollama list models error: Reason=#{inspect(reason)}")
          :telemetry.execute([:ai_flow, :ollama, :list_models], %{cache: false}, %{result: :network_error})
          {:error, Error.network(reason)}
        end
      {:error, reason} ->
        Logger.error("Ollama list models request failed: #{inspect(reason)}")
        if debug, do: Logger.debug("Ollama list models error: Reason=#{inspect(reason)}")
        :telemetry.execute([:ai_flow, :ollama, :list_models], %{cache: false}, %{result: :unknown_error})
        {:error, Error.unknown(reason)}
    end
  end

  @doc """
  Lists models available locally, raising on error.

  ## Parameters
  - `opts`: See `list_models/1`.

  ## Returns
  - `list()`: List of model names or details.

  ## Raises
  - `RuntimeError`: If the request fails.

  ## Example
      iex> models = AiFlow.Ollama.list_models!()
      iex> is_list(models)
      true
  """
  @spec list_models!(opts()) :: list()
  def list_models!(opts \\ []) do
    case list_models(opts) do
      {:ok, result} ->
        result

      {:error, {:http_error, status, body}} ->
        raise RuntimeError,
          message: "Ollama list models request failed with status #{status}: #{inspect(body)}"

      {:error, reason} ->
        raise RuntimeError, message: "Ollama list models request failed: #{inspect(reason)}"
    end
  end

  @doc """
  Sends a prompt to the Ollama server to generate a completion.

  ## Parameters
  - `prompt` (String.t()): The prompt to send.
  - `model` (String.t()): The model name (default: "llama3.1").
  - `opts`: Keyword list of options, including:
    - `:debug` (boolean()): Enable debug logging (default: false).
    - Other options are passed as JSON parameters (e.g., `:stream`, `:format`, `:options`).

  ## Returns
  - `{:ok, String.t()}`: The generated response.
  - `{:error, error_reason()}`: On failure.

  ## Example
      iex> {:ok, response} = AiFlow.Ollama.query("Why is the sky blue?")
      iex> is_binary(response)
      true
  """
  @spec query(String.t(), String.t(), opts()) :: {:ok, String.t()} | {:error, Error.t()}
  def query(prompt, model \\ "llama3.1", opts \\ []) do
    state = Agent.get(__MODULE__, & &1)
    timeout = state.timeout
    debug = Keyword.get(opts, :debug, false)
    retries = Keyword.get(opts, :retries, 0)
    url = "http://#{state.hostname}:#{state.port}/api/generate"

    body =
      %{
        model: model,
        prompt: prompt,
        stream: false
      }
      |> Map.merge(Enum.into(Keyword.delete(opts, :debug), %{}))

    if debug do
      Logger.debug("Ollama query request: URL=#{url}, Body=#{inspect(body)}")
    end

    :telemetry.execute([:ai_flow, :ollama, :query, :request], %{retries: retries}, %{url: url})
    do_query(url, body, timeout, debug, retries)
  end

  defp do_query(url, body, timeout, debug, retries) do
    case http_client().post(url, json: body, receive_timeout: timeout) do
      {:ok, %Req.Response{status: 200, body: %{"response" => response} = body}} ->
        if debug, do: Logger.debug("Ollama query response: Status=200, Body=#{inspect(body)}")
        :telemetry.execute([:ai_flow, :ollama, :query], %{}, %{result: :ok})
        {:ok, response}
      {:ok, %Req.Response{status: status, body: body}} ->
        Logger.error("Ollama query request failed with status #{status}: #{inspect(body)}")
        if debug, do: Logger.debug("Ollama query response: Status=#{status}, Body=#{inspect(body)}")
        :telemetry.execute([:ai_flow, :ollama, :query], %{}, %{result: :error, status: status})
        {:error, Error.http(status, body)}
      {:error, %Req.TransportError{reason: reason}} ->
        if retries > 0 do
          :timer.sleep(200)
          do_query(url, body, timeout, debug, retries - 1)
        else
          Logger.error("Ollama query request failed: #{inspect(reason)}")
          if debug, do: Logger.debug("Ollama query error: Reason=#{inspect(reason)}")
          :telemetry.execute([:ai_flow, :ollama, :query], %{}, %{result: :network_error})
          {:error, Error.network(reason)}
        end
      {:error, reason} ->
        Logger.error("Ollama query request failed: #{inspect(reason)}")
        if debug, do: Logger.debug("Ollama query error: Reason=#{inspect(reason)}")
        :telemetry.execute([:ai_flow, :ollama, :query], %{}, %{result: :unknown_error})
        {:error, Error.unknown(reason)}
    end
  end

  @doc """
  Sends a prompt to generate a completion, raising on error.

  ## Parameters
  - `prompt` (String.t()): The prompt to send.
  - `model` (String.t()): The model name (default: "llama3.1").
  - `opts`: See `query/3`.

  ## Returns
  - `String.t()`: The generated response.

  ## Raises
  - `RuntimeError`: If the request fails.

  ## Example
      iex> response = AiFlow.Ollama.query!("Why is the sky blue?")
      iex> is_binary(response)
      true
  """
  @spec query!(String.t(), String.t(), opts()) :: String.t()
  def query!(prompt, model \\ "llama3.1", opts \\ []) do
    case query(prompt, model, opts) do
      {:ok, result} ->
        result

      {:error, {:http_error, status, body}} ->
        raise RuntimeError,
          message: "Ollama query request failed with status #{status}: #{inspect(body)}"

      {:error, reason} ->
        raise RuntimeError, message: "Ollama query request failed: #{inspect(reason)}"
    end
  end

  # Основные функции с полным набором параметров
  @doc """
  Sends a chat message to the Ollama server and stores the conversation in `chats.json` for a specific user.

  ## Parameters
  - `prompt` (String.t()): The user's message.
  - `chat_id` (String.t()): The ID of the chat session.
  - `user_id` (String.t()): The ID of the user (default: "default_user").
  - `model` (String.t()): The model name (default: "llama3.1").
  - `opts`: Keyword list of options, including:
    - `:debug` (boolean()): Enable debug logging (default: false).
    - Other options are passed as JSON parameters (e.g., `:stream`, `:tools`).

  ## Returns
  - `{:ok, String.t()}`: The assistant's response.
  - `{:error, error_reason()}`: On failure.

  ## Examples
      iex> {:ok, response} = AiFlow.Ollama.chat("Hello!", "chat_123")
      iex> is_binary(response)
      true

      iex> {:ok, response} = AiFlow.Ollama.chat("Hello!", "chat_123", "user_456")
      iex> is_binary(response)
      true

      iex> {:ok, response} = AiFlow.Ollama.chat("Hello!", "chat_123", "user_456", "llama3.1")
      iex> is_binary(response)
      true

      iex> {:ok, response} = AiFlow.Ollama.chat("Hello!", "chat_123", "user_456", "llama3.1", debug: true)
      iex> is_binary(response)
      true
  """
  @spec chat(String.t(), String.t(), String.t(), String.t(), opts()) ::
          {:ok, String.t()} | {:error, Error.t()}
  def chat(prompt, chat_id, user_id \\ "default_user", model \\ "llama3.1", opts \\ [])

  def chat(prompt, chat_id, user_id, model, opts) do
    state = Agent.get(__MODULE__, & &1)
    timeout = state.timeout
    debug = Keyword.get(opts, :debug, false)
    retries = Keyword.get(opts, :retries, 0)
    url = "http://#{state.hostname}:#{state.port}/api/chat"

    {:ok, chat_data} = load_encrypted_chat_data()

    user_chats = Map.get(chat_data["chats"], user_id, %{})

    chat_history =
      Map.get(user_chats, chat_id, %{
        name: chat_id,
        model: model,
        messages: [],
        created_at: DateTime.utc_now(),
        updated_at: DateTime.utc_now()
      })

    existing_messages = Map.get(chat_history, :messages, [])

    if debug do
      Logger.debug(
        "Loaded chat history for #{user_id}/#{chat_id}: #{length(existing_messages)} messages"
      )
    end

    new_message = %{role: "user", content: encrypt(prompt), timestamp: DateTime.utc_now()}
    messages_for_ollama = existing_messages ++ [new_message]

    formatted_messages =
      Enum.map(messages_for_ollama, fn msg ->
        %{role: msg.role, content: msg.content}
      end)

    body =
      %{
        model: model,
        messages: formatted_messages,
        stream: false
      }
      |> Map.merge(Enum.into(Keyword.delete(opts, :debug), %{}))

    if debug do
      Logger.debug(
        "Ollama chat request: URL=#{url}, ChatID=#{chat_id}, UserID=#{user_id}, Messages count=#{length(formatted_messages)}, Body=#{inspect(body)}"
      )
    end

    :telemetry.execute([:ai_flow, :ollama, :chat, :request], %{retries: retries}, %{url: url})
    do_chat(url, body, timeout, debug, retries, chat_data, user_id, chat_id, chat_history, messages_for_ollama)
  end

  defp do_chat(url, body, timeout, debug, retries, chat_data, user_id, chat_id, chat_history, messages_for_ollama) do
    case http_client().post(url, json: body, receive_timeout: timeout) do
      {:ok, %Req.Response{status: 200, body: %{"message" => %{"content" => response}} = body}} ->
        if debug, do: Logger.debug("Ollama chat response: Status=200, Body=#{inspect(body)}")
        updated_messages =
          messages_for_ollama ++
            [%{role: "assistant", content: encrypt(response), timestamp: DateTime.utc_now()}]

        updated_chat = %{
          chat_history
          | messages: updated_messages,
            updated_at: DateTime.utc_now()
        }

        user_chats = Map.get(chat_data["chats"], user_id, %{})
        updated_user_chats = Map.put(user_chats, chat_id, updated_chat)
        updated_chats = Map.put(chat_data["chats"], user_id, updated_user_chats)

        updated_chat_data = %{
          chats: updated_chats,
          created_at: chat_data["created_at"]
        }

        save_encrypted_chat_data(updated_chat_data)
        :telemetry.execute([:ai_flow, :ollama, :chat], %{}, %{result: :ok})
        {:ok, decrypt(response)}
      {:ok, %Req.Response{status: status, body: body}} ->
        Logger.error("Ollama chat request failed with status #{status}: #{inspect(body)}")
        if debug, do: Logger.debug("Ollama chat response: Status=#{status}, Body=#{inspect(body)}")
        :telemetry.execute([:ai_flow, :ollama, :chat], %{}, %{result: :error, status: status})
        {:error, Error.http(status, body)}
      {:error, %Req.TransportError{reason: reason}} ->
        if retries > 0 do
          :timer.sleep(200)
          do_chat(url, body, timeout, debug, retries - 1, chat_data, user_id, chat_id, chat_history, messages_for_ollama)
        else
          Logger.error("Ollama chat request failed: #{inspect(reason)}")
          if debug, do: Logger.debug("Ollama chat error: Reason=#{inspect(reason)}")
          :telemetry.execute([:ai_flow, :ollama, :chat], %{}, %{result: :network_error})
          {:error, Error.network(reason)}
        end
      {:error, reason} ->
        Logger.error("Ollama chat request failed: #{inspect(reason)}")
        if debug, do: Logger.debug("Ollama chat error: Reason=#{inspect(reason)}")
        :telemetry.execute([:ai_flow, :ollama, :chat], %{}, %{result: :unknown_error})
        {:error, Error.unknown(reason)}
    end
  end

  @doc """
  Sends a chat message, raising on error.

  ## Parameters
  - `prompt` (String.t()): The user's message.
  - `chat_id` (String.t()): The ID of the chat session.
  - `user_id` (String.t()): The ID of the user (default: "default_user").
  - `model` (String.t()): The model name (default: "llama3.1").
  - `opts`: See `chat/5`.

  ## Returns
  - `String.t()`: The assistant's response.

  ## Raises
  - `RuntimeError`: If the request fails.

  ## Examples
      iex> response = AiFlow.Ollama.chat!("Hello!", "chat_123")
      iex> is_binary(response)
      true

      iex> response = AiFlow.Ollama.chat!("Hello!", "chat_123", "user_456")
      iex> is_binary(response)
      true

      iex> response = AiFlow.Ollama.chat!("Hello!", "chat_123", "user_456", "llama3.1")
      iex> is_binary(response)
      true
  """
  @spec chat!(String.t(), String.t(), String.t(), String.t(), opts()) :: String.t()
  def chat!(prompt, chat_id, user_id \\ "default_user", model \\ "llama3.1", opts \\ [])

  def chat!(prompt, chat_id, user_id, model, opts) do
    case chat(prompt, chat_id, user_id, model, opts) do
      {:ok, result} ->
        result

      {:error, {:http_error, status, body}} ->
        raise RuntimeError,
          message: "Ollama chat request failed with status #{status}: #{inspect(body)}"

      {:error, reason} ->
        raise RuntimeError, message: "Ollama chat request failed: #{inspect(reason)}"
    end
  end


  def clear_chat_history do
    save_encrypted_chat_data(%{"chats" => %{}, "created_at" => DateTime.utc_now()})
    :ok
  end


  def show_chat_history(chat_id, user_id \\ "default_user") do
    {:ok, chat_data} = load_encrypted_chat_data()
    user_chats = Map.get(chat_data["chats"], user_id, %{})
    chat_history = Map.get(user_chats, chat_id, %{messages: []})
    messages = Map.get(chat_history, :messages, [])
    messages
  end


  def show_all_chats do
    {:ok, chat_data} = load_encrypted_chat_data()
    atomize_keys(chat_data)
  end


  def show_chat_file_content do
    case load_encrypted_chat_data() do
      {:ok, data} -> data
      {:error, reason} -> {:error, reason}
    end
  end

  def debug_load_chat_data do
    case load_encrypted_chat_data() do
      {:ok, data} when is_map(data) ->
        atomized = atomize_keys(data)
        IO.inspect(atomized, label: "Debug chat data (atom keys)")
        {:ok, atomized}
      {:ok, data} ->
        IO.inspect(data, label: "Debug chat data (other)")
        {:ok, data}
      {:error, reason} ->
        IO.puts("Error loading chat data: #{inspect(reason)}")
        {:error, reason}
    end
  end

  def debug_show_chat_history(chat_id, user_id \\ "default_user") do
    {:ok, chat_data} = load_encrypted_chat_data()

    user_chats = Map.get(chat_data["chats"], user_id, %{})
    chat_history = Map.get(user_chats, chat_id, %{ "messages" => [] })
    messages = Map.get(chat_history, "messages", [])

    IO.inspect(messages, label: "Debug show chat history messages")

    messages
  end


  def check_chat_file do
    if File.exists?(@chat_file) do
      case File.read(@chat_file) do
        {:ok, content} ->
          case Jason.decode(content) do
            {:ok, data} ->
              if Map.has_key?(data, "chats") do
                users = Map.keys(data["chats"])
                Enum.each(users, fn user ->
                  _chats = Map.keys(data["chats"][user])
                end)
              end
              {:ok, data}
            {:error, reason} ->
              {:error, reason}
          end
        {:error, reason} ->
          {:error, reason}
      end
    else
      {:error, :file_not_found}
    end
  end

  @doc """
  Creates a new model on the Ollama server.

  ## Parameters
  - `name` (String.t()): The name of the model to create.
  - `model` (String.t()): Base model name to create from.
  - `system` (String.t()): System prompt for the new model.
  - `opts`: Keyword list of options, including:
    - `:debug` (boolean): Enable debug logging (default: false).
    - `:stream` (boolean): Enable streaming response (default: true).

  ## Returns
  - `{:ok, :success}`: If the model is created successfully.
  - `{:ok, term()}`: If streaming is enabled and the response is not a success status.
  - `{:error, term()}`: On failure, including HTTP errors or server-reported errors.

  ## Example
      iex> AiFlow.Ollama.create_model("mario", "llama3.1", "You are Mario from Super Mario Bros.")
      {:ok, :success}
  """
  @spec create_model(String.t(), String.t(), String.t(), keyword()) ::
          {:ok, :success | term()} | {:error, Error.t()}
  def create_model(name, model, system, opts \\ []) do
    state = Agent.get(__MODULE__, & &1)
    timeout = state.timeout
    debug = Keyword.get(opts, :debug, false)
    retries = Keyword.get(opts, :retries, 0)
    stream = Keyword.get(opts, :stream, true)
    url = "http://#{state.hostname}:#{state.port}/api/create"

    body =
      %{
        name: name,
        from: model,
        system: system,
        stream: stream
      }
      |> Map.merge(Enum.into(Keyword.drop(opts, [:debug, :stream]), %{}))

    if debug, do: Logger.debug("Ollama create model request: URL=#{url}, Body=#{inspect(body)}")
    :telemetry.execute([:ai_flow, :ollama, :create_model, :request], %{retries: retries}, %{url: url})
    do_create_model(url, body, timeout, debug, retries, stream)
  end

  defp do_create_model(url, body, timeout, debug, retries, stream) do
    case http_client().post(url, json: body, receive_timeout: timeout) do
      {:ok, %Req.Response{status: 200, body: response_body}} ->
        if debug, do: Logger.debug("Ollama create model response: Status=200, Raw Body=#{inspect(response_body)}")
        result = parse_create_response(response_body, stream, debug)
        :telemetry.execute([:ai_flow, :ollama, :create_model], %{}, %{result: :ok})
        case result do
          {:success} -> {:ok, :success}
          {:error, error} -> {:error, Error.unknown(error)}
          {:ok, data} -> {:ok, data}
        end
      {:ok, %Req.Response{status: status, body: response_body}} ->
        parsed_body = safe_parse_json(response_body)
        Logger.error("Ollama create model request failed with status #{status}: #{inspect(parsed_body)}")
        if debug, do: Logger.debug("Ollama create model response: Status=#{status}, Body=#{inspect(parsed_body)}")
        :telemetry.execute([:ai_flow, :ollama, :create_model], %{}, %{result: :error, status: status})
        {:error, Error.http(status, parsed_body)}
      {:error, %Req.TransportError{reason: reason}} ->
        if retries > 0 do
          :timer.sleep(200)
          do_create_model(url, body, timeout, debug, retries - 1, stream)
        else
          Logger.error("Ollama create model request failed: #{inspect(reason)}")
          if debug, do: Logger.debug("Ollama create model error: Reason=#{inspect(reason)}")
          :telemetry.execute([:ai_flow, :ollama, :create_model], %{}, %{result: :network_error})
          {:error, Error.network(reason)}
        end
      {:error, reason} ->
        Logger.error("Ollama create model request failed: #{inspect(reason)}")
        if debug, do: Logger.debug("Ollama create model error: Reason=#{inspect(reason)}")
        :telemetry.execute([:ai_flow, :ollama, :create_model], %{}, %{result: :unknown_error})
        {:error, Error.unknown(reason)}
    end
  end

  @spec show_model(String.t(), opts()) :: {:ok, map()} | {:error, Error.t()}
  def show_model(name, opts \\ []) do
    state = Agent.get(__MODULE__, & &1)
    timeout = state.timeout
    debug = Keyword.get(opts, :debug, false)
    retries = Keyword.get(opts, :retries, 0)
    cache_ttl = Keyword.get(opts, :cache_ttl, 30_000)
    url = "http://#{state.hostname}:#{state.port}/api/show"

    cache_key = {state.hostname, state.port, name}
    now = System.system_time(:millisecond)
    model_info_cache =
      case :ets.whereis(@model_info_cache) do
        :undefined ->
          :ets.new(@model_info_cache, [:named_table, :public, :set])
          @model_info_cache
        _ ->
          @model_info_cache
      end

    case :ets.lookup(model_info_cache, cache_key) do
      [{^cache_key, {cached_info, ts}}] when now - ts < cache_ttl ->
        if debug, do: Logger.debug("Returning model info from cache")
        :telemetry.execute([:ai_flow, :ollama, :show_model], %{cache: true}, %{result: :ok})
        {:ok, cached_info}
      _ ->
        body =
          %{
            name: name
          }
          |> Map.merge(Enum.into(Keyword.delete(opts, :debug), %{}))
        if debug, do: Logger.debug("Ollama show model request: URL=#{url}, Body=#{inspect(body)}")
        :telemetry.execute([:ai_flow, :ollama, :show_model, :request], %{retries: retries}, %{url: url})
        do_show_model(url, body, timeout, debug, retries, model_info_cache, cache_key, now, cache_ttl)
    end
  end

  defp do_show_model(url, body, timeout, debug, retries, model_info_cache, cache_key, now, cache_ttl) do
    case http_client().post(url, json: body, receive_timeout: timeout) do
      {:ok, %Req.Response{status: 200, body: body}} ->
        if debug, do: Logger.debug("Ollama show model response: Status=200, Body=#{inspect(body)}")
        :ets.insert(model_info_cache, {cache_key, {body, now}})
        :telemetry.execute([:ai_flow, :ollama, :show_model], %{cache: false}, %{result: :ok})
        {:ok, body}
      {:ok, %Req.Response{status: status, body: body}} ->
        Logger.error("Ollama show model request failed with status #{status}: #{inspect(body)}")
        if debug, do: Logger.debug("Ollama show model response: Status=#{status}, Body=#{inspect(body)}")
        :telemetry.execute([:ai_flow, :ollama, :show_model], %{cache: false}, %{result: :error, status: status})
        {:error, Error.http(status, body)}
      {:error, %Req.TransportError{reason: reason}} ->
        if retries > 0 do
          :timer.sleep(200)
          do_show_model(url, body, timeout, debug, retries - 1, model_info_cache, cache_key, now, cache_ttl)
        else
          Logger.error("Ollama show model request failed: #{inspect(reason)}")
          if debug, do: Logger.debug("Ollama show model error: Reason=#{inspect(reason)}")
          :telemetry.execute([:ai_flow, :ollama, :show_model], %{cache: false}, %{result: :network_error})
          {:error, Error.network(reason)}
        end
      {:error, reason} ->
        Logger.error("Ollama show model request failed: #{inspect(reason)}")
        if debug, do: Logger.debug("Ollama show model error: Reason=#{inspect(reason)}")
        :telemetry.execute([:ai_flow, :ollama, :show_model], %{cache: false}, %{result: :unknown_error})
        {:error, Error.unknown(reason)}
    end
  end

  @spec copy_model(String.t(), String.t(), opts()) :: {:ok, :success} | {:error, Error.t()}
  def copy_model(source, destination, opts \\ []) do
    state = Agent.get(__MODULE__, & &1)
    timeout = state.timeout
    debug = Keyword.get(opts, :debug, false)
    retries = Keyword.get(opts, :retries, 0)
    url = "http://#{state.hostname}:#{state.port}/api/copy"

    body = %{
      source: source,
      destination: destination
    }

    if debug, do: Logger.debug("Ollama copy model request: URL=#{url}, Body=#{inspect(body)}")
    :telemetry.execute([:ai_flow, :ollama, :copy_model, :request], %{retries: retries}, %{url: url})
    do_copy_model(url, body, timeout, debug, retries)
  end

  defp do_copy_model(url, body, timeout, debug, retries) do
    case http_client().post(url, json: body, receive_timeout: timeout) do
      {:ok, %Req.Response{status: 200}} ->
        if debug, do: Logger.debug("Ollama copy model response: Status=200")
        :telemetry.execute([:ai_flow, :ollama, :copy_model], %{}, %{result: :ok})
        {:ok, :success}
      {:ok, %Req.Response{status: status, body: body}} ->
        Logger.error("Ollama copy model request failed with status #{status}: #{inspect(body)}")
        if debug, do: Logger.debug("Ollama copy model response: Status=#{status}, Body=#{inspect(body)}")
        :telemetry.execute([:ai_flow, :ollama, :copy_model], %{}, %{result: :error, status: status})
        {:error, Error.http(status, body)}
      {:error, %Req.TransportError{reason: reason}} ->
        if retries > 0 do
          :timer.sleep(200)
          do_copy_model(url, body, timeout, debug, retries - 1)
        else
          Logger.error("Ollama copy model request failed: #{inspect(reason)}")
          if debug, do: Logger.debug("Ollama copy model error: Reason=#{inspect(reason)}")
          :telemetry.execute([:ai_flow, :ollama, :copy_model], %{}, %{result: :network_error})
          {:error, Error.network(reason)}
        end
      {:error, reason} ->
        Logger.error("Ollama copy model request failed: #{inspect(reason)}")
        if debug, do: Logger.debug("Ollama copy model error: Reason=#{inspect(reason)}")
        :telemetry.execute([:ai_flow, :ollama, :copy_model], %{}, %{result: :unknown_error})
        {:error, Error.unknown(reason)}
    end
  end

  @spec delete_model(String.t(), opts()) :: {:ok, :success} | {:error, Error.t()}
  def delete_model(name, opts \\ []) do
    state = Agent.get(__MODULE__, & &1)
    timeout = state.timeout
    debug = Keyword.get(opts, :debug, false)
    retries = Keyword.get(opts, :retries, 0)
    url = "http://#{state.hostname}:#{state.port}/api/delete"

    body = %{name: name}

    if debug, do: Logger.debug("Ollama delete model request: URL=#{url}, Body=#{inspect(body)}")
    :telemetry.execute([:ai_flow, :ollama, :delete_model, :request], %{retries: retries}, %{url: url})
    do_delete_model(url, body, timeout, debug, retries)
  end

  defp do_delete_model(url, body, timeout, debug, retries) do
    case http_client().delete(url, json: body, receive_timeout: timeout) do
      {:ok, %Req.Response{status: 200}} ->
        if debug, do: Logger.debug("Ollama delete model response: Status=200")
        :telemetry.execute([:ai_flow, :ollama, :delete_model], %{}, %{result: :ok})
        {:ok, :success}
      {:ok, %Req.Response{status: status, body: body}} ->
        Logger.error("Ollama delete model request failed with status #{status}: #{inspect(body)}")
        if debug, do: Logger.debug("Ollama delete model response: Status=#{status}, Body=#{inspect(body)}")
        :telemetry.execute([:ai_flow, :ollama, :delete_model], %{}, %{result: :error, status: status})
        {:error, Error.http(status, body)}
      {:error, %Req.TransportError{reason: reason}} ->
        if retries > 0 do
          :timer.sleep(200)
          do_delete_model(url, body, timeout, debug, retries - 1)
        else
          Logger.error("Ollama delete model request failed: #{inspect(reason)}")
          if debug, do: Logger.debug("Ollama delete model error: Reason=#{inspect(reason)}")
          :telemetry.execute([:ai_flow, :ollama, :delete_model], %{}, %{result: :network_error})
          {:error, Error.network(reason)}
        end
      {:error, reason} ->
        Logger.error("Ollama delete model request failed: #{inspect(reason)}")
        if debug, do: Logger.debug("Ollama delete model error: Reason=#{inspect(reason)}")
        :telemetry.execute([:ai_flow, :ollama, :delete_model], %{}, %{result: :unknown_error})
        {:error, Error.unknown(reason)}
    end
  end

  @doc """
  Pulls a model from the Ollama library.

  ## Parameters
  - `name` (String.t()): The name of the model to pull (e.g., "llama3.1").
  - `opts`: Keyword list of options, including:
    - `:debug` (boolean()): Enable debug logging (default: false).
    - `:stream` (boolean()): Enable streaming responses (default: true).
    - `:insecure` (boolean()): Allow insecure connections (default: false).

  ## Returns
  - `{:ok, :success | term()}`: `:success` if the pull is successful, or the response body for streaming.
  - `{:error, error_reason()}`: On failure.

  ## Example
      iex> {:ok, response} = AiFlow.Ollama.pull_model("llama3.1")
  """
  @spec pull_model(String.t(), opts()) :: {:ok, response_result()} | {:error, Error.t()}
  def pull_model(name, opts \\ []) do
    state = Agent.get(__MODULE__, & &1)
    timeout = state.timeout
    debug = Keyword.get(opts, :debug, false)
    retries = Keyword.get(opts, :retries, 0)
    url = "http://#{state.hostname}:#{state.port}/api/pull"

    body =
      %{
        name: name,
        stream: Keyword.get(opts, :stream, true)
      }
      |> Map.merge(Enum.into(Keyword.delete(opts, :debug) |> Keyword.delete(:stream), %{}))

    if debug, do: Logger.debug("Ollama pull model request: URL=#{url}, Body=#{inspect(body)}")
    :telemetry.execute([:ai_flow, :ollama, :pull_model, :request], %{retries: retries}, %{url: url})
    do_pull_model(url, body, timeout, debug, retries)
  end

  defp do_pull_model(url, body, timeout, debug, retries) do
    case Req.post(url, json: body, receive_timeout: timeout) do
      {:ok, %Req.Response{status: 200, body: body}} ->
        parsed_body =
          cond do
            is_binary(body) and String.contains?(body, "\n") ->
              body
              |> String.split("\n", trim: true)
              |> Enum.map(fn line ->
                case Jason.decode(line) do
                  {:ok, decoded} -> decoded
                  _ -> line
                end
              end)
            is_binary(body) and String.trim_leading(body) |> String.starts_with?(["{", "["]) ->
              case Jason.decode(body) do
                {:ok, decoded} -> decoded
                _ -> body
              end
            true -> body
          end
        if debug, do: Logger.debug("Ollama pull model response: Status=200, Body=#{inspect(parsed_body)}")
        :telemetry.execute([:ai_flow, :ollama, :pull_model], %{}, %{result: :ok})
        {:ok, parsed_body}
      {:ok, %Req.Response{status: status, body: body}} ->
        parsed_body =
          cond do
            is_binary(body) and String.contains?(body, "\n") ->
              body
              |> String.split("\n", trim: true)
              |> Enum.map(fn line ->
                case Jason.decode(line) do
                  {:ok, decoded} -> decoded
                  _ -> line
                end
              end)
            is_binary(body) and String.trim_leading(body) |> String.starts_with?(["{", "["]) ->
              case Jason.decode(body) do
                {:ok, decoded} -> decoded
                _ -> body
              end
            true -> body
          end
        Logger.error("Ollama pull model request failed with status #{status}: #{inspect(parsed_body)}")
        if debug, do: Logger.debug("Ollama pull model response: Status=#{status}, Body=#{inspect(parsed_body)}")
        :telemetry.execute([:ai_flow, :ollama, :pull_model], %{}, %{result: :error, status: status})
        {:error, Error.http(status, parsed_body)}
      {:error, %Req.TransportError{reason: reason}} ->
        if retries > 0 do
          :timer.sleep(200)
          do_pull_model(url, body, timeout, debug, retries - 1)
        else
          Logger.error("Ollama pull model request failed: #{inspect(reason)}")
          if debug, do: Logger.debug("Ollama pull model error: Reason=#{inspect(reason)}")
          :telemetry.execute([:ai_flow, :ollama, :pull_model], %{}, %{result: :network_error})
          {:error, Error.network(reason)}
        end
      {:error, reason} ->
        Logger.error("Ollama pull model request failed: #{inspect(reason)}")
        if debug, do: Logger.debug("Ollama pull model error: Reason=#{inspect(reason)}")
        :telemetry.execute([:ai_flow, :ollama, :pull_model], %{}, %{result: :unknown_error})
        {:error, Error.unknown(reason)}
    end
  end

  @doc """
  Pulls a model, raising on error.

  ## Parameters
  - `name` (String.t()): The name of the model to pull.
  - `opts`: See `pull_model/2`.

  ## Returns
  - `:success | term()`: The result of the operation.

  ## Raises
  - `RuntimeError`: If the request fails.

  ## Example
      iex> AiFlow.Ollama.pull_model!("llama3.1")
      {response}
  """
  @spec pull_model!(String.t(), opts()) :: response_result()
  def pull_model!(name, opts \\ []) do
    case pull_model(name, opts) do
      {:ok, result} ->
        result

      {:error, {:http_error, status, body}} ->
        raise RuntimeError,
          message: "Ollama pull model request failed with status #{status}: #{inspect(body)}"

      {:error, reason} ->
        raise RuntimeError, message: "Ollama pull model request failed: #{inspect(reason)}"
    end
  end

  @doc """
  Pushes a model to the Ollama library.

  ## Parameters
  - `name` (String.t()): The name of the model to push (e.g., "mattw/pygmalion:latest").
  - `opts`: Keyword list of options, including:
    - `:debug` (boolean()): Enable debug logging (default: false).
    - `:stream` (boolean()): Enable streaming responses (default: true).
    - `:insecure` (boolean()): Allow insecure connections (default: false).

  ## Returns
  - `{:ok, :success | term()}`: `:success` if the push is successful, or the response body for streaming.
  - `{:error, error_reason()}`: On failure.

  ## Example
      iex> {:ok, :success} = AiFlow.Ollama.push_model("mattw/pygmalion:latest")
  """
  @spec push_model(String.t(), opts()) :: {:ok, response_result()} | {:error, error_reason()}
  def push_model(name, opts \\ []) do
    state = Agent.get(__MODULE__, & &1)
    timeout = state.timeout
    debug = Keyword.get(opts, :debug, false)
    url = "http://#{state.hostname}:#{state.port}/api/push"

    body =
      %{
        name: name,
        stream: Keyword.get(opts, :stream, true)
      }
      |> Map.merge(Enum.into(Keyword.delete(opts, :debug) |> Keyword.delete(:stream), %{}))

    if debug do
      Logger.debug("Ollama push model request: URL=#{url}, Body=#{inspect(body)}")
    end

    case Req.post(url, json: body, receive_timeout: timeout) do
      {:ok, %Req.Response{status: 200, body: body}} ->
        parsed_body =
          cond do
            is_binary(body) and String.contains?(body, "\n") ->
              body
              |> String.split("\n", trim: true)
              |> Enum.map(fn line ->
                case Jason.decode(line) do
                  {:ok, decoded} -> decoded
                  _ -> line
                end
              end)
            is_binary(body) and String.trim_leading(body) |> String.starts_with?(["{", "["]) ->
              case Jason.decode(body) do
                {:ok, decoded} -> decoded
                _ -> body
              end
            true -> body
          end
        if debug do
          Logger.debug("Ollama push model response: Status=200, Body=#{inspect(parsed_body)}")
        end

        is_success =
          cond do
            is_list(parsed_body) -> Enum.any?(parsed_body, &is_map(&1) and Map.get(&1, "status") == "success")
            is_map(parsed_body) -> Map.get(parsed_body, "status") == "success"
            true -> false
          end
        if is_success do
          {:ok, :success}
        else
          {:ok, parsed_body}
        end
      {:ok, %Req.Response{status: status, body: body}} ->
        parsed_body =
          cond do
            is_binary(body) and String.contains?(body, "\n") ->
              body
              |> String.split("\n", trim: true)
              |> Enum.map(fn line ->
                case Jason.decode(line) do
                  {:ok, decoded} -> decoded
                  _ -> line
                end
              end)
            is_binary(body) and String.trim_leading(body) |> String.starts_with?(["{", "["]) ->
              case Jason.decode(body) do
                {:ok, decoded} -> decoded
                _ -> body
              end
            true -> body
          end
        Logger.error("Ollama push model request failed with status #{status}: #{inspect(parsed_body)}")
        if debug do
          Logger.debug("Ollama push model response: Status=#{status}, Body=#{inspect(parsed_body)}")
        end
        {:error, {:http_error, status, parsed_body}}
      {:error, reason} ->
        Logger.error("Ollama push model request failed: #{inspect(reason)}")
        if debug do
          Logger.debug("Ollama push model error: Reason=#{inspect(reason)}")
        end
        {:error, reason}
    end
  end

  @doc """
  Pushes a model, raising on error.

  ## Parameters
  - `name` (String.t()): The name of the model to push.
  - `opts`: See `push_model/2`.

  ## Returns
  - `:success | term()`: The result of the operation.

  ## Raises
  - `RuntimeError`: If the request fails.

  ## Example
      iex> AiFlow.Ollama.push_model!("mattw/pygmalion:latest")
      :success
  """
  @spec push_model!(String.t(), opts()) :: response_result()
  def push_model!(name, opts \\ []) do
    case push_model(name, opts) do
      {:ok, result} ->
        result

      {:error, {:http_error, status, body}} ->
        raise RuntimeError,
          message: "Ollama push model request failed with status #{status}: #{inspect(body)}"

      {:error, reason} ->
        raise RuntimeError, message: "Ollama push model request failed: #{inspect(reason)}"
    end
  end

  @doc """
  Generates embeddings for one or more input texts.

  ## Parameters
  - `input` (String.t() | [String.t()]): The text or list of texts to generate embeddings for.
  - `model` (String.t()): The model name (default: "all-minilm").
  - `opts`: Keyword list of options, including:
    - `:debug` (boolean()): Enable debug logging (default: false).
    - Other options are passed as JSON parameters (e.g., `:truncate`, `:options`).

  ## Returns
  - `{:ok, list()}`: The list of embeddings.
  - `{:error, error_reason()}`: On failure.

  ## Example
      iex> {:ok, embeddings} = AiFlow.Ollama.generate_embeddings(["Why is the sky blue?"])
      iex> is_list(embeddings)
      true
  """
  @spec generate_embeddings(String.t() | [String.t()], String.t(), opts()) ::
          {:ok, list()} | {:error, error_reason()}
  def generate_embeddings(input, model \\ "all-minilm", opts \\ [])

  def generate_embeddings(input, model, opts) do
    state = Agent.get(__MODULE__, & &1)
    timeout = state.timeout
    debug = Keyword.get(opts, :debug, false)
    url = "http://#{state.hostname}:#{state.port}/api/embed"

    body =
      %{
        model: model,
        input: input
      }
      |> Map.merge(Enum.into(Keyword.delete(opts, :debug), %{}))

    if debug do
      Logger.debug("Ollama generate embeddings request: URL=#{url}, Body=#{inspect(body)}")
    end

    case Req.post(url, json: body, receive_timeout: timeout) do
      {:ok, %Req.Response{status: 200, body: %{"embeddings" => embeddings} = body}} ->
        if debug do
          Logger.debug("Ollama generate embeddings response: Status=200, Body=#{inspect(body)}")
        end
        {:ok, embeddings}

      {:ok, %Req.Response{status: 404, body: %{"error" => _error} = body}} ->
        Logger.error(
          "Ollama generate embeddings request failed with status 404: #{inspect(body)}"
        )
        if debug do
          Logger.debug("Ollama generate embeddings response: Status=404, Body=#{inspect(body)}")
        end
        Logger.info("Attempting to pull model '#{model}'...")
        case pull_model(model, debug: debug) do
          {:ok, :success} ->
            Logger.info("Model '#{model}' pulled successfully, retrying embedding generation")
            generate_embeddings(input, model, opts)
          {:ok, _} ->
            {:error, {:http_error, 404, body}}
          {:error, pull_reason} ->
            {:error, {:pull_error, pull_reason}}
        end
      {:ok, %Req.Response{status: status, body: body}} ->
        Logger.error(
          "Ollama generate embeddings request failed with status #{status}: #{inspect(body)}"
        )
        if debug do
          Logger.debug(
            "Ollama generate embeddings response: Status=#{status}, Body=#{inspect(body)}"
          )
        end
        {:error, {:http_error, status, body}}
      {:error, reason} ->
        Logger.error("Ollama generate embeddings request failed: #{inspect(reason)}")
        if debug do
          Logger.debug("Ollama generate embeddings error: Reason=#{inspect(reason)}")
        end
        {:error, reason}
    end
  end

  @doc """
  Generates embeddings, raising on error.

  ## Parameters
  - `input` (String.t() | [String.t()]): The text or list of texts.
  - `model` (String.t()): The model name (default: "all-minilm").
  - `opts`: See `generate_embeddings/3`.

  ## Returns
  - `list()`: The list of embeddings.

  ## Raises
  - `RuntimeError`: If the request fails.

  ## Example
      iex> embeddings = AiFlow.Ollama.generate_embeddings!(["Why is the sky blue?"])
      iex> is_list(embeddings)
      true
  """
  @spec generate_embeddings!(String.t() | [String.t()], String.t(), opts()) :: list()
  def generate_embeddings!(input, model \\ "all-minilm", opts \\ [])

  def generate_embeddings!(input, model, opts) do
    case generate_embeddings(input, model, opts) do
      {:ok, result} ->
        result

      {:error, {:http_error, status, body}} ->
        raise RuntimeError,
          message:
            "Ollama generate embeddings request failed with status #{status}: #{inspect(body)}"

      {:error, reason} ->
        raise RuntimeError,
          message: "Ollama generate embeddings request failed: #{inspect(reason)}"
    end
  end

  @doc """
  Lists models currently loaded into memory.

  ## Parameters
  - `opts`: Keyword list of options, including:
    - `:debug` (boolean()): Enable debug logging (default: false).

  ## Returns
  - `{:ok, list()}`: List of running model details.
  - `{:error, error_reason()}`: On failure.

  ## Example
      iex> {:ok, models} = AiFlow.Ollama.list_running_models()
      iex> is_list(models)
      true
  """
  @spec list_running_models(opts()) :: {:ok, list()} | {:error, error_reason()}
  def list_running_models(opts \\ []) do
    state = Agent.get(__MODULE__, & &1)
    timeout = state.timeout
    debug = Keyword.get(opts, :debug, false)
    url = "http://#{state.hostname}:#{state.port}/api/ps"

    if debug do
      Logger.debug("Ollama list running models request: URL=#{url}")
    end

    case http_client().get(url, receive_timeout: timeout) do
      {:ok, %Req.Response{status: 200, body: %{"models" => models} = body}} ->
        if debug do
          Logger.debug("Ollama list running models response: Status=200, Body=#{inspect(body)}")
        end

        {:ok, models}

      {:ok, %Req.Response{status: status, body: body}} ->
        Logger.error(
          "Ollama list running models request failed with status #{status}: #{inspect(body)}"
        )

        if debug do
          Logger.debug(
            "Ollama list running models response: Status=#{status}, Body=#{inspect(body)}"
          )
        end

        {:error, {:http_error, status, body}}

      {:error, reason} ->
        Logger.error("Ollama list running models request failed: #{inspect(reason)}")

        if debug do
          Logger.debug("Ollama list running models error: Reason=#{inspect(reason)}")
        end

        {:error, reason}
    end
  end

  @doc """
  Lists running models, raising on error.

  ## Parameters
  - `opts`: See `list_running_models/1`.

  ## Returns
  - `list()`: List of running model details.

  ## Raises
  - `RuntimeError`: If the request fails.

  ## Example
      iex> models = AiFlow.Ollama.list_running_models!()
      iex> is_list(models)
      true
  """
  @spec list_running_models!(opts()) :: list()
  def list_running_models!(opts \\ []) do
    case list_running_models(opts) do
      {:ok, result} ->
        result

      {:error, {:http_error, status, body}} ->
        raise RuntimeError,
          message:
            "Ollama list running models request failed with status #{status}: #{inspect(body)}"

      {:error, reason} ->
        raise RuntimeError,
          message: "Ollama list running models request failed: #{inspect(reason)}"
    end
  end

  @doc """
  Generates embeddings for a single prompt (legacy endpoint).

  ## Parameters
  - `prompt` (String.t()): The text to generate embeddings for.
  - `model` (String.t()): The model name (default: "all-minilm").
  - `opts`: Keyword list of options, including:
    - `:debug` (boolean()): Enable debug logging (default: false).
    - Other options are passed as JSON parameters (e.g., `:options`, `:keep_alive`).

  ## Returns
  - `{:ok, list()}`: The embedding vector.
  - `{:error, error_reason()}`: On failure.

  ## Example
      iex> {:ok, embedding} = AiFlow.Ollama.generate_embeddings_legacy("Why is the sky blue?")
      iex> is_list(embedding)
      true
  """
  @spec generate_embeddings_legacy(String.t(), String.t(), opts()) ::
          {:ok, list()} | {:error, error_reason()}
  def generate_embeddings_legacy(prompt, model \\ "all-minilm", opts \\ []) do
    state = Agent.get(__MODULE__, & &1)
    timeout = state.timeout
    debug = Keyword.get(opts, :debug, false)
    url = "http://#{state.hostname}:#{state.port}/api/embeddings"

    body =
      %{
        model: model,
        prompt: prompt
      }
      |> Map.merge(Enum.into(Keyword.delete(opts, :debug), %{}))

    if debug do
      Logger.debug("Ollama generate embeddings legacy request: URL=#{url}, Body=#{inspect(body)}")
    end

    case Req.post(url, json: body, receive_timeout: timeout) do
      {:ok, %Req.Response{status: 200, body: %{"embedding" => embedding} = body}} ->
        if debug do
          Logger.debug(
            "Ollama generate embeddings legacy response: Status=200, Body=#{inspect(body)}"
          )
        end

        {:ok, embedding}

      {:ok, %Req.Response{status: status, body: body}} ->
        Logger.error(
          "Ollama generate embeddings legacy request failed with status #{status}: #{inspect(body)}"
        )

        if debug do
          Logger.debug(
            "Ollama generate embeddings legacy response: Status=#{status}, Body=#{inspect(body)}"
          )
        end

        {:error, {:http_error, status, body}}

      {:error, reason} ->
        Logger.error("Ollama generate embeddings legacy request failed: #{inspect(reason)}")

        if debug do
          Logger.debug("Ollama generate embeddings legacy error: Reason=#{inspect(reason)}")
        end

        {:error, reason}
    end
  end

  @doc """
  Generates embeddings (legacy endpoint), raising on error.

  ## Parameters
  - `prompt` (String.t()): The text to generate embeddings for.
  - `model` (String.t()): The model name (default: "all-minilm").
  - `opts`: See `generate_embeddings_legacy/3`.

  ## Returns
  - `list()`: The embedding vector.

  ## Raises
  - `RuntimeError`: If the request fails.

  ## Example
      iex> embedding = AiFlow.Ollama.generate_embeddings_legacy!("Why is the sky blue?")
      iex> is_list(embedding)
      true
  """
  @spec generate_embeddings_legacy!(String.t(), String.t(), opts()) :: list()
  def generate_embeddings_legacy!(prompt, model \\ "all-minilm", opts \\ []) do
    case generate_embeddings_legacy(prompt, model, opts) do
      {:ok, result} ->
        result

      {:error, {:http_error, status, body}} ->
        raise RuntimeError,
          message:
            "Ollama generate embeddings legacy request failed with status #{status}: #{inspect(body)}"

      {:error, reason} ->
        raise RuntimeError,
          message: "Ollama generate embeddings legacy request failed: #{inspect(reason)}"
    end
  end

  @doc """
  Loads a model into memory.

  ## Parameters
  - `model` (String.t()): The name of the model to load.
  - `opts`: Keyword list of options, including:
    - `:debug` (boolean()): Enable debug logging (default: false).

  ## Returns
  - `{:ok, :success}`: If the model is loaded successfully.
  - `{:error, error_reason()}`: On failure.

  ## Example
      iex> {:ok, :success} = AiFlow.Ollama.load_model("llama3.1")
  """
  @spec load_model(String.t(), opts()) :: {:ok, :success} | {:error, error_reason()}
  def load_model(model, opts \\ []) do
    state = Agent.get(__MODULE__, & &1)
    timeout = state.timeout
    debug = Keyword.get(opts, :debug, false)
    url = "http://#{state.hostname}:#{state.port}/api/generate"

    body = %{
      model: model,
      prompt: "",
      stream: false
    }

    if debug do
      Logger.debug("Ollama load model request: URL=#{url}, Body=#{inspect(body)}")
    end

    case Req.post(url, json: body, receive_timeout: timeout) do
      {:ok, %Req.Response{status: 200, body: %{"done" => true} = body}} ->
        if debug do
          Logger.debug("Ollama load model response: Status=200, Body=#{inspect(body)}")
        end

        {:ok, :success}

      {:ok, %Req.Response{status: status, body: body}} ->
        Logger.error("Ollama load model request failed with status #{status}: #{inspect(body)}")

        if debug do
          Logger.debug("Ollama load model response: Status=#{status}, Body=#{inspect(body)}")
        end

        {:error, {:http_error, status, body}}

      {:error, reason} ->
        Logger.error("Ollama load model request failed: #{inspect(reason)}")

        if debug do
          Logger.debug("Ollama load model error: Reason=#{inspect(reason)}")
        end

        {:error, reason}
    end
  end

  @doc """
  Loads a model into memory, raising on error.

  ## Parameters
  - `model` (String.t()): The name of the model to load.
  - `opts`: See `load_model/2`.

  ## Returns
  - `:success`: If the model is loaded successfully.

  ## Raises
  - `RuntimeError`: If the request fails.

  ## Example
      iex> AiFlow.Ollama.load_model!("llama3.1")
      :success
  """
  @spec load_model!(String.t(), opts()) :: :success
  def load_model!(model, opts \\ []) do
    case load_model(model, opts) do
      {:ok, result} ->
        result

      {:error, {:http_error, status, body}} ->
        raise RuntimeError,
          message: "Ollama load model request failed with status #{status}: #{inspect(body)}"

      {:error, reason} ->
        raise RuntimeError, message: "Ollama load model request failed: #{inspect(reason)}"
    end
  end

  @doc false
  defp load_encrypted_chat_data do
    case File.read(@encrypted_chat_file) do
      {:ok, encrypted} ->
        json = decrypt(encrypted)
        case Jason.decode(json) do
          {:ok, data} -> {:ok, data}
          error -> error
        end
      {:error, :enoent} ->
        initial_data = %{chats: %{}, created_at: DateTime.utc_now()}
        save_encrypted_chat_data(initial_data)
        {:ok, initial_data}
      {:error, reason} -> {:error, reason}
    end
  end

  defp save_encrypted_chat_data(data) do
    json = Jason.encode!(data)
    encrypted = encrypt(json)
    File.write!(@encrypted_chat_file, encrypted)
  end

  # Function for safe parsing JSON
  @doc false
  defp safe_parse_json(data) when is_binary(data) do
    case Jason.decode(data) do
      {:ok, parsed} -> parsed
      {:error, _} -> data
    end
  end

  defp safe_parse_json(data), do: data

  # Function for handling the response from the create API
  @doc false
  defp parse_create_response(response_body, stream, debug) when is_binary(response_body) do
    if stream do
      lines = String.split(response_body, "\n", trim: true)
      parsed_lines =
        lines
        |> Enum.map(&safe_parse_json/1)
        |> Enum.reject(&is_binary/1)
      if debug, do: Logger.debug("Parsed streaming lines: #{inspect(parsed_lines)}")
      success_found = Enum.any?(parsed_lines, fn
        %{"status" => "success"} -> true
        _ -> false
      end)
      error_found = Enum.find(parsed_lines, fn
        %{"error" => _} -> true
        _ -> false
      end)
      cond do
        error_found -> {:error, {:server_error, error_found["error"]}}
        success_found -> {:success}
        true -> {:ok, parsed_lines}
      end
    else
      case safe_parse_json(response_body) do
        %{"status" => "success"} -> {:success}
        %{"error" => error} -> {:error, {:server_error, error}}
        parsed_body -> {:ok, parsed_body}
      end
    end
  end

  defp parse_create_response(response_body, _stream, debug) do
    if debug, do: Logger.debug("Response body already parsed: #{inspect(response_body)}")
    case response_body do
      %{"status" => "success"} -> {:success}
      %{"error" => error} -> {:error, {:server_error, error}}
      data -> {:ok, data}
    end
  end

  @doc """
  Gets model info, raising on error.

  ## Parameters
  - `name` (String.t()): The model name.
  - `opts`: See `show_model/2`.

  ## Returns
  - `map()`: Model info.

  ## Raises
  - `RuntimeError`: If the request fails.

  ## Example
      iex> info = AiFlow.Ollama.show_model!("llama3.1")
      iex> is_map(info)
      true
  """
  @spec show_model!(String.t(), opts()) :: map()
  def show_model!(name, opts \\ []) do
    case show_model(name, opts) do
      {:ok, result} -> result
      {:error, {:http_error, status, body}} ->
        raise RuntimeError, message: "Ollama show model request failed with status #{status}: #{inspect(body)}"
      {:error, reason} ->
        raise RuntimeError, message: "Ollama show model request failed: #{inspect(reason)}"
    end
  end

  @doc """
  Creates a new model, raising on error.

  ## Parameters
  - `name` (String.t()): The name of the model to create.
  - `model` (String.t()): Base model name to create from.
  - `system` (String.t()): System prompt for the new model.
  - `opts`: See `create_model/4`.

  ## Returns
  - `:success | term()`: The result of the operation.

  ## Raises
  - `RuntimeError`: If the request fails.

  ## Example
      iex> AiFlow.Ollama.create_model!("mario", "llama3.1", "You are Mario from Super Mario Bros.")
      :success
  """
  @spec create_model!(String.t(), String.t(), String.t(), keyword()) :: :success | term()
  def create_model!(name, model, system, opts \\ []) do
    case create_model(name, model, system, opts) do
      {:ok, result} -> result
      {:error, {:http_error, status, body}} ->
        raise RuntimeError, message: "Ollama create model request failed with status #{status}: #{inspect(body)}"
      {:error, reason} ->
        raise RuntimeError, message: "Ollama create model request failed: #{inspect(reason)}"
    end
  end

  @doc """
  Copies a model, raising on error.

  ## Parameters
  - `source` (String.t()): Source model name.
  - `destination` (String.t()): Destination model name.
  - `opts`: See `copy_model/3`.

  ## Returns
  - `:success`: If the copy is successful.

  ## Raises
  - `RuntimeError`: If the request fails.

  ## Example
      iex> AiFlow.Ollama.copy_model!("llama3.1", "llama3.1-copy")
      :success
  """
  @spec copy_model!(String.t(), String.t(), opts()) :: :success
  def copy_model!(source, destination, opts \\ []) do
    case copy_model(source, destination, opts) do
      {:ok, result} -> result
      {:error, {:http_error, status, body}} ->
        raise RuntimeError, message: "Ollama copy model request failed with status #{status}: #{inspect(body)}"
      {:error, reason} ->
        raise RuntimeError, message: "Ollama copy model request failed: #{inspect(reason)}"
    end
  end

  @doc """
  Deletes a model, raising on error.

  ## Parameters
  - `name` (String.t()): The model name.
  - `opts`: See `delete_model/2`.

  ## Returns
  - `:success`: If the deletion is successful.

  ## Raises
  - `RuntimeError`: If the request fails.

  ## Example
      iex> AiFlow.Ollama.delete_model!("llama3.1")
      :success
  """
  @spec delete_model!(String.t(), opts()) :: :success
  def delete_model!(name, opts \\ []) do
    case delete_model(name, opts) do
      {:ok, result} -> result
      {:error, {:http_error, status, body}} ->
        raise RuntimeError, message: "Ollama delete model request failed with status #{status}: #{inspect(body)}"
      {:error, reason} ->
        raise RuntimeError, message: "Ollama delete model request failed: #{inspect(reason)}"
    end
  end

  @doc """
  Checks if a blob exists on the Ollama server.

  ## Parameters
  - `digest` (String.t()): The blob digest (sha256).
  - `opts`: Keyword list of options, including:
    - `:debug` (boolean): Enable debug logging (default: false).

  ## Returns
  - `{:ok, :exists}`: If the blob exists.
  - `{:ok, :not_found}`: If the blob does not exist.
  - `{:error, reason}`: On error.

  ## Example
      iex> AiFlow.Ollama.check_blob("sha256digest")
      {:ok, :exists}
  """
  @spec check_blob(String.t(), opts()) :: {:ok, :exists | :not_found} | {:error, any()}
  def check_blob(digest, opts \\ []) do
    state = Agent.get(__MODULE__, & &1)
    timeout = state.timeout
    debug = Keyword.get(opts, :debug, false)
    url = "http://#{state.hostname}:#{state.port}/api/blobs/sha256:#{digest}"
    if debug, do: Logger.debug("Ollama check blob request: URL=#{url}")
    case http_client().head(url, receive_timeout: timeout) do
      {:ok, %Req.Response{status: 200}} ->
        if debug, do: Logger.debug("Ollama check blob response: Status=200 (exists)")
        {:ok, :exists}
      {:ok, %Req.Response{status: 404}} ->
        if debug, do: Logger.debug("Ollama check blob response: Status=404 (not found)")
        {:ok, :not_found}
      {:ok, %Req.Response{status: status, body: body}} ->
        Logger.error("Ollama check blob request failed with status #{status}: #{inspect(body)}")
        {:error, {:http_error, status, body}}
      {:error, reason} ->
        Logger.error("Ollama check blob request failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Checks if a blob exists, raising on error.

  ## Parameters
  - `digest` (String.t()): The blob digest (sha256).
  - `opts`: See `check_blob/2`.

  ## Returns
  - `:exists | :not_found`: Blob status.

  ## Raises
  - `RuntimeError`: If the request fails.

  ## Example
      iex> AiFlow.Ollama.check_blob!("sha256digest")
      :exists
  """
  @spec check_blob!(String.t(), opts()) :: :exists | :not_found
  def check_blob!(digest, opts \\ []) do
    case check_blob(digest, opts) do
      {:ok, result} -> result
      {:error, {:http_error, status, body}} ->
        raise RuntimeError, message: "Ollama check blob request failed with status #{status}: #{inspect(body)}"
      {:error, reason} ->
        raise RuntimeError, message: "Ollama check blob request failed: #{inspect(reason)}"
    end
  end

  @doc """
  Uploads a blob to the Ollama server.

  ## Parameters
  - `digest` (String.t()): The blob digest (sha256).
  - `file_path` (String.t()): Path to the file.
  - `opts`: Keyword list of options, including:
    - `:debug` (boolean): Enable debug logging (default: false).

  ## Returns
  - `{:ok, :created}`: If the blob was created.
  - `{:ok, :success}`: If the blob already existed or was updated.
  - `{:error, reason}`: On error.

  ## Example
      iex> AiFlow.Ollama.create_blob("sha256digest", "model.bin")
      {:ok, :created}
  """
  @spec create_blob(String.t(), String.t(), opts()) :: {:ok, :created | :success} | {:error, any()}
  def create_blob(digest, file_path, opts \\ [])

  def create_blob(digest, file_path, opts) do
    state = Agent.get(__MODULE__, & &1)
    timeout = state.timeout
    debug = Keyword.get(opts, :debug, false)
    url = "http://#{state.hostname}:#{state.port}/api/blobs/sha256:#{digest}"
    if debug, do: Logger.debug("Ollama create blob request: URL=#{url}")
    case File.read(file_path) do
      {:ok, content} ->
        case http_client().post(url, [body: content, receive_timeout: timeout]) do
          {:ok, %Req.Response{status: 201}} ->
            if debug, do: Logger.debug("Ollama create blob response: Status=201 (created)")
            {:ok, :created}
          {:ok, %Req.Response{status: 200}} ->
            if debug, do: Logger.debug("Ollama create blob response: Status=200 (success)")
            {:ok, :success}
          {:ok, %Req.Response{status: status, body: body}} ->
            Logger.error("Ollama create blob request failed with status #{status}: #{inspect(body)}")
            {:error, {:http_error, status, body}}
          {:error, reason} ->
            Logger.error("Ollama create blob request failed: #{inspect(reason)}")
            {:error, reason}
        end
      {:error, reason} ->
        Logger.error("Failed to read file #{file_path}: #{inspect(reason)}")
        {:error, {:file_error, reason}}
    end
  end

  @doc """
  Uploads a blob, raising on error.

  ## Parameters
  - `digest` (String.t()): The blob digest.
  - `file_path` (String.t()): Path to the file.
  - `opts`: See `create_blob/3`.

  ## Returns
  - `:created | :success`: Blob upload status.

  ## Raises
  - `RuntimeError`: If the request fails.

  ## Example
      iex> AiFlow.Ollama.create_blob!("sha256digest", "model.bin")
      :created
  """
  @spec create_blob!(String.t(), String.t(), opts()) :: :created | :success
  def create_blob!(digest, file_path, opts \\ []) do
    case create_blob(digest, file_path, opts) do
      {:ok, result} -> result
      {:error, {:http_error, status, body}} ->
        raise RuntimeError, message: "Ollama create blob request failed with status #{status}: #{inspect(body)}"
      {:error, reason} ->
        raise RuntimeError, message: "Ollama create blob request failed: #{inspect(reason)}"
    end
  end

  @doc """
  Returns chat history, raising on error.

  ## Parameters
  - `chat_id` (String.t()): The chat ID.
  - `user_id` (String.t()): The user ID (default: "default_user").

  ## Returns
  - `list()`: List of messages.

  ## Raises
  - `RuntimeError`: If the request fails.

  ## Example
      iex> messages = AiFlow.Ollama.show_chat_history!("chat1", "user1")
      is_list(messages)
      true
  """
  @spec show_chat_history!(String.t(), String.t()) :: list()
  def show_chat_history!(chat_id, user_id \\ "default_user") do
    case show_chat_history(chat_id, user_id) do
      messages when is_list(messages) -> messages
      {:error, reason} ->
        raise RuntimeError, message: "Ollama show chat history failed: #{inspect(reason)}"
    end
  end

  @doc """
  Returns all chat data, raising on error.

  ## Returns
  - `map()`: All chat data.

  ## Raises
  - `RuntimeError`: If the request fails.

  ## Example
      iex> data = AiFlow.Ollama.show_all_chats!()
      is_map(data)
      true
  """
  @spec show_all_chats!() :: map()
  def show_all_chats!() do
    case show_all_chats() do
      data when is_map(data) -> data
      {:error, reason} ->
        raise RuntimeError, message: "Ollama show all chats failed: #{inspect(reason)}"
    end
  end

  @doc """
  Returns raw chat file content, raising on error.

  ## Returns
  - `map()`: Decoded chat file content.

  ## Raises
  - `RuntimeError`: If the request fails.

  ## Example
      iex> content = AiFlow.Ollama.show_chat_file_content!()
      is_map(content)
      true
  """
  @spec show_chat_file_content!() :: map()
  def show_chat_file_content!() do
    case show_chat_file_content() do
      data when is_map(data) -> data
      {:error, reason} ->
        raise RuntimeError, message: "Ollama show chat file content failed: #{inspect(reason)}"
    end
  end

  @doc """
  Clears chat history, raising on error.

  ## Returns
  - `:ok`: On success.

  ## Raises
  - `RuntimeError`: If the request fails.

  ## Example
      iex> AiFlow.Ollama.clear_chat_history!()
      :ok
  """
  @spec clear_chat_history!() :: :ok
  def clear_chat_history!() do
    case clear_chat_history() do
      :ok -> :ok
    end
  end

  # Encryption helpers for chat message content
  @doc false
  @spec encryption_key() :: binary() | nil
  defp encryption_key do
    key = Application.get_env(:ai_flow_ollama, :chat_encryption_key)
    cond do
      is_binary(key) and byte_size(key) == 32 -> key
      is_binary(key) -> Base.decode64(key)
      true -> nil
    end
  end

  @doc false
  defp encrypt(plaintext) when is_binary(plaintext) do
    case encryption_key() do
      nil ->
        Logger.warning("Chat encryption key not set: storing messages in plaintext!")
        plaintext
      key ->
        iv = :crypto.strong_rand_bytes(12)
        {ciphertext, ciphertag} = :crypto.crypto_one_time_aead(:aes_256_gcm, key, iv, plaintext, <<>>, true)
        iv <> ciphertag <> ciphertext
    end
  end

  defp decrypt(encrypted) when is_binary(encrypted) do
    case encryption_key() do
      nil -> encrypted
      key ->
        <<iv::binary-12, ciphertag::binary-16, ciphertext::binary>> = encrypted
        :crypto.crypto_one_time_aead(:aes_256_gcm, key, iv, ciphertext, <<>>, ciphertag, false)
    end
  end

  # Helper to convert string-keyed maps to atom-keyed maps (shallow)
  defp atomize_keys(map) when is_map(map) do
    map
    |> Enum.map(fn {k, v} ->
      key = if is_binary(k), do: String.to_atom(k), else: k
      {key, v}
    end)
    |> Enum.into(%{})
  end
end
