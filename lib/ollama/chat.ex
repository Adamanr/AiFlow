defmodule AiFlow.Ollama.Chat do
  @moduledoc """
  Handles chat interactions with the Ollama API and manages conversation history.

  This module allows sending messages to Ollama chat models and automatically
  maintains a persistent chat history. It supports various options for
  customizing requests and responses.

  ## Features

  - Automatic chat history persistence (JSON file)
  - Conversation management by `chat_id` and `user_id`
  - Automatic model pulling when a model is not found
  - Support for standard and raising (`!`) function variants
  - Configurable response formatting with `:short` and `:field` options
  - Debug logging

  ## Examples

      # Simple chat interaction
      {:ok, response} = AiFlow.Ollama.Chat.chat("Hello!", "my_chat", "user1")

      # Get full API response
      {:ok, full_response} = AiFlow.Ollama.Chat.chat("Hello!", "my_chat", "user1", short: false)

      # Extract specific field from API response
      {:ok, model_name} = AiFlow.Ollama.Chat.chat("Hello!", "my_chat", "user1", short: false, field: "model")

      # Extract specific field from message object
      {:ok, role} = AiFlow.Ollama.Chat.chat("Hello!", "my_chat", "user1", field: "role")

      # Use a specific model
      {:ok, response} = AiFlow.Ollama.Chat.chat("Bonjour!", "french_chat", "user1", model: "mistral")

      # Enable debug logging
      {:ok, response} = AiFlow.Ollama.Chat.chat("Debug me", "debug_chat", debug: true)

      # View chat history
      {:ok, history} = AiFlow.Ollama.Chat.show_chat_history("my_chat", "user1")

      # Raise on error instead of returning {:error, ...}
      response = AiFlow.Ollama.Chat.chat!("Hello!", "my_chat", "user1")
  """

  require Logger
  alias AiFlow.Ollama.{Config, Error, HTTPClient, Model}

  # Use configurable file module for easier testing/mocking
  @file_module Application.compile_env(:ai_flow, :file_module, File)

  @doc """
  Sends a chat message to the Ollama API and stores the conversation in the chat history.

  ## Parameters

  - `prompt`: The user's message (string).
  - `chat_id`: Unique identifier for the chat session (string).
  - `user_id`: Identifier for the user (string, defaults to `"default_user"`).
  - `opts`: Keyword list of options:
    - `:model` (string): The Ollama model to use (defaults to config model).
    - `:debug` (boolean): If `true`, logs request/response details (defaults to `false`).
    - `:short` (boolean): Controls response format.
      - `true` (default): Returns the assistant's message content or the value of `:field` if specified.
      - `false`: Returns the full Ollama API response map.
    - `:field` (string): Specifies a field to extract from the response.
      - If `short: true` (default): Extracts the field from the `message` object (e.g., `"role"`, `"content"`).
        If the field is not found in `message`, it attempts to find it at the top level of the response.
      - If `short: false`: Extracts the field from the top-level API response (e.g., `"model"`, `"total_duration"`).
    - Other options are passed to the Ollama API (e.g., `temperature`, `max_tokens`).

  ## Returns

  - `{:ok, response}`: The formatted response based on `:short` and `:field` options.
  - `{:error, %Error{}}`: An error with a reason (e.g., API failure, invalid response format).

  ## Examples

      # Send a message and get the response content (default behavior)
      iex> AiFlow.Ollama.Chat.chat("Hello!", "chat123", "usr1")
      {:ok, "Hi there! How can I help you today?"}

      # Send a message with debug logging
      iex> AiFlow.Ollama.Chat.chat("What's the weather?", "chat123", "usr1", debug: true)
      12:00:00.000 [debug] Sending request to http://localhost:11434/api/chat
      12:00:00.000 [debug] Request body: %{"messages" => [%{"content" => "What's the weather?", "role" => "user"}], "model" => "llama3.1", "stream" => false}
      12:00:00.000 [debug] Ollama chat response: Status=200, Body=%{"model" => "llama3.1", "message" => %{"role" => "assistant", "content" => "It's sunny!"}, "done" => true}
      {:ok, "It's sunny!"}

      # Get the full API response
      iex> AiFlow.Ollama.Chat.chat("Hello!", "chat123", "usr1", short: false)
      {:ok, %{"model" => "llama3.1", "message" => %{"role" => "assistant", "content" => "Hi there!"}, "done" => true, "total_duration" => 123456789}}

      # Extract a specific top-level field from the full API response
      iex> AiFlow.Ollama.Chat.chat("Hello!", "chat123", "usr1", short: false, field: "model")
      {:ok, "llama3.1"}

      # Extract a specific field from the message object (default short: true)
      iex> AiFlow.Ollama.Chat.chat("Hello!", "chat123", "usr1", field: "role")
      {:ok, "assistant"}

      # Handle an API error
      iex> AiFlow.Ollama.Chat.chat("Hi", "chat123", "usr1")
      {:error, %AiFlow.Ollama.Error{type: :unknown, reason: "Connection timeout"}}
  """
  @spec chat(String.t(), String.t(), String.t(), keyword()) :: {:ok, term()} | {:error, Error.t()}
  def chat(prompt, chat_id, user_id \\ "default_user", opts \\ []) do
    config = AiFlow.Ollama.get_config()
    debug = Keyword.get(opts, :debug, false)
    model = Keyword.get(opts, :model, config.model)

    case load_chat_data(config) do
      {:ok, chat_data} ->
        chat_context = %{
          config: config,
          debug: debug,
          model: model,
          url: Config.build_url(config, "/api/chat"),
          chat_data: chat_data,
          prompt: prompt,
          chat_id: chat_id,
          user_id: user_id,
          opts: opts
        }
        handle_chat_logic(chat_context)

      {:error, reason} ->
        Logger.error("Failed to load chat data: #{inspect(reason)}")
        {:error, %Error{type: :unknown, reason: reason}}
    end
  end

  @doc """
  Same as `chat/4`, but raises a `RuntimeError` if the request fails instead of returning an error tuple.

  ## Parameters
  - Same as `chat/4`.

  ## Returns
  - The formatted response based on `:short` and `:field` options.

  ## Raises
  - `RuntimeError` if the chat request fails or the chat data cannot be loaded/saved.

  ## Examples

      # Successful chat
      iex> AiFlow.Ollama.Chat.chat!("Hello!", "chat123", "usr1")
      "Hi there!"

      # Raises on error
      iex> AiFlow.Ollama.Chat.chat!("Hi", "chat123", "usr1")
      ** (RuntimeError) Chat request failed: %AiFlow.Ollama.Error{type: :unknown, reason: "Connection timeout"}
  """
  @spec chat!(String.t(), String.t(), String.t(), keyword()) :: term()
  def chat!(prompt, chat_id, user_id \\ "default_user", opts \\ []) do
    case chat(prompt, chat_id, user_id, opts) do
      {:ok, result} -> result
      {:error, error} -> raise RuntimeError, message: "Chat request failed: #{inspect(error)}"
    end
  end

  # --- Private Helper Functions ---

  # Centralized logic for the chat interaction
  defp handle_chat_logic(chat_context) do
    {user_chats, chat_history, user_message, messages} =
      prepare_chat_data(
        chat_context.prompt,
        chat_context.chat_id,
        chat_context.user_id,
        chat_context.model,
        chat_context.chat_data
      )

    formatted_messages = Enum.map(messages, &Map.take(&1, ["role", "content"]))

    body = prepare_request_body(chat_context, formatted_messages)

    log_request(chat_context, body)

    http_response =
      HTTPClient.request(
        :post,
        chat_context.url,
        body,
        chat_context.config.timeout,
        chat_context.debug,
        0,
        :chat
      )

    case handle_chat_response(http_response, chat_context) do
      {:ok, %Req.Response{status: 200, body: api_response_body}} ->
        process_successful_response(chat_context, api_response_body, user_chats, chat_history, messages)

      error_tuple ->
        error_tuple
    end
  end

  # Prepares the HTTP request body by merging model, messages, and user-provided options
  defp prepare_request_body(chat_context, formatted_messages) do
    request_opts =
      chat_context.opts
      |> Keyword.delete(:debug)
      |> Keyword.delete(:short)
      |> Keyword.delete(:field)
      |> Enum.into(%{})

    Map.merge(
      %{
        "model" => chat_context.model,
        "messages" => formatted_messages,
        "stream" => false
      },
      request_opts
    )
  end

  # Logs the outgoing request details if debug mode is enabled
  defp log_request(chat_context, body) do
    if chat_context.debug do
      Logger.debug("Sending request to #{chat_context.url}")
      Logger.debug("Request body: #{inspect(body)}")
    end
  end

  # Processes a successful API response, updates chat history, and formats the user response
  defp process_successful_response(chat_context, api_response_body, user_chats, chat_history, messages) do
    short = Keyword.get(chat_context.opts, :short, true)
    field_opt = Keyword.get(chat_context.opts, :field)

    content_for_history = get_in(api_response_body, ["message", "content"]) || ""

    user_response = determine_user_response(api_response_body, short, field_opt)

    assistant_message = %{
      "role" => "assistant",
      "content" => content_for_history,
      "timestamp" => DateTime.utc_now() |> DateTime.to_iso8601()
    }

    updated_messages = messages ++ [assistant_message]

    updated_chat =
      Map.merge(chat_history, %{
        "messages" => updated_messages,
        "updated_at" => DateTime.utc_now() |> DateTime.to_iso8601()
      })

    updated_user_chats = Map.put(user_chats, chat_context.chat_id, updated_chat)

    updated_chat_data =
      Map.put(
        chat_context.chat_data,
        "chats",
        Map.put(
          Map.get(chat_context.chat_data, "chats", %{}),
          chat_context.user_id,
          updated_user_chats
        )
      )

    case save_chat_data(chat_context.config, updated_chat_data) do
      :ok -> {:ok, user_response}
      {:error, reason} ->
        Logger.error("Failed to save chat  #{inspect(reason)}")
        {:ok, user_response}
    end
  end

  # Determines the final response returned to the user based on :short and :field options
  defp determine_user_response(api_response_body, short, field_opt) do
    case {short, field_opt} do
      {true, nil} ->
        get_in(api_response_body, ["message", "content"])

      {true, field_name} ->
        case Map.get(api_response_body, field_name) do
          nil -> get_in(api_response_body, ["message", field_name])
          value -> value
        end

      {false, nil} ->
        api_response_body

      {false, field_name} ->
        Map.get(api_response_body, field_name)
    end
  end

  # Handles 404 "model not found" errors by attempting to pull the model
  defp handle_chat_response(
         {:ok, %Req.Response{status: 404, body: %{"error" => error_message}}},
         chat_context
       )
       when is_binary(error_message) do
    if error_message =~ "not found" and error_message =~ "try pulling it first" do
      model = chat_context.model
      debug = chat_context.debug

      Logger.info("Model '#{model}' not found, attempting to pull...")

      case Model.pull_model(model, debug: debug) do
        {:ok, _} ->
          Logger.info("Model '#{model}' pulled successfully, retrying chat request")
          retry_chat_request(chat_context)

        {:error, pull_error} ->
          Logger.error("Failed to pull model '#{model}': #{inspect(pull_error)}")
          {:error, pull_error}
      end
    else
      {:ok, %Req.Response{status: 404, body: %{"error" => error_message}}}
    end
  end

  defp handle_chat_response(response, _chat_context) do
    response
  end

  # Retries the original chat request after a successful model pull
  defp retry_chat_request(chat_context) do
    request_opts =
      chat_context.opts
      |> Keyword.delete(:debug)
      |> Keyword.delete(:short)
      |> Keyword.delete(:field)
      |> Enum.into(%{})

    formatted_messages =
      ((Map.get(chat_context.chat_data, "chats", %{}) |> Map.get(chat_context.user_id, %{}) |> Map.get(chat_context.chat_id, %{"messages" => []}))["messages"] || []) ++
        [%{"role" => "user", "content" => chat_context.prompt}]

    body =
      Map.merge(
        %{
          "model" => chat_context.model,
          "messages" => Enum.map(formatted_messages, &Map.take(&1, ["role", "content"])),
          "stream" => false
        },
        request_opts
      )

    HTTPClient.request(
      :post,
      chat_context.url,
      body,
      chat_context.config.timeout,
      chat_context.debug,
      0,
      :chat
    )
  end

  # Prepares chat data structures (user chats, chat history, current message) for the interaction
  defp prepare_chat_data(prompt, chat_id, user_id, model, chat_data) do
    user_chats = Map.get(chat_data, "chats", %{}) |> Map.get(user_id, %{})

    chat_history =
      Map.get(user_chats, chat_id, %{
        "name" => chat_id,
        "model" => model,
        "messages" => [],
        "created_at" => DateTime.utc_now() |> DateTime.to_iso8601(),
        "updated_at" => DateTime.utc_now() |> DateTime.to_iso8601()
      })

    user_message = %{
      "role" => "user",
      "content" => prompt,
      "timestamp" => DateTime.utc_now() |> DateTime.to_iso8601()
    }

    messages = (chat_history["messages"] || []) ++ [user_message]

    {user_chats, chat_history, user_message, messages}
  end

  # --- Chat History Management Functions ---

  @doc """
  Retrieves the chat history for a specific chat and user.

  ## Parameters
  - `chat_id`: The ID of the chat.
  - `user_id`: The ID of the user (defaults to `"default_user"`).

  ## Returns
  - `{:ok, list()}`: A list of message maps in the chat history.
  - `{:error, term()}`: An error if the chat data could not be loaded.
  """
  @spec show_chat_history(String.t(), String.t()) :: {:ok, list()} | {:error, term()}
  def show_chat_history(chat_id, user_id \\ "default_user") do
    config = AiFlow.Ollama.get_config()
    case load_chat_data(config) do
      {:ok, chat_data} ->
        chats = Map.get(chat_data, "chats", %{})
        user_chats = Map.get(chats, user_id, %{})
        chat = Map.get(user_chats, chat_id, %{"messages" => []})
        messages = Map.get(chat, "messages", [])
        {:ok, messages}
      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Retrieves the chat history for a specific chat and user, raising on error.

  ## Parameters
  - `chat_id`: The ID of the chat.
  - `user_id`: The ID of the user (defaults to `"default_user"`).

  ## Returns
  - `list()`: A list of message maps in the chat history.

  ## Raises
  - `RuntimeError` if the chat data could not be loaded.
  """
  @spec show_chat_history!(String.t(), String.t()) :: list()
  def show_chat_history!(chat_id, user_id \\ "default_user") do
    case show_chat_history(chat_id, user_id) do
      {:ok, messages} -> messages
      {:error, reason} -> raise RuntimeError, message: "Failed to show chat history: #{inspect(reason)}"
    end
  end

  @doc """
  Retrieves all stored chat data.

  ## Returns
  - `{:ok, map()}`: The entire chat data map with atom keys.
  - `{:error, term()}`: An error if the chat data could not be loaded.
  """
  @spec show_all_chats() :: {:ok, map()} | {:error, term()}
  def show_all_chats do
    config = AiFlow.Ollama.get_config()
    case load_chat_data(config) do
      {:ok, chat_data} -> {:ok, atomize_keys(chat_data)}
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Retrieves all stored chat data, raising on error.

  ## Returns
  - `map()`: The entire chat data map with atom keys.

  ## Raises
  - `RuntimeError` if the chat data could not be loaded.
  """
  @spec show_all_chats!() :: map()
  def show_all_chats! do
    case show_all_chats() do
      {:ok, data} -> data
      {:error, reason} -> raise RuntimeError, message: "Failed to show all chats: #{inspect(reason)}"
    end
  end

  @doc """
  Retrieves the raw content of the chat data file.

  ## Returns
  - `{:ok, map()}`: The raw chat data map.
  - `{:error, term()}`: An error if the chat data could not be loaded.
  """
  @spec show_chat_file_content() :: {:ok, map()} | {:error, term()}
  def show_chat_file_content do
    config = AiFlow.Ollama.get_config()
    load_chat_data(config)
  end

  @doc """
  Retrieves the raw content of the chat data file, raising on error.

  ## Returns
  - `map()`: The raw chat data map.

  ## Raises
  - `RuntimeError` if the chat data could not be loaded.
  """
  @spec show_chat_file_content!() :: map()
  def show_chat_file_content! do
    case show_chat_file_content() do
      {:ok, data} -> data
      {:error, reason} -> raise RuntimeError, message: "Failed to show chat file content: #{inspect(reason)}"
    end
  end

  @doc """
  Clears chat history based on provided options.

  ## Options
  - `:confirm` (boolean): Must be `true` to proceed with deletion (defaults to `false`).
  - `:user` (string): The user ID whose chats to delete (optional).
  - `:chat` (string): The specific chat ID to delete (requires `:user`, optional).

  ## Returns
  - `{:ok, :deleted}`: Confirmation of deletion.
  - `{:error, term()}`: An error if deletion failed or confirmation was not given.

  ## Examples

      # Delete user chat
      AiFlow.Ollama.Chat.chat(chat: "my_chat", user: "user1")
      :ok
  """
  @spec clear_chat_history(keyword()) :: {:ok, :deleted} | {:error, term()}
  def clear_chat_history(opts \\ []) do
    config = AiFlow.Ollama.get_config()
    confirm = Keyword.get(opts, :confirm, false)
    user_id = Keyword.get(opts, :user, nil)
    chat_id = Keyword.get(opts, :chat, nil)

    cond do
      chat_id == nil and user_id == nil ->
        delete_all_chats(config, confirm)
      chat_id == nil and user_id != nil ->
        delete_user_chats(config, user_id, confirm)
      true ->
        delete_specific_chat(config, user_id, chat_id)
    end
  end

  # --- Private Chat History Helpers ---

  # Deletes all chat data after confirmation
  @doc false
  @spec delete_all_chats(map(), boolean()) :: {:ok, :deleted} | {:error, term()}
  defp delete_all_chats(config, confirm) do
    if confirm do
      with {:ok, _chat_data} <- load_chat_data(config) do
        updated_chat_data = %{
          "chats" => %{},
          "created_at" => DateTime.utc_now() |> DateTime.to_iso8601()
        }
        save_chat_data(config, updated_chat_data)
      end
    else
      {:error, "Please confirm deletion of all chats by passing 'confirm: true' as an option."}
    end
  end

  # Deletes all chats for a specific user after confirmation
  @doc false
  @spec delete_user_chats(map(), String.t(), boolean()) :: {:ok, :deleted} | {:error, term()}
  defp delete_user_chats(config, user_id, confirm) do
    if confirm do
      with {:ok, chat_data} <- load_chat_data(config) do
        updated_chats = Map.delete(Map.get(chat_data, "chats", %{}), user_id)
        updated_chat_data = %{
          "chats" => updated_chats,
          "created_at" => Map.get(chat_data, "created_at")
        }
        save_chat_data(config, updated_chat_data)
      end
    else
      {:error, "Please confirm deletion of all chats for user '#{user_id}' by passing 'confirm: true' as an option, or specify a chat ID with 'chat: chat_id' to delete a specific chat."}
    end
  end

  # Deletes a specific chat for a specific user
  @doc false
  @spec delete_specific_chat(map(), String.t() | nil, String.t()) :: {:ok, :deleted} | {:error, term()}
  defp delete_specific_chat(config, user_id, chat_id) do
    with {:ok, chat_data} <- load_chat_data(config),
         user_chats when is_map(user_chats) <- Map.get(chat_data, "chats", %{}) |> Map.get(user_id || "default_user", %{}),
         true <- Map.has_key?(user_chats, chat_id) do
      updated_user_chats = Map.delete(user_chats, chat_id)
      updated_chats = Map.put(Map.get(chat_data, "chats", %{}), user_id || "default_user", updated_user_chats)
      updated_chat_data = %{
        "chats" => updated_chats,
        "created_at" => Map.get(chat_data, "created_at")
      }
      save_chat_data(config, updated_chat_data)
    else
      {:error, reason} -> {:error, reason}
      _ -> {:ok, :deleted}
    end
  end

  # --- Debugging Functions ---

  # Loads and logs chat data for debugging purposes
  @doc false
  @spec debug_load_chat_data() :: {:ok, map()} | {:error, term()}
  defp debug_load_chat_data do
    config = AiFlow.Ollama.get_config()
    case load_chat_data(config) do
      {:ok, data} when is_map(data) ->
        atomized = atomize_keys(data)
        Logger.debug("Debug chat data (atom keys): #{inspect(atomized)}")
        {:ok, atomized}
      {:ok, data} ->
        Logger.debug("Debug chat data (other): #{inspect(data)}")
        {:ok, data}
      {:error, reason} ->
        Logger.error("Error loading chat data: #{inspect(reason)}")
        {:error, reason}
    end
  end

  # Shows chat history and logs it for debugging purposes
  @doc false
  @spec debug_show_chat_history(String.t(), String.t()) :: list()
  defp debug_show_chat_history(chat_id, user_id \\ "default_user") do
    config = AiFlow.Ollama.get_config()
    case load_chat_data(config) do
      {:ok, chat_data} ->
        chats = Map.get(chat_data, "chats", %{})
        user_chats = Map.get(chats, user_id, %{})
        chat = Map.get(user_chats, chat_id, %{"messages" => []})
        messages = Map.get(chat, "messages", [])
        Logger.debug("Debug show chat history messages: #{inspect(messages)}")
        messages
      {:error, reason} ->
        Logger.error("Failed to load chat data for debug: #{inspect(reason)}")
        []
    end
  end

  @doc """
  Checks the integrity and format of the chat data file.

  ## Returns
  - `{:ok, map()}`: The parsed chat data if the file is valid.
  - `{:error, term()}`: An error if the file is missing or malformed.
  """
  @spec check_chat_file() :: {:ok, map()} | {:error, term()}
  def check_chat_file do
    config = AiFlow.Ollama.get_config()
    case @file_module.read(config.chat_file) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} ->
            if Map.has_key?(data, "chats"), do: {:ok, data}, else: {:error, :invalid_format}
          {:error, reason} -> {:error, reason}
        end
      {:error, :enoent} -> {:error, :file_not_found}
      {:error, reason} -> {:error, reason}
    end
  end

  # --- Core Data Persistence ---

  # Loads chat data from the file, initializing if necessary
  @doc false
  defp load_chat_data(config) do
    case @file_module.read(config.chat_file) do
      {:ok, json} ->
        case Jason.decode(json) do
          {:ok, data} -> {:ok, data}
          {:error, reason} -> {:error, reason}
        end
      {:error, :enoent} ->
        initial_data = %{
          "chats" => %{},
          "created_at" => DateTime.utc_now() |> DateTime.to_iso8601()
        }
        case save_chat_data(config, initial_data) do
          :ok -> {:ok, initial_data}
          {:error, reason} -> {:error, reason}
        end
      {:error, reason} -> {:error, reason}
    end
  end

  # Saves chat data to the file
  @doc false
  defp save_chat_data(config, data) do
    try do
      json_data = Jason.encode!(data, pretty: true)
      @file_module.write!(config.chat_file, json_data)
      :ok
    rescue
      error ->
        Logger.error("Failed to save chat data: #{inspect(error)}")
        {:error, error}
    end
  end

  # Converts string keys in a map to atoms (for debugging)
  @doc false
  defp atomize_keys(map) when is_map(map) do
    Enum.into(map, %{}, fn {k, v} -> {if(is_binary(k), do: String.to_atom(k), else: k), v} end)
  end
end
