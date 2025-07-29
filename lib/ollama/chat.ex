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

  alias AiFlow.Ollama.{Config, Error, HTTPClient}

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
      iex> AiFlow.Ollama.Chat.chat("Hello!", "chat123", "usr1", field: {:body, "model"})
      {:ok, "llama3.1"}

      # Extract a specific field from the message object (default short: true)
      iex> AiFlow.Ollama.Chat.chat("Hello!", "chat123", "usr1", field: {:body, ["message", "role"]})
      {:ok, "assistant"}

      # Handle an API error
      iex> AiFlow.Ollama.Chat.chat("Hi", "chat123", "usr1")
      {:error, %AiFlow.Ollama.Error{type: :unknown, reason: "Connection timeout"}}
  """
  @spec chat(String.t(), String.t(), String.t(), keyword()) :: {:ok, term()} | {:error, Error.t()}
  def chat(prompt, chat_id, user_id \\ "default_user", opts \\ []) do
    config = AiFlow.Ollama.get_config()
    url = Config.build_url(config, "/api/chat")
    model = Keyword.get(opts, :retries, config.model)
    debug = Keyword.get(opts, :debug, false)
    short = Keyword.get(opts, :short, true)
    field = Keyword.get(opts, :field, {:body, "message", "response"})
    retries = Keyword.get(opts, :retries, 0)

    with {:ok, chat_data} <- load_chat_data(config) do
      chat_context = %{
        config: config,
        url: url,
        chat_data: chat_data,
        model: model,
        debug: debug,
        short: short,
        retries: retries,
        field: field,
        prompt: prompt,
        chat_id: chat_id,
        user_id: user_id,
        opts: opts
      }

      handle_chat_logic(chat_context)
    else
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
      {:error, error} -> error
      other -> other
    end
  end

  defp handle_chat_logic(chat_context) do
    chat_data = chat_context.chat_data
    user_id = chat_context.user_id
    chat_id = chat_context.chat_id
    model = chat_context.model
    prompt = chat_context.prompt

    user_chats = get_in(chat_data, ["chats", user_id]) || %{}
    chat_history = Map.get(user_chats, chat_id, %{
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
    formatted_messages = Enum.map(messages, &Map.take(&1, ["role", "content"]))

    request_body = %{
      "model" => model,
      "messages" => formatted_messages,
      "stream" => false
    }

    http_resp = HTTPClient.request(:post, chat_context.url, request_body, chat_context.config.timeout, chat_context.debug, 0, :chat)

    case http_resp do
      {:ok, %Req.Response{status: 200, body: api_response_body}} ->
        process_successful_response(chat_context, api_response_body, user_chats, chat_history, messages)
        HTTPClient.handle_response(http_resp, chat_context.field, :chat, chat_context.opts)

      {:error, error} ->
        {:error, error}

      {:ok, %Req.Response{status: status}} ->
        Logger.error("Unexpected status #{status} in chat response")
        {:error, %Error{type: :http, reason: "Unexpected status #{status}", status: status}}
    end
  end

  defp process_successful_response(chat_context, api_response_body, user_chats, chat_history, messages) do
    assistant_content = get_in(api_response_body, ["message", "content"]) || ""

    user_response = determine_user_response(api_response_body, chat_context.short, chat_context.field)

    assistant_message = %{
      "role" => "assistant",
      "content" => assistant_content,
      "timestamp" => DateTime.utc_now() |> DateTime.to_iso8601()
    }

    updated_messages = messages ++ [assistant_message]
    updated_chat = Map.merge(chat_history, %{
      "messages" => updated_messages,
      "updated_at" => DateTime.utc_now() |> DateTime.to_iso8601()
    })

    updated_user_chats = Map.put(user_chats, chat_context.chat_id, updated_chat)

    updated_chat_data =
      chat_context.chat_data
      |> Map.put("chats", Map.put(Map.get(chat_context.chat_data, "chats", %{}), chat_context.user_id, updated_user_chats))

    case save_chat_data(chat_context.config, updated_chat_data) do
      :ok ->
        {:ok, user_response}
      {:error, reason} ->
        Logger.error("Failed to save chat data: #{inspect(reason)}")
        {:ok, user_response}
    end
  end

  defp determine_user_response(api_response_body, short, field_opt) when is_map(api_response_body) do
    case {short, field_opt} do
      {true, {:body, path}} when is_binary(path) ->
        Map.get(api_response_body, path)
      {true, {:body, path}} when is_list(path) ->
        get_in(api_response_body, path)
      {true, nil} ->
        get_in(api_response_body, ["message", "content"])
      {true, field_name} ->
        case Map.get(api_response_body, field_name) do
          nil -> get_in(api_response_body, ["message", field_name])
          value -> value
        end
      {false, nil} ->
        api_response_body
      {false, field_name} when is_binary(field_name) ->
        Map.get(api_response_body, field_name)
      {false, {:body, path}} when is_binary(path) ->
        Map.get(api_response_body, path)
      {false, {:body, path}} when is_list(path) ->
        get_in(api_response_body, path)
      {_, _} ->
        api_response_body
    end
  end

  defp determine_user_response(_api_response_body, _short, _field_opt) do
    nil
  end

  # --- Chat History Management Functions ---

  @doc """
  Retrieves the chat history or list of chats for a specific user.

  ## Parameters

  - `opts`: Keyword list of options:
    - `user_id`: The ID of the user (string).
    - `chat_id`: The ID of a specific chat to retrieve (string, optional).
    - `:short` (boolean): Controls the response format.
      - `true` (default): Returns a list of chat names/IDs for the user (if `chat_id` is nil),
        or the content (list of messages) of the specific chat (if `chat_id` is provided).
      - `false`: Returns the full map(s) of chat data.
        If `chat_id` is nil, returns a map of all user's chats.
        If `chat_id` is provided, returns the full map of that specific chat.

  ## Returns

  - `{:ok, term()}`: The requested data based on `user_id`, `chat_id`, and `:short` option.
  - `{:error, term()}`: An error if the chat data could not be loaded or chat is not found.

  ## Examples

      # Get all users in short: true version
      iex> AiFlow.Ollama.show_chat_history()
      {:ok, %{users: ["default_user"]}}


      # Get all users in short: false version
      iex> AiFlow.Ollama.show_chat_history()
      {:ok, %{users: ["default_user"]}}
      iex(41)> AiFlow.Ollama.show_chat_history(short: false)
      {:ok, %{"chats" => %{"default_user" => %{...}}2

      # Get list of chat names for a user (short: true by default)
      iex> AiFlow.Ollama.Chat.show_chat_history(user_id: "user1")
      {:ok, ["chat1", "chat2"]}

      # Get full map of all chats for a user
      iex> AiFlow.Ollama.Chat.show_chat_history(user_id: "user1", short: false)
      {:ok, %{"chat1" => %{"name" => "chat1", "messages" => [...], ...}, "chat2" => %{...}}}

      # Get messages list for a specific chat (short: true by default)
      iex> AiFlow.Ollama.Chat.show_chat_history(user_id: "user1", chat_id: "chat1")
      {:ok, [%{"role" => "user", "content" => "..."}, %{"role" => "assistant", "content" => "..."}]}

      # Get full map for a specific chat
      iex> AiFlow.Ollama.Chat.show_chat_history(user_id: "user1", chat_id: "chat1", short: false)
      {:ok, %{"name" => "chat1", "messages" => [...], "created_at" => "...", ...}}

      # Handle user not found or no chats
      iex> AiFlow.Ollama.Chat.show_chat_history(user_id: "unknown_user")
      {:ok, []}

      # Handle specific chat not found
      iex> AiFlow.Ollama.Chat.show_chat_history(user_id: "user1", chat_id: "unknown_chat")
      {:error, :chat_not_found}
  """
  @spec show_chat_history(keyword()) :: {:ok, term()} | {:error, term()}
  def show_chat_history(opts \\ []) do
    config = AiFlow.Ollama.get_config()
    user_id = Keyword.get(opts, :user_id, nil)
    chat_id = Keyword.get(opts, :chat_id, nil)
    short = Keyword.get(opts, :short, true)

    with {:ok, chat_data} <- load_chat_data(config) do
      case {user_id, chat_id} do
        {nil, nil} ->
          {:ok, (if short, do: %{users: Map.keys(chat_data["chats"])}, else: chat_data)}

        {user_id, nil} ->
          user_chats_map = get_in(chat_data, ["chats"]) || %{}
          user_chats = Map.get(user_chats_map, user_id, %{})
          {:ok, (if short, do: Map.keys(user_chats), else: user_chats)}

        {user_id, chat_id} ->
          user_chats_map = get_in(chat_data, ["chats"]) || %{}
          user_chats = Map.get(user_chats_map, user_id, %{})
          chat_data = Map.get(user_chats, chat_id)

          if not is_nil(chat_data) do
            {:ok, (if short, do: Map.get(chat_data, "messages", []), else: chat_data)}
          else
            {:error, :chat_not_found}
          end
      end
    end
  end

  @doc """
  Retrieves the chat history or list of chats for a specific user, raising on error.

  ## Parameters

  - `opts`: Keyword list of options:
    - `user_id`: The ID of the user (string).
    - `chat_id`: The ID of a specific chat to retrieve (string, optional).
    - `:short` (boolean): Controls the response format.
      - `true` (default): Returns a list of chat names/IDs for the user (if `chat_id` is nil),
        or the content (list of messages) of the specific chat (if `chat_id` is provided).
      - `false`: Returns the full map(s) of chat data.
        If `chat_id` is nil, returns a map of all user's chats.
        If `chat_id` is provided, returns the full map of that specific chat.

  ## Returns

  - `term()`: The requested data based on `user_id`, `chat_id`, and `:short` option.

  ## Raises

  - `RuntimeError` if the chat data could not be loaded or chat is not found.

  ## Examples

      # Get list of chat names for a user
      iex> AiFlow.Ollama.Chat.show_chat_history!(user_id: "user1")
      ["chat1", "chat2"]

      # Get messages list for a specific chat
      iex> AiFlow.Ollama.Chat.show_chat_history!(user_id: "user1", chat_id: "chat1")
      [%{"role" => "user", "content" => "..."}, ...]

      # Raises on error
      iex> AiFlow.Ollama.Chat.show_chat_history!(user_id: "unknown_user", chat_id: "unknown_chat")
      ** (RuntimeError) Failed to show chat history: :chat_not_found
  """
  @spec show_chat_history!(keyword()) :: term()
  def show_chat_history!(opts \\ []) do
    case show_chat_history(opts) do
      {:ok, result} -> result
      {:error, error} -> error
      other -> other
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

    with {:ok, chat_data} <- load_chat_data(config) do
      {:ok, atomize_keys(chat_data)}
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
      {:ok, result} -> result
      {:error, error} -> error
      other -> other
    end
  end

  @doc """
  Clears chat history based on provided options.

  ## Options

  - `:confirm` (boolean): Must be `true` to proceed with deletion (defaults to `false`).
  - `:user_id` (string): The user ID whose chats to delete (optional).
  - `:chat_id` (string): The specific chat ID to delete (requires `:user`, optional).

  ## Returns

  - `{:ok, :success}`: Confirmation of deletion.
  - `{:ok, :deleted}`: If selected chat has already been deleted
  - `{:error, term()}`: An error if deletion failed or confirmation was not given.

  ## Examples

      # Delete user chat
      AiFlow.Ollama.Chat.chat(chat_id: "my_chat", user_id: "user1")
      {:ok, :success}

      # If the selected chat has already been deleted
      AiFlow.Ollama.Chat.chat(chat_id: "my_chat", user_id: "user1")
      {:ok, :deleted}
  """
  @spec clear_chat_history(keyword()) :: {:ok, :deleted} | {:error, term()}
  def clear_chat_history(opts \\ []) do
    config = AiFlow.Ollama.get_config()
    confirm = Keyword.get(opts, :confirm, false)
    user_id = Keyword.get(opts, :user_id, nil)
    chat_id = Keyword.get(opts, :chat_id, nil)

    case {user_id, chat_id} do
      {nil, nil} ->
        delete_all_chats(config, confirm)
      {user_id, nil} ->
        delete_user_chats(config, user_id, confirm)
      {user_id, chat_id} ->
        delete_specific_chat(config, user_id, chat_id)
      end
  end

  @doc """
  Clears chat history based on provided options.

  ## Options

  - `:confirm` (boolean): Must be `true` to proceed with deletion (defaults to `false`).
  - `:user_id` (string): The user ID whose chats to delete (optional).
  - `:chat_id` (string): The specific chat ID to delete (requires `:user`, optional).

  ## Returns

  - `{:ok, :success}`: Confirmation of deletion.
  - `{:ok, :deleted}`: If selected chat has already been deleted
  - `{:error, term()}`: An error if deletion failed or confirmation was not given.

  ## Examples

      # Delete user chat
      AiFlow.Ollama.Chat.chat!(chat_id: "my_chat", user_id: "user1")
      :success

      # If the selected chat has already been deleted
      AiFlow.Ollama.Chat.chat(chat_id: "my_chat", user_id: "user1")
      :deleted
  """
  @spec clear_chat_history(keyword()) :: {:ok, :deleted} | {:error, term()}
  def clear_chat_history!(opts \\ []) do
    case clear_chat_history(opts) do
      {:ok, result} -> result
      {:error, error} -> error
      other -> other
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
        {:ok, :success}
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
        {:ok, :success}
      end
    else
      {:error,
       "Please confirm deletion of all chats for user '#{user_id}' by passing 'confirm: true' as an option, or specify a chat ID with 'chat: chat_id' to delete a specific chat."}
    end
  end

  # Deletes a specific chat for a specific user
  @doc false
  @spec delete_specific_chat(map(), String.t() | nil, String.t()) :: {:ok, :deleted} | {:error, term()}
  defp delete_specific_chat(config, user_id, chat_id) do
    with {:ok, chat_data} <- load_chat_data(config),
         user_chats when is_map(user_chats) <-
           get_in(chat_data, ["chats", user_id || "default_user"]) || %{},
         true <- Map.has_key?(user_chats, chat_id) do
      updated_user_chats = Map.delete(user_chats, chat_id)
      updated_chats =
        Map.put(Map.get(chat_data, "chats", %{}), user_id || "default_user", updated_user_chats)

      updated_chat_data = %{
        "chats" => updated_chats,
        "created_at" => Map.get(chat_data, "created_at")
      }

      save_chat_data(config, updated_chat_data)
      {:ok, :success}
    else
      {:error, reason} -> {:error, reason}
      _ -> {:ok, :deleted}
    end
  end

  @doc """
  Loads the raw chat data from the configured file and converts its keys to atoms for easier debugging.

  This function is intended for debugging purposes. It reads the chat data file specified in the
  configuration, parses the JSON content, and then recursively converts all string keys in the
  resulting map to atom keys. This can make the data structure easier to inspect and work with
  in an interactive debugging session (e.g., in `iex`).

  ## Returns

  - `{:ok, map()}`: A map representing the chat data with atom keys.
  - `{:error, term()}`: An error reason if the file could not be read or parsed.

  ## Examples

      # In a successful case, loads and atomizes the chat data
      iex> AiFlow.Ollama.Chat.debug_load_chat_data()
      {:ok,
      %{
        chats: %{
          "default_user" => %{
            "my_chat" => %{
              "created_at" => "2023-10-27T10:00:00Z",
              "messages" => [
                %{"content" => "Hello", "role" => "user", "timestamp" => "2023-10-27T10:00:01Z"},
                %{"content" => "Hi there!", "role" => "assistant", "timestamp" => "2023-10-27T10:00:02Z"}
              ],
              "model" => "llama3.1",
              "name" => "my_chat",
              "updated_at" => "2023-10-27T10:00:02Z"
            }
          }
        },
        created_at: "2023-10-27T10:00:00Z"
      }}

      # In an error case (e.g., file not found or invalid JSON)
      iex> # Assuming the chat file is corrupted
      iex> AiFlow.Ollama.Chat.debug_load_chat_data()
      {:error, %Jason.DecodeError{data: "<<", position: 0, token: nil}}
  """
  @spec debug_load_chat_data() :: {:ok, map()} | {:error, term()}
  def debug_load_chat_data do
    config = AiFlow.Ollama.get_config()

    case load_chat_data(config) do
      {:ok, data} when is_map(data) ->
        atomized = atomize_keys(data)
        {:ok, atomized}

      {:ok, data} ->
        {:ok, data}

      {:error, reason} ->
        Logger.error("Error loading chat data: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Retrieves and returns the message history for a specific chat and user, intended for debugging purposes.

  This function directly fetches the list of messages associated with a given chat ID and user ID
  from the chat data file. It's a simplified way to inspect the raw message history without any
  additional processing or error wrapping, making it useful for debugging chat state or history
  issues.

  ## Parameters

  - `chat_id`: The unique identifier of the chat session (string).
  - `user_id`: The identifier of the user (string, defaults to `"default_user"`).

  ## Returns

  - `list()`: A list of message maps (e.g., `[ %{"role" => "user", "content" => "...", ...}, ...]`)
    belonging to the specified chat and user. Returns an empty list `[]` if the chat/user is not
    found or if there's an error loading the chat data.

  ## Examples

      # Retrieve messages for an existing chat
      iex> AiFlow.Ollama.Chat.debug_show_chat_history("my_chat", "user1")
      [
        %{"role" => "user", "content" => "Hello!", "timestamp" => "2023-10-27T10:00:00Z"},
        %{"role" => "assistant", "content" => "Hi there!", "timestamp" => "2023-10-27T10:00:01Z"}
      ]

      # Retrieve messages for a non-existent chat (returns empty list)
      iex> AiFlow.Ollama.Chat.debug_show_chat_history("unknown_chat", "user1")
      []

      # Retrieve messages using the default user ID
      iex> AiFlow.Ollama.Chat.debug_show_chat_history("default_chat")
      [%{"role" => "user", "content" => "Default user message", ...}]

      # If there's an error loading the chat file (e.g., corrupt JSON), it logs the error and returns []
      # (Assuming the chat file is corrupted)
      iex> AiFlow.Ollama.Chat.debug_show_chat_history("any_chat")
      # 12:00:00.000 [error] Failed to load chat data for debug: %Jason.DecodeError{...}
      []
  """
  @spec debug_show_chat_history(String.t(), String.t()) :: list()
  def debug_show_chat_history(chat_id, user_id \\ "default_user") do
    config = AiFlow.Ollama.get_config()

    case load_chat_data(config) do
      {:ok, chat_data} ->
        messages =
          get_in(chat_data, ["chats", user_id, chat_id, "messages"]) || []

        messages

      {:error, reason} ->
        Logger.error("Failed to load chat data for debug: #{inspect(reason)}")
        []
    end
  end

  @doc """
  Checks the integrity and format of the chat data file {It will be deprecated later =)}

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

          {:error, reason} ->
            {:error, reason}
        end

      {:error, :enoent} ->
        {:error, :file_not_found}

      {:error, reason} ->
        {:error, reason}
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

      {:error, reason} ->
        {:error, reason}
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
    Enum.into(map, %{}, fn {k, v} ->
      {if(is_binary(k), do: String.to_atom(k), else: k), v}
    end)
  end
end
