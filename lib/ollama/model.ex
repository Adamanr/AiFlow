defmodule AiFlow.Ollama.Model do
  @moduledoc """
  Manages model-related operations for the Ollama API.

  This module provides functions to interact with Ollama models, including listing,
  showing details, pulling, pushing, copying, deleting, creating, and loading models.
  It leverages the shared `AiFlow.Ollama.HTTPClient` for consistent HTTP handling,
  error management, and support for options like `:short`, `:field`, `:debug`, and `:retries`.

  ## Features

  - List local and running models.
  - Show detailed information about a specific model.
  - Pull models from remote registries.
  - Push models to remote registries.
  - Copy models locally.
  - Delete models.
  - Create new models from Modelfile.
  - Load a model into memory.
  - Support for standard (`{:ok, result} | {:error, reason}`) and raising (`!`) variants.
  - Configurable response formatting with `:short` and `:field` options (via `HTTPClient`).
  - Debug logging and retry mechanisms.

  ## Examples

      # List all local models
      {:ok, models} = AiFlow.Ollama.Model.list()

      # Get detailed info about a model
      {:ok, model_info} = AiFlow.Ollama.Model.show_model("llama3.1")

      # Pull a model with debug output
      {:ok, _} = AiFlow.Ollama.Model.pull_model("mistral", debug: true)

      # Delete a model
      {:ok, :success} = AiFlow.Ollama.Model.delete_model("llama3.1")

      # Use short format to get just model names from list
      {:ok, names} = AiFlow.Ollama.Model.list(short: true, field: "name")

      # Raise on error
      names = AiFlow.Ollama.Model.list!()
  """

  require Logger
  alias AiFlow.Ollama.{Config, Error, HTTPClient}

  # ETS tables for caching
  @models_cache :ai_flow_ollama_models_cache
  @model_info_cache :ai_flow_ollama_model_info_cache

  @doc """
  Lists available local models.

  Fetches a list of models that are currently available on the Ollama server.

  ## Parameters

  - `opts`: Keyword list of options:
    - `:debug` (boolean): Enables debug logging (default: `false`).
    - `:retries` (integer): Number of retry attempts for failed requests (default: `0`).
    - `:short` (boolean): If `true`, returns a list of model names only.
                          If `false`, returns the full list of model information maps.
                          (default: `true`).

  ## Returns

  - `{:ok, list(String.t())}`: A list of model names (e.g., `["llama3.1:latest", "mistral:latest"]`) when `:short` is `true`.
  - `{:ok, list(map())}`: A list of maps containing full model details when `:short` is `false`.
  - `{:error, Error.t()}`: An error struct if the request fails.

  ## Examples

      # Get only model names (default behavior with :short=true)
      {:ok, model_names} = AiFlow.Ollama.list_models()
      # model_names is ["llama3.1:latest", "mistral:latest", ...]

      # Get only model names explicitly
      {:ok, model_names} = AiFlow.Ollama.list_models(short: true)

      # Get full model details
      {:ok, model_details} = AiFlow.Ollama.list_models(short: false)
      # model_details is [%{"name" => "llama3.1:latest", "size" => ..., ...}, ...]

      # Enable debug logging
      {:ok, _} = AiFlow.Ollama.list_models(debug: true)
  """
  @spec list_models(keyword()) :: {:ok, list(String.t()) | list(map())} | {:error, Error.t()}
  def list_models(opts \\ []) do
    config = AiFlow.Ollama.get_config()
    debug = Keyword.get(opts, :debug, false)
    short = Keyword.get(opts, :short, true)
    retries = Keyword.get(opts, :retries, 0)

    url = Config.build_url(config, "/api/tags")

    http_response = HTTPClient.request(:get, url, nil, config.timeout, debug, retries, :list_models)

    case http_response do
      {:ok, %Req.Response{status: 200, body: %{"models" => models_list}}} when is_list(models_list) ->
        result = if short do
          Enum.map(models_list, fn model_info ->
            Map.get(model_info, "name", "")
          end)
        else
          models_list
        end
        {:ok, result}

      _ ->
        HTTPClient.handle_response(http_response, "models", :list_models, opts)
        |> case do
          {:ok, _} -> {:error, Error.unknown("Unexpected response format")}
          error_tuple -> error_tuple
        end
    end
  end

  @doc """
  Lists available local models, raising on error.

  Similar to `list/1`, but returns the result directly or raises an `AiFlow.Ollama.Error`.

  ## Parameters

  - `opts`: Keyword list of options (same as `list/1`).

  ## Returns

  - `list()` or `term()`: The list of models or processed response based on options.
  - Raises `AiFlow.Ollama.Error` on failure.

  ## Examples

      models = AiFlow.Ollama.Model.list!()
      model_names = AiFlow.Ollama.Model.list!(short: true, field: "name")
  """
  @spec list_models!(keyword()) :: list() | term()
  def list_models!(opts \\ []) do
    list_models(opts)
    |> HTTPClient.handle_result(:list_models!)
  end

  @doc """
  Shows detailed information about a specific model.

  Retrieves detailed configuration and metadata for a given model name.

  ## Parameters

  - `name`: The name of the model (string).
  - `opts`: Keyword list of options:
    - `:debug` (boolean): Enables debug logging (default: `false`).
    - `:retries` (integer): Number of retry attempts for failed requests (default: `0`).
    - `:short` (boolean): If `false`, returns the raw `%Req.Response{}`. (default: `true`).
    - `:field` (String.t()): When `:short` is `true`, specifies the field to extract from
                             the response body. (default: `"model"` - the entire response body is the model info).

  ## Returns

  - `{:ok, map()}`: A map containing detailed information about the model when `:short` is `true`.
  - `{:ok, term()}`: The processed response based on `:short` and `:field` options.
  - `{:error, Error.t()}`: An error struct if the request fails.

  ## Examples

      # Get full model details
      {:ok, model_info} = AiFlow.Ollama.Model.show_model("llama3.1")
      # model_info is %{"license" => ..., "modelfile" => ..., "parameters" => ..., "template" => ...}

      # Enable debug logging
      {:ok, _} = AiFlow.Ollama.Model.show_model("llama3.1", debug: true)

      # Get raw HTTP response
      {:ok, %Req.Response{status: 200, body: %{...}}} = AiFlow.Ollama.Model.show_model("llama3.1", short: false)

      # Extract a specific field from the model info (e.g., license)
      {:ok, license_text} = AiFlow.Ollama.Model.show_model("llama3.1", short: true, field: "license")
  """
  @spec show_model(String.t(), keyword()) :: {:ok, map() | term()} | {:error, Error.t()}
  def show_model(name, opts \\ []) do
    if is_binary(name) and name != "" do
      config = AiFlow.Ollama.get_config()
      debug = Keyword.get(opts, :debug, false)
      retries = Keyword.get(opts, :retries, 0)
      field = Keyword.get(opts, :field, nil)

      url = Config.build_url(config, "/api/show")
      body = %{name: name}

      HTTPClient.request(:post, url, body, config.timeout, debug, retries, :show_model)
      |> HTTPClient.handle_response(field, :show_model, opts)

    else
      {:error, Error.invalid("Model name must be a non-empty string")}
    end
  end

  @doc """
  Shows detailed information about a specific model, raising on error.

  Similar to `show_model/2`, but returns the result directly or raises an `AiFlow.Ollama.Error`.

  ## Parameters

  - `name`: The name of the model (string).
  - `opts`: Keyword list of options (same as `show_model/2`).

  ## Returns

  - `map()` or `term()`: The model information map or processed response based on options.
  - Raises `AiFlow.Ollama.Error` on failure.

  ## Examples

      model_info = AiFlow.Ollama.Model.show_model!("llama3.1")
      license_text = AiFlow.Ollama.Model.show_model!("llama3.1", short: true, field: "license")
  """
  @spec show_model!(String.t(), keyword()) :: map() | term()
  def show_model!(name, opts \\ []) do
    show_model(name, opts)
    |> HTTPClient.handle_result(:show_model)
    |> case do
      %Req.Response{} = resp -> resp
      body when is_map(body) -> body
      other -> other
    end
  end

  @doc """
  Pulls a model from a remote registry.

  Downloads a model from the Ollama library or another registry to the local machine.
  This function handles streaming responses from the API.

  ## Parameters

  - `name`: The name of the model to pull (string).
  - `opts`: Keyword list of options:
    - `:debug` (boolean): Enables debug logging (default: `false`).
    - `:retries` (integer): Number of retry attempts for failed requests (default: `0`).
    - `:stream` (boolean): Whether to request a streaming response (default: `true`).
                           Note: Streaming responses require special handling not fully covered
                           by the standard `HTTPClient.handle_response/4`. This function
                           might need specific logic for streaming.
    - `:short` (boolean): If `false`, attempts to return the raw `%Req.Response{}`.
                          Streaming responses make this complex. (default: `true`).
    - `:field` (String.t()): When `:short` is `true`, specifies the field to extract.
                             For streaming, this is usually not applicable. (default: `"status"`).

  ## Returns

  - `{:ok, term()}`: Typically `:success` or a map with status information when the pull completes.
                     For streaming, this might be the final parsed status message.
  - `{:error, Error.t()}`: An error struct if the request fails or is interrupted.

  ## Examples

      # Pull a model
      {:ok, :success} = AiFlow.Ollama.Model.pull_model("llama3.1")

      # Pull with debug logging
      {:ok, _} = AiFlow.Ollama.Model.pull_model("llama3.1", debug: true)

      # Pull without streaming (if supported by Ollama API)
      {:ok, full_response} = AiFlow.Ollama.Model.pull_model("llama3.1", stream: false, short: false)
  """
  @spec pull_model(String.t(), keyword()) :: {:ok, term()} | {:error, Error.t()}
  def pull_model(name, opts \\ []) do
    if is_binary(name) and name != "" do
      config = AiFlow.Ollama.get_config()
      debug = Keyword.get(opts, :debug, false)
      retries = Keyword.get(opts, :retries, 0)
      url = Config.build_url(config, "/api/pull")

      body_opts = Keyword.drop(opts, [:debug, :retries, :short, :field])
      body = Map.merge(%{name: name, stream: Keyword.get(opts, :stream, true)}, Enum.into(body_opts, %{}))

      http_response = HTTPClient.request(:post, url, body, config.timeout, debug, retries, :pull_model)

      success_handler = fn body ->
        check_pull_success(body)
      end

      HTTPClient.handle_response(http_response, "status", :pull_model, opts ++ [success_handler: success_handler])
    else
      {:error, Error.invalid("Model name must be a non-empty string")}
    end
  end

  # Helper to check pull success, reducing nesting in the main function
  defp check_pull_success(body) do
     if is_map(body) and (Map.get(body, "status") == "success" or Map.has_key?(body, "model")) do
       :success
     else
       body
     end
  end

  @doc """
  Pulls a model, raising on error.

  Similar to `pull_model/2`, but returns the result directly or raises an `AiFlow.Ollama.Error`.

  ## Parameters

  - `name`: The name of the model to pull (string).
  - `opts`: Keyword list of options (same as `pull_model/2`).

  ## Returns

  - `term()`: The result of the pull operation.
  - Raises `AiFlow.Ollama.Error` on failure.

  ## Examples

      AiFlow.Ollama.Model.pull_model!("llama3.1")
  """
  @spec pull_model!(String.t(), keyword()) :: term()
  def pull_model!(name, opts \\ []) do
    pull_model(name, opts)
    |> HTTPClient.handle_result(:pull_model)
  end

  @doc """
  Pushes a model to a remote registry.

  Uploads a locally available model to the Ollama library or another registry.

  ## Parameters

  - `name`: The name of the model to push (string).
  - `opts`: Keyword list of options:
    - `:debug` (boolean): Enables debug logging (default: `false`).
    - `:retries` (integer): Number of retry attempts for failed requests (default: `0`).
    - `:stream` (boolean): Whether to request a streaming response (default: `true`).
    - `:short` (boolean): If `false`, attempts to return the raw `%Req.Response{}`. (default: `true`).
    - `:field` (String.t()): When `:short` is `true`, specifies the field to extract. (default: `"status"`).

  ## Returns

  - `{:ok, :success | term()}`: Indicates successful push or contains status information.
  - `{:error, Error.t()}`: An error struct if the request fails.

  ## Examples

      # Push a model
      {:ok, :success} = AiFlow.Ollama.Model.push_model("my-model")

      # Push with debug logging
      {:ok, _} = AiFlow.Ollama.Model.push_model("my-model", debug: true)
  """
  @spec push_model(String.t(), keyword()) :: {:ok, :success | term()} | {:error, Error.t()}
  def push_model(name, opts \\ []) do
    if is_binary(name) and name != "" do
      config = AiFlow.Ollama.get_config()
      debug = Keyword.get(opts, :debug, false)
      retries = Keyword.get(opts, :retries, 0)
      url = Config.build_url(config, "/api/push")

      body_opts = Keyword.drop(opts, [:debug, :retries, :short, :field])
      body = Map.merge(%{name: name, stream: Keyword.get(opts, :stream, true)}, Enum.into(body_opts, %{}))

      http_response = HTTPClient.request(:post, url, body, config.timeout, debug, retries, :push_model)

      success_handler = fn body ->
        parsed = parse_streaming_response(body)
        handle_parsed_push_response(parsed)
      end

      HTTPClient.handle_response(http_response, "status", :push_model, opts ++ [success_handler: success_handler])
    else
      {:error, Error.invalid("Model name must be a non-empty string")}
    end
  end

  # Helper to handle parsed push response, reducing nesting
  defp handle_parsed_push_response(parsed) do
     if success_response?(parsed) do
        :success
     else
        parsed
     end
  end

  @doc """
  Pushes a model, raising on error.

  Similar to `push_model/2`, but returns the result directly or raises an `AiFlow.Ollama.Error`.

  ## Parameters

  - `name`: The name of the model to push (string).
  - `opts`: Keyword list of options (same as `push_model/2`).

  ## Returns

  - `:success | term()`: Indicates successful push or contains status information.
  - Raises `AiFlow.Ollama.Error` on failure.

  ## Examples

      AiFlow.Ollama.Model.push_model!("my-model")
  """
  @spec push_model!(String.t(), keyword()) :: :success | term()
  def push_model!(name, opts \\ []) do
    push_model(name, opts)
    |> HTTPClient.handle_result(:push_model)
  end

  @doc """
  Lists running models.

  Fetches a list of models that are currently loaded and running.

  ## Parameters

  - `opts`: Keyword list of options:
    - `:debug` (boolean): Enables debug logging (default: `false`).
    - `:retries` (integer): Number of retry attempts for failed requests (default: `0`).
    - `:short` (boolean): If `true`, processes the response to extract a specific part. (default: `true`).
    - `:field` (String.t()): When `:short` is `true`, specifies the field to extract.
                             Commonly `"models"` for the list of running model objects. (default: `"models"`).

  ## Returns

  - `{:ok, list()}`: A list of running model information maps when `:short` is `true`.
  - `{:ok, term()}`: The processed response based on `:short` and `:field` options.
  - `{:error, Error.t()}`: An error struct if the request fails.

  ## Examples

      # Get the full list of running models
      {:ok, running_models} = AiFlow.Ollama.Model.list_running_models()

      # Enable debug logging
      {:ok, _} = AiFlow.Ollama.Model.list_running_models(debug: true)

      # Get raw HTTP response
      {:ok, %Req.Response{status: 200, body: %{"models" => [...]}}} = AiFlow.Ollama.Model.list_running_models(short: false)
  """
  @spec list_running_models(keyword()) :: {:ok, list() | term()} | {:error, Error.t()}
  def list_running_models(opts \\ []) do
    config = AiFlow.Ollama.get_config()
    debug = Keyword.get(opts, :debug, false)
    short = Keyword.get(opts, :short, true) # Default to true for running models
    retries = Keyword.get(opts, :retries, 0)
    url = Config.build_url(config, "/api/ps")

    http_response = HTTPClient.request(:get, url, nil, config.timeout, debug, retries, :list_running_models)

    HTTPClient.handle_response(http_response, "models", :list_running_models, opts)
  end

  @doc """
  Lists running models, raising on error.

  Similar to `list_running_models/1`, but returns the result directly or raises an `AiFlow.Ollama.Error`.

  ## Parameters

  - `opts`: Keyword list of options (same as `list_running_models/1`).

  ## Returns

  - `list()` or `term()`: The list of running models or processed response based on options.
  - Raises `AiFlow.Ollama.Error` on failure.

  ## Examples

      running_models = AiFlow.Ollama.Model.list_running_models!()
  """
  @spec list_running_models!(keyword()) :: list() | term()
  def list_running_models!(opts \\ []) do
    list_running_models(opts)
    |> HTTPClient.handle_result(:list_running_models)
  end

  @doc """
  Loads a model into memory.

  Ensures a model is loaded into the Ollama server's memory, which can speed up
  subsequent requests.

  ## Parameters

  - `model`: The name of the model to load (string).
  - `opts`: Keyword list of options:
    - `:debug` (boolean): Enables debug logging (default: `false`).
    - `:retries` (integer): Number of retry attempts for failed requests (default: `0`).
    - `:short` (boolean): If `false`, returns the raw `%Req.Response{}`. (default: `true`).
    - `:field` (String.t()): When `:short` is `true`, specifies the field to extract. (default: `"status"`).

  ## Returns

  - `{:ok, :success | term()}`: Indicates successful load or contains status information.
  - `{:error, Error.t()}`: An error struct if the request fails.

  ## Examples

      # Load a model
      {:ok, :success} = AiFlow.Ollama.Model.load_model("llama3.1")

      # Load with debug logging
      {:ok, _} = AiFlow.Ollama.Model.load_model("llama3.1", debug: true)
  """
  @spec load_model(String.t(), keyword()) :: {:ok, :success | term()} | {:error, Error.t()}
  def load_model(model, opts \\ []) do
    if is_binary(model) and model != "" do
      config = AiFlow.Ollama.get_config()
      debug = Keyword.get(opts, :debug, false)
      retries = Keyword.get(opts, :retries, 0)
      url = Config.build_url(config, "/api/generate")
      body = %{model: model, prompt: "", stream: false}

      http_response = HTTPClient.request(:post, url, body, config.timeout, debug, retries, :load_model)

      success_handler = fn
        %{"done" => true} -> :success
        %{"status" => "success"} -> :success
        body -> body
      end

      HTTPClient.handle_response(http_response, "status", :load_model, opts ++ [success_handler: success_handler])
    else
      {:error, Error.invalid("Model name must be a non-empty string")}
    end
  end

  @doc """
  Loads a model, raising on error.

  Similar to `load_model/2`, but returns the result directly or raises an `AiFlow.Ollama.Error`.

  ## Parameters

  - `model`: The name of the model to load (string).
  - `opts`: Keyword list of options (same as `load_model/2`).

  ## Returns

  - `:success | term()`: Indicates successful load or contains status information.
  - Raises `AiFlow.Ollama.Error` on failure.

  ## Examples

      AiFlow.Ollama.Model.load_model!("llama3.1")
  """
  @spec load_model!(String.t(), keyword()) :: :success | term()
  def load_model!(model, opts \\ []) do
    load_model(model, opts)
    |> HTTPClient.handle_result(:load_model)
  end

  @doc """
  Creates a new model from a Modelfile.

  Defines and creates a new model based on a provided Modelfile content, base model,
  and system message.

  ## Parameters

  - `name`: The name for the new model (string).
  - `model`: The base model to use (string).
  - `system`: The system message or prompt template (string).
  - `opts`: Keyword list of options:
    - `:debug` (boolean): Enables debug logging (default: `false`).
    - `:retries` (integer): Number of retry attempts for failed requests (default: `0`).
    - `:modelfile` (String.t()): The content of the Modelfile (alternative to `model` and `system`).
    - `:stream` (boolean): Whether to request a streaming response (default: `true`).
    - `:short` (boolean): If `false`, attempts to return the raw `%Req.Response{}`. (default: `true`).
    - `:field` (String.t()): When `:short` is `true`, specifies the field to extract. (default: `"status"`).

  ## Returns

  - `{:ok, :success | term()}`: Indicates successful creation or contains status information.
  - `{:error, Error.t()}`: An error struct if the request fails.

  ## Examples

      # Create a model with base model and system message
      {:ok, :success} = AiFlow.Ollama.Model.create_model("my-custom-model", "llama3.1", "You are a helpful assistant.")

      # Create a model with a full Modelfile
      modelfile_content = "FROM llama3.1\\nSYSTEM You are a poet.\\nPARAMETER temperature 0.7"
      {:ok, :success} = AiFlow.Ollama.Model.create_model("my-poet-model", "", "", modelfile: modelfile_content)
  """
  @spec create_model(String.t(), String.t(), String.t(), keyword()) :: {:ok, :success | term()} | {:error, Error.t()}
  def create_model(name, model, system, opts \\ []) do
    if is_binary(name) and name != "" do
      config = AiFlow.Ollama.get_config()
      debug = Keyword.get(opts, :debug, false)
      retries = Keyword.get(opts, :retries, 0)
      url = Config.build_url(config, "/api/create")

      body =
        case Keyword.get(opts, :modelfile) do
          mf when is_binary(mf) and mf != "" ->
            %{name: name, stream: Keyword.get(opts, :stream, true), modelfile: mf}
          _ ->
            %{name: name, model: model, system: system, stream: Keyword.get(opts, :stream, true)}
        end

      http_response = HTTPClient.request(:post, url, body, config.timeout, debug, retries, :create_model)

      success_handler = fn body ->
        check_create_success(body)
      end

      HTTPClient.handle_response(http_response, "status", :create_model, opts ++ [success_handler: success_handler])
    else
      {:error, Error.invalid("Model name must be a non-empty string")}
    end
  end

  # Helper to check create success, avoiding =~ in guards and reducing nesting
  defp check_create_success(body) do
    case body do
      %{"status" => status} when is_binary(status) ->
        if String.contains?(status, "success"), do: :success, else: body
      _ ->
        body
    end
  end

  @doc """
  Creates a new model, raising on error.

  Similar to `create_model/4`, but returns the result directly or raises an `AiFlow.Ollama.Error`.

  ## Parameters

  - `name`: The name for the new model (string).
  - `model`: The base model to use (string).
  - `system`: The system message or prompt template (string).
  - `opts`: Keyword list of options (same as `create_model/4`).

  ## Returns

  - `:success | term()`: Indicates successful creation or contains status information.
  - Raises `AiFlow.Ollama.Error` on failure.

  ## Examples

      AiFlow.Ollama.Model.create_model!("my-custom-model", "llama3.1", "You are a helpful assistant.")
  """
  @spec create_model!(String.t(), String.t(), String.t(), keyword()) :: :success | term()
  def create_model!(name, model, system, opts \\ []) do
    create_model(name, model, system, opts)
    |> HTTPClient.handle_result(:create_model)
  end

  @doc """
  Copies a model.

  Creates a copy of an existing model under a new name.

  ## Parameters

  - `source`: The name of the existing model to copy (string).
  - `destination`: The name for the new copied model (string).
  - `opts`: Keyword list of options:
    - `:debug` (boolean): Enables debug logging (default: `false`).
    - `:retries` (integer): Number of retry attempts for failed requests (default: `0`).
    - `:short` (boolean): If `false`, returns the raw `%Req.Response{}`. (default: `true`).
    - `:field` (String.t()): When `:short` is `true`, specifies the field to extract.
                            Note: The `/api/copy` endpoint often returns an empty body,
                            so `:field` is less relevant for this function. (default: `"status"`).

  ## Returns

  - `{:ok, :success}`: Indicates successful copy.
  - `{:ok, %Req.Response{}}`: The raw response if `short: false`.
  - `{:error, Error.t()}`: An error struct if the request fails.

  ## Examples

      # Copy a model
      {:ok, :success} = AiFlow.Ollama.Model.copy_model("llama3.1", "llama3.1-backup")

      # Copy with debug logging
      {:ok, _} = AiFlow.Ollama.Model.copy_model("llama3.1", "llama3.1-test", debug: true)

      # Get raw response (less common for copy)
      {:ok, %Req.Response{status: 200}} = AiFlow.Ollama.Model.copy_model("llama3.1", "llama3.1-test2", short: false)
  """
  @spec copy_model(String.t(), String.t(), keyword()) :: {:ok, :success | Req.Response.t()} | {:error, Error.t()}
  def copy_model(source, destination, opts \\ []) do
    if is_binary(source) and source != "" and is_binary(destination) and destination != "" do
      config = AiFlow.Ollama.get_config()
      debug = Keyword.get(opts, :debug, false)
      retries = Keyword.get(opts, :retries, 0)
      url = Config.build_url(config, "/api/copy")

      body = %{source: source, destination: destination}

      http_response = HTTPClient.request(:post, url, body, config.timeout, debug, retries, :copy_model)

      case http_response do
        {:ok, %Req.Response{status: status}} when status >= 200 and status < 300 ->
          short = Keyword.get(opts, :short, true)
          if short do
            {:ok, :success}
          else
            {:ok, http_response}
          end
        {:ok, %Req.Response{} = error_response} ->
          {:error, Error.http(error_response.status, "HTTP request failed")}

        {:error, _reason} = error_tuple ->
          error_tuple
      end
    else
      {:error, Error.invalid("Source and destination model names must be non-empty strings")}
    end
  end

  @doc """
  Copies a model, raising on error.

  Similar to `copy_model/3`, but returns the result directly or raises an `AiFlow.Ollama.Error`.

  ## Parameters

  - `source`: The name of the existing model to copy (string).
  - `destination`: The name for the new copied model (string).
  - `opts`: Keyword list of options (same as `copy_model/3`).

  ## Returns

  - `:success`: Indicates successful copy.
  - `%Req.Response{}`: The raw response if `short: false`.
  - Raises `AiFlow.Ollama.Error` on failure.

  ## Examples

      AiFlow.Ollama.Model.copy_model!("llama3.1", "llama3.1-backup")
  """
  @spec copy_model!(String.t(), String.t(), keyword()) :: :success | Req.Response.t()
  def copy_model!(source, destination, opts \\ []) do
    case copy_model(source, destination, opts) do
      {:ok, result} -> result
      {:error, error} -> raise RuntimeError, message: "Copy model failed: #{inspect(error)}"
    end
  end

  @doc """
  Deletes a model.

  Removes a model from the local Ollama server.

  ## Parameters

  - `name`: The name of the model to delete (string).
  - `opts`: Keyword list of options:
    - `:debug` (boolean): Enables debug logging (default: `false`).
    - `:retries` (integer): Number of retry attempts for failed requests (default: `0`).
    - `:short` (boolean): If `false`, returns the raw `%Req.Response{}`. (default: `true`).
    - `:field` (String.t()): When `:short` is `true`, specifies the field to extract. (default: `"status"`).

  ## Returns

  - `{:ok, :success}`: Indicates successful deletion.
  - `{:ok, term()}`: The processed response based on `:short` and `:field` options.
  - `{:error, Error.t()}`: An error struct if the request fails.

  ## Examples

      # Delete a model
      {:ok, :success} = AiFlow.Ollama.Model.delete_model("old-model")

      # Delete with debug logging
      {:ok, _} = AiFlow.Ollama.Model.delete_model("temp-model", debug: true)
  """
  @spec delete_model(String.t(), keyword()) :: {:ok, :success | term()} | {:error, Error.t()}
  def delete_model(name, opts \\ []) do
    if is_binary(name) and name != "" do
      config = AiFlow.Ollama.get_config()
      debug = Keyword.get(opts, :debug, false)
      retries = Keyword.get(opts, :retries, 0)
      url = Config.build_url(config, "/api/delete")

      body = %{name: name}

      http_response = HTTPClient.request(:delete, url, body, config.timeout, debug, retries, :delete_model)

      success_handler = fn _ -> :success end

      HTTPClient.handle_response(http_response, "status", :delete_model, opts ++ [success_handler: success_handler])
    else
      {:error, Error.invalid("Model name must be a non-empty string")}
    end
  end

  @doc """
  Deletes a model, raising on error.

  Similar to `delete_model/2`, but returns the result directly or raises an `AiFlow.Ollama.Error`.

  ## Parameters

  - `name`: The name of the model to delete (string).
  - `opts`: Keyword list of options (same as `delete_model/2`).

  ## Returns

  - `:success | term()`: Indicates successful deletion or contains status information.
  - Raises `AiFlow.Ollama.Error` on failure.

  ## Examples

      AiFlow.Ollama.Model.delete_model!("temp-model")
  """
  @spec delete_model!(String.t(), keyword()) :: :success | term()
  def delete_model!(name, opts \\ []) do
    delete_model(name, opts)
    |> HTTPClient.handle_result(:delete_model)
  end

  # --- Private Helper Functions (if needed) ---

  # These helper functions are assumed to exist based on the original code snippet.
  # Their implementation would depend on the specific format of streaming responses.
  # They are not documented as they are internal.

  # Placeholder for streaming response parsing logic
  defp parse_streaming_response(body) do
    # Implementation would depend on how Ollama streams data
    # Often involves parsing NDJSON or handling chunks
    body
  end

  defp success_response?(parsed) do
    case parsed do
      %{"status" => status} when is_binary(status) ->
        String.contains?(status, "success")
      %{"done" => true} -> true
      _ -> false
    end
  end

  # Ensure ETS cache tables exist (if caching is used)
  defp ensure_cache(table) do
    case :ets.whereis(table) do
      :undefined -> :ets.new(table, [:named_table, :public, :set])
      tid -> tid
    end
  end
end
