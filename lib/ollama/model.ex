defmodule AiFlow.Ollama.Model do
  @moduledoc """
  Manages model-related operations for the Ollama API.

  This module provides functions to interact with Ollama models, including listing,
  showing details, pulling, pushing, copying, deleting, creating, and loading models.
  It leverages the shared `AiFlow.Ollama.HTTPClient` for consistent HTTP handling,
  error management, and support for options like `:short`, `:field`, `:debug`, and `:retries`.

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
      AiFlow.Ollama.list_models()
      {:ok,
        [
          %{
            ...,
            "model" => "custom:latest",
            "name" => "custom1:latest",
            "size" => 669615493
            ...
          },
          %{
            ...,
            "model" => "mxbai-embed-large:latest",
            "name" => "mxbai-embed-large:latest",
            "size" => 669615493
          },
        ]}

      # Get only model names explicitly
      {:ok, model_names} = AiFlow.Ollama.list_models(short: true)
      {:ok,
        %Req.Response{
          status: 200,
          headers: %{...},
          body: %{
            "models" =>  [
              %{
                ...,
                "model" => "custom:latest",
                "name" => "custom1:latest",
                "size" => 669615493
                ...
              },
              %{
                ...,
                "model" => "mxbai-embed-large:latest",
                "name" => "mxbai-embed-large:latest",
                "size" => 669615493
              },
            ]
          },
          ...
        }}

      # Enable debug logging
      AiFlow.Ollama.list_models(debug: true)
      22:31:17.759 [debug] Ollama :get list_models request: URL=http://127.0.0.1:11434/api/tags, Body=nil
      22:31:17.786 [debug] Ollama show_model response: Status=200, Body=%{"models" => [%{"details" => %{"families" => ["llama"], "family" => "llama", "format" => "gguf", "parameter_size" => "8.0B", "parent_model" => "", "quantization_level" => "Q4_K_M"}
      {:ok,
      [
        %{
          ...,
          "model" => "custom:latest",
          "name" => "custom1:latest",
          "size" => 669615493,
          ...
        },
        %{
          ...,
          "model" => "mxbai-embed-large:latest",
          "name" => "mxbai-embed-large:latest",
          "size" => 669615493,
          ...
        },
      ]}

  """
  @spec list_models(keyword()) :: {:ok, list(String.t()) | list(map())} | {:error, Error.t()}
  def list_models(opts \\ []) do
    config = AiFlow.Ollama.get_config()
    debug = Keyword.get(opts, :debug, false)
    field = Keyword.get(opts, :field, {:body, "models"})
    retries = Keyword.get(opts, :retries, 0)
    url = Config.build_url(config, "/api/tags")

    HTTPClient.request(:get, url, nil, config.timeout, debug, retries, :list_models)
    |> HTTPClient.handle_response(field, :show_model, opts)
  end

  @doc """
  Lists available local models, raising on error.

  Similar to `list/1`, but returns the result directly or raises an `AiFlow.Ollama.Error`.

  ## Parameters

  - `opts`: Keyword list of options (same as `list/1`).

  ## Returns

  - `list()` or `term()`: The list of models or processed response based on options.
  - `Error.t()`: An error struct if the request fails.

  ## Examples

      models = AiFlow.Ollama.Model.list!()
      model_names = AiFlow.Ollama.Model.list!(short: true, field: "name")
  """
  @spec list_models!(keyword()) :: list() | term()
  def list_models!(opts \\ []) do
    case list_models(opts) do
      {:ok, result} -> result
      {:error, error} -> error
      other -> other
    end
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
      AiFlow.Ollama.Model.show_model("llama3.1")
      {:ok,
        %{
          "families" => ["llama"],
          "family" => "llama",
          "format" => "gguf",
          "parameter_size" => "8.0B",
          "parent_model" => "",
          "quantization_level" => "Q4_K_M"
        }}

      # Enable debug logging
      {:ok, _} = AiFlow.Ollama.Model.show_model("llama3.1", debug: true)
      22:33:22.461 [debug] Ollama :post show_model request: URL=http://127.0.0.1:11434/api/show, Body=%{name: "llama3.1"}
      22:33:22.506 [debug] Ollama show_model response: Status=200, Body=%{"capabilities" => ["completion", "tools"], "details" => %{"families" => ["llama"], "family" => "llama", "format" => "gguf", "parameter_size" => "8.0B", "parent_model" => "", "quantization_level" => "Q4_K_M"}, ...
      {:ok,
        %{
          "families" => ["llama"],
          "family" => "llama",
          "format" => "gguf",
          "parameter_size" => "8.0B",
          "parent_model" => "",
          "quantization_level" => "Q4_K_M"
        }}

      # Get raw HTTP response
      AiFlow.Ollama.Model.show_model("llama3.1", short: false)
      {:ok,
        %Req.Response{
          status: 200,
          headers: %{...},
          body: %{
            "capabilities" => ["completion", "tools"],
            "details" => %{
              "families" => ["llama"],
              "family" => "llama",
              "format" => "gguf",
              "parameter_size" => "8.0B",
              "parent_model" => "",
              "quantization_level" => "Q4_K_M"
            },
          ...}}

      # Extract a specific field from the model info (e.g., license)
      {:ok, license_text} = AiFlow.Ollama.Model.show_model("llama3.1", short: true, field: {:body, "capabilities"})
      {:ok, ["completion", "tools"]}
  """
  @spec show_model(String.t(), keyword()) :: {:ok, map() | term()} | {:error, Error.t()}
  def show_model(name, opts \\ []) do
    if is_binary(name) and name != "" do
      config = AiFlow.Ollama.get_config()
      debug = Keyword.get(opts, :debug, false)
      retries = Keyword.get(opts, :retries, 0)
      field = Keyword.get(opts, :field, {:body, "details"})

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
  - `Error.t()`: An error struct if the request fails.

  ## Examples

      AiFlow.Ollama.Model.show_model!("llama3.1", short: true, field: "license")
      ["LLAMA 3.1 COMMUNITY LICENSE AGREEMENT",
       "Llama 3.1 Version Release Date: July 23, 2024",
       "“Agreement” means the terms and conditions for use, reproduction, distribution and modification of the",
       "Llama Materials set forth herein.",
  """
  @spec show_model!(String.t(), keyword()) :: map() | term() | Error.t()
  def show_model!(name, opts \\ []) do
    case show_model(name, opts) do
      {:ok, result} -> result
      {:error, error} -> error
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
    - `:short` (boolean): If `false`, returns the raw response (default: `true`).
    - `:field` (String.t()): When `:short` is `true`, specifies the field to extract (default: `:status`).
    - `:display` (boolean): If `true`, displays a progress bar during pull (default: `true`).

  ## Returns
  - `{:ok, term()}`: Typically `:success` or a map with status information when the pull completes.
  - `{:error, Error.t()}`: An error struct if the request fails or is interrupted.

  ## Examples
      # Pull a model
      AiFlow.Ollama.Model.pull_model("llama3.1")
      📥 Pulling model: llama3.1
      {:ok, 200}

      AiFlow.Ollama.Model.pull_model("llama3.1", display: false)
      {:ok, 200}

      # Pull with debug logging
      {:ok, _} = AiFlow.Ollama.Model.pull_model("llama3.1", debug: true)
      iex(6)> AiFlow.Ollama.Model.pull_model("llama3.1", debug: true)
      📥 Pulling model: llama3.1
      23:06:57.489 [debug] Ollama :post pull_model request: URL=http://127.0.0.1:11434/api/pull, Body=%{name: "llama3.1", stream: true}
      23:06:58.110 [debug] Ollama generate_embeddings_legacy response: Status=200, Body="{\"status\":\"pulling manifest\"}\n{\"status\":\"pulling 667b0c1932bc\",\"digest\":\"sha256:667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284...
      {:ok, 200}

      # Show the entire response
      AiFlow.Ollama.Model.pull_model("llama3.1", short: false)
      {:ok,
        %Req.Response{
          status: 200,
          headers: %{...},
          body: "{\"status\":\"pulling manifest\"}\n{\"status\":\"pulling...",
          ...
        }}

      # Show a specific response field
      AiFlow.Ollama.Model.pull_model("llama3.1", field: :body)
      {:ok, "{\"status\":\"pulling manifest\"}\n{\"status\":\"pulling..."}
  """
  @spec pull_model(String.t(), keyword()) :: {:ok, term()} | {:error, Error.t()}
  def pull_model(name, opts \\ []) do
    config = AiFlow.Ollama.get_config()
    debug = Keyword.get(opts, :debug, false)
    display = Keyword.get(opts, :display, true)
    retries = Keyword.get(opts, :retries, 0)
    field = Keyword.get(opts, :field, :status)
    url = Config.build_url(config, "/api/pull")

    with true <- is_binary(name) and name != "" do
      body_opts = Keyword.drop(opts, [:debug, :retries, :short, :field, :display])

      body =
        Map.merge(
          %{name: name, stream: Keyword.get(opts, :stream, true)},
          Enum.into(body_opts, %{})
        )

      if display, do: IO.puts("📥 Pulling model: #{name}")

      HTTPClient.request(:post, url, body, config.timeout, debug, retries, :pull_model)
      |> HTTPClient.handle_response(field, :generate_embeddings_legacy, opts)
    else
      _ -> {:error, Error.invalid("Model name must be a non-empty string")}
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
  - `Error.t()`: An error struct if the request fails or is interrupted.

  ## Examples

      AiFlow.Ollama.Model.pull_model!("llama3.1")
  """
  @spec pull_model!(String.t(), keyword()) :: term() | Error.t()
  def pull_model!(name, opts \\ []) do
     case pull_model(name, opts) do
      {:ok, result} -> result
      {:error, error} -> error
      other -> other
    end
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
    - `:field` (String.t()): When `:short` is `true`, specifies the field to extract. (default: `:body`).

  ## Returns

  - `{:ok, term()}`: Indicates successful push or contains status information.
  - `{:error, Error.t()}`: An error struct if the request fails.

  ## Examples

      # Push a wrong model
      AiFlow.Ollama.Model.push_model("my-model")
      {:ok,
        [
          %{"status" => "retrieving manifest"},
          %{"status" => "couldn't retrieve manifest"},
          %{
            "error" => "open .ollama/models/manifests/registry.ollama.ai/library/my-model/latest: no such file or directory"
          }
        ]}
  """
  @spec push_model(String.t(), keyword()) :: {:ok, :success | term()} | {:error, Error.t()}
  def push_model(name, opts \\ []) do
    if is_binary(name) and name != "" do
      config = AiFlow.Ollama.get_config()
      debug = Keyword.get(opts, :debug, false)
      field = Keyword.get(opts, :field, :body)
      retries = Keyword.get(opts, :retries, 0)
      url = Config.build_url(config, "/api/push")

      body_opts = Keyword.drop(opts, [:debug, :retries, :short, :field])

      body =
        Map.merge(
          %{name: name, stream: Keyword.get(opts, :stream, true)},
          Enum.into(body_opts, %{})
        )

      HTTPClient.request(:post, url, body, config.timeout, debug, retries, :push_model)
      |> HTTPClient.handle_response(field, :push_model, opts)
    else
      {:error, Error.invalid("Model name must be a non-empty string")}
    end
  end

  @doc """
  Pushes a model, raising on error.

  Similar to `push_model/2`, but returns the result directly or raises an `AiFlow.Ollama.Error`.

  ## Parameters

  - `name`: The name of the model to push (string).
  - `opts`: Keyword list of options (same as `push_model/2`).

  ## Returns

  - `term()`: Indicates successful push or contains status information.
  - `Error.t()`: An error struct if the request fails or is interrupted.

  ## Examples

    # Push a wrong model
    AiFlow.Ollama.Model.push_model!("my-model")
    [
      %{"status" => "retrieving manifest"},
      %{"status" => "couldn't retrieve manifest"},
      %{
        "error" => "open .ollama/models/manifests/registry.ollama.ai/library/my-model/latest: no such file or directory"
      }
    ]

  """
  @spec push_model!(String.t(), keyword()) :: term() | Error.t()
  def push_model!(name, opts \\ []) do
    case push_model(name, opts) do
      {:ok, result} -> result
      {:error, error} -> error
      other -> other
    end
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
                             Commonly `"models"` for the list of running model objects. (default: `{:body, "models"}`).

  ## Returns

  - `{:ok, list()}`: A list of running model information maps when `:short` is `true`.
  - `{:ok, term()}`: The processed response based on `:short` and `:field` options.
  - `{:error, Error.t()}`: An error struct if the request fails.

  ## Examples

      # Get the empty list of running models
      AiFlow.Ollama.Model.list_running_models()
      {:ok, []}

      # Get the list of running models
      AiFlow.Ollama.load_model("llama3.1")
      AiFlow.Ollama.Model.list_running_models()
      {:ok,
        [
          %{
            "details" => %{...},
            "model" => "llama3.1:latest",
            "name" => "llama3.1:latest",
            "size" => 6450833408,
            ...
          }
        ]}

      # Enable debug logging
      AiFlow.Ollama.Model.list_running_models(debug: true)
      23:24:15.850 [debug] Ollama :get list_running_models request: URL=http://127.0.0.1:11434/api/ps, Body=nil
      23:24:15.850 [debug] Ollama list_running_models response: Status=200, Body=%{"models" => [%{"details" => %{"families" => ["llama"], "family" => "llama", "format" => "gguf", "parameter_size" => "8.0B", "parent_model" => "", "qu...
      {:ok,
        [
          %{
            "details" => %{...},
            "model" => "llama3.1:latest",
            "name" => "llama3.1:latest",
            "size" => 6450833408,
            ...
          }
        ]}

      # Show the entire response
      AiFlow.Ollama.Model.list_running_models(short: false)
      {:ok,
        %Req.Response{
          status: 200,
          headers: %{...},
          body: %{
            "models" => [
              %{
                "details" => %{....},
                "model" => "llama3.1:latest",
                "name" => "llama3.1:latest",
                "size" => 6450833408,
                ...
              }
            ]
          },
          ...
        }}


      # Show a specific response field
      AiFlow.Ollama.Model.list_running_models(field: :status)
      {:ok, 200}
  """
  @spec list_running_models(keyword()) :: {:ok, list() | term()} | {:error, Error.t()}
  def list_running_models(opts \\ []) do
    config = AiFlow.Ollama.get_config()
    debug = Keyword.get(opts, :debug, false)
    field = Keyword.get(opts, :field, {:body, "models"})
    retries = Keyword.get(opts, :retries, 0)
    url = Config.build_url(config, "/api/ps")

    HTTPClient.request(:get, url, nil, config.timeout, debug, retries, :list_running_models)
    |> HTTPClient.handle_response(field, :list_running_models, opts)
  end

  @doc """
  Lists running models, raising on error.

  Similar to `list_running_models/1`, but returns the result directly or raises an `AiFlow.Ollama.Error`.

  ## Parameters

  - `opts`: Keyword list of options (same as `list_running_models/1`).

  ## Returns

  - `list()` or `term()`: The list of running models or processed response based on options.
  - `Error.t()`: An error struct if the request fails.

  ## Examples

      running_models = AiFlow.Ollama.Model.list_running_models!()
  """
  @spec list_running_models!(keyword()) :: list() | term() | Error.t()
  def list_running_models!(opts \\ []) do
    case list_running_models(opts) do
      {:ok, result} -> result
      {:error, error} -> error
      other -> other
    end
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
    - `:field` (String.t()): When `:short` is `true`, specifies the field to extract. (default: `:status`).

  ## Returns

  - `{:ok, term()}`: Indicates successful load or contains status information.
  - `{:error, Error.t()}`: An error struct if the request fails.

  ## Examples

    # Load a model
    AiFlow.Ollama.Model.load_model("llama3.1")
    # => {:ok, 200}

    # Load with debug logging
    AiFlow.Ollama.Model.load_model("llama3.1", debug: true)
    23:28:14.329 [debug] Ollama :post load_model request: URL=http://127.0.0.1:11434/api/generate, Body=%{stream: false, prompt: "", model: "llama3.1"}
    23:28:14.360 [debug] Ollama load_model response: Status=200, Body=%{"created_at" => "2025-07-29T18:28:14.360732866Z", "done" => true, "done_reason" => "load", "model" => "llama3.1", "response" => ""}
    # => {:ok, 200}

    AiFlow.Ollama.Model.load_model("llama3.1", short: false)
    {:ok,
      %Req.Response{
        status: 200,
        headers: %{...},
        body: %{
          "model" => "llama3.1",
          ...
        },
        ...
      }}

    # Show a specific response field
    AiFlow.Ollama.Model.load_model("llama3.1", field: :status)
    # => {:ok, 200}
  """
  @spec load_model(String.t(), keyword()) :: {:ok, term()} | {:error, Error.t()}
  def load_model(model, opts \\ []) do
    if is_binary(model) and model != "" do
      config = AiFlow.Ollama.get_config()
      debug = Keyword.get(opts, :debug, false)
      field = Keyword.get(opts, :field, :status)
      retries = Keyword.get(opts, :retries, 0)
      url = Config.build_url(config, "/api/generate")
      body = %{model: model, prompt: "", stream: false}

      HTTPClient.request(:post, url, body, config.timeout, debug, retries, :load_model)
      |> HTTPClient.handle_response(field, :load_model, opts)
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

  - `term()`: Indicates successful load or contains status information.
  - `Error.t()`: An error struct if the request fails.

  ## Examples
      AiFlow.Ollama.Model.load_model!("llama3.1")
      200
  """
  @spec load_model!(String.t(), keyword()) :: term() | Error.t()
  def load_model!(model, opts \\ []) do
    case load_model(model, opts) do
      {:ok, result} -> result
      {:error, error} -> error
      other -> other
    end
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

  - `{:ok, term()}`: Indicates successful creation or contains status information.
  - `{:error, Error.t()}`: An error struct if the request fails.

  ## Examples

      # Create a model with base model and system message
      AiFlow.Ollama.Model.create_model("my-custom-model", "llama3.1", "You are a helpful assistant.")
      {:ok, 200}

      # Get debug logs
      AiFlow.Ollama.Model.create_model("my-custom-model", "llama3.1", "You are a helpful assistant.", debug: true)
      23:34:53.070 [debug] Ollama :post create_model request: URL=http://127.0.0.1:11434/api/create, Body=%{name: "my-custom-model", system: "You are a helpful assistant.", from: "llama3.1"}
      23:34:53.096 [debug] Ollama create_model response: Status=200, Body="{\"status\":\"using existing...
      {:ok, 200}

      # Get full response
      AiFlow.Ollama.Model.create_model("my-custom-model", "llama3.1", "You are a helpful assistant.", short: false)
      {:ok,
        %Req.Response{
          status: 200,
          headers: %{...},
          body: "{\"status\":\"using existing layer...",
          ...
        }}

      # Get specificate field
      AiFlow.Ollama.Model.create_model("my-custom-model", "llama3.1", "You are a helpful assistant.", short: false)
      {:ok,
        %Req.Response{
          status: 200,
          headers: %{...},
          body: "{\"status\":\"using existing layer...",
          ...
        }}

      # Show a specific response field
      AiFlow.Ollama.Model.create_model("my-custom-model", "llama3.1", "You are a helpful assistant.", field: :status)
      {:ok, 200}
  """
  @spec create_model(String.t(), String.t(), String.t(), keyword()) :: {:ok, term()} | {:error, Error.t()}
  def create_model(name, model, system, opts \\ []) do
    if is_binary(name) and name != "" do
      config = AiFlow.Ollama.get_config()
      debug = Keyword.get(opts, :debug, false)
      field = Keyword.get(opts, :field, :status)
      retries = Keyword.get(opts, :retries, 0)
      url = Config.build_url(config, "/api/create")

      body = %{name: name, from: model, system: system}

      HTTPClient.request(:post, url, body, config.timeout, debug, retries, :create_model)
      |> HTTPClient.handle_response(field, :create_model, opts)
    else
      {:error, Error.invalid("Model name must be a non-empty string")}
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

  - `term()`: Indicates successful creation or contains status information.
  - `Error.t()`: An error struct if the request fails.

  ## Examples

      AiFlow.Ollama.Model.create_model!("my-custom-model", "llama3.1", "You are a helpful assistant.")
  """
  @spec create_model!(String.t(), String.t(), String.t(), keyword()) :: term() | Error.t()
  def create_model!(name, model, system, opts \\ []) do
    case create_model(name, model, system, opts) do
      {:ok, result} -> result
      {:error, error} -> error
      other -> other
    end
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

  - `{:ok, term()}`: Indicates successful copy.
  - `{:error, Error.t()}`: An error struct if the request fails.

  ## Examples

      # Copy a model
      {:ok, :success} = AiFlow.Ollama.Model.copy_model("llama3.1", "llama3.1-backup")

      # Copy with debug logging
      {:ok, _} = AiFlow.Ollama.Model.copy_model("llama3.1", "llama3.1-test", debug: true)

      # Get raw response (less common for copy)
      {:ok, %Req.Response{status: 200}} = AiFlow.Ollama.Model.copy_model("llama3.1", "llama3.1-test2", short: false)
  """
  @spec copy_model(String.t(), String.t(), keyword()) :: {:ok, term()} | {:error, Error.t()}
  def copy_model(source, destination, opts \\ []) do
    if is_binary(source) and source != "" and is_binary(destination) and destination != "" do
      config = AiFlow.Ollama.get_config()
      debug = Keyword.get(opts, :debug, false)
      field = Keyword.get(opts, :field, :status)
      retries = Keyword.get(opts, :retries, 0)
      url = Config.build_url(config, "/api/copy")

      body = %{source: source, destination: destination}

      HTTPClient.request(:post, url, body, config.timeout, debug, retries, :copy_model)
      |> HTTPClient.handle_response(field, :copy_model, opts)
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

  - `term()`: Indicates successful copy.
  - `Error.t()`: An error struct if the request fails.

  ## Examples

      AiFlow.Ollama.Model.copy_model!("llama3.1", "llama3.1-backup")
  """
  @spec copy_model!(String.t(), String.t(), keyword()) :: term() | Error.t()
  def copy_model!(source, destination, opts \\ []) do
    case copy_model(source, destination, opts) do
      {:ok, result} -> result
      {:error, error} -> error
      other -> other
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

  - `{:ok, term()}`: The processed response based on `:short` and `:field` options.
  - `{:error, Error.t()}`: An error struct if the request fails.

  ## Examples

      # Delete a model
      {:ok, :success} = AiFlow.Ollama.Model.delete_model("old-model")

      # Delete with debug logging
      {:ok, _} = AiFlow.Ollama.Model.delete_model("temp-model", debug: true)
  """
  @spec delete_model(String.t(), keyword()) :: {:ok, term()} | {:error, Error.t()}
  def delete_model(name, opts \\ []) do
    if is_binary(name) and name != "" do
      config = AiFlow.Ollama.get_config()
      debug = Keyword.get(opts, :debug, false)
      field = Keyword.get(opts, :field, :status)
      retries = Keyword.get(opts, :retries, 0)
      url = Config.build_url(config, "/api/delete")

      body = %{name: name}

      HTTPClient.request(:delete, url, body, config.timeout, debug, retries, :delete_model)
      |> HTTPClient.handle_response(field, :generate_embeddings, opts)
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

  - `term()`: Indicates successful deletion or contains status information.
  - `Error.t()`: An error struct if the request fails.

  ## Examples

      AiFlow.Ollama.Model.delete_model!("temp-model")
  """
  @spec delete_model!(String.t(), keyword()) :: term() | Error.t()
  def delete_model!(name, opts \\ []) do
    case delete_model(name, opts) do
      {:ok, result} -> result
      {:error, error} -> error
      other -> other
    end
  end
end
