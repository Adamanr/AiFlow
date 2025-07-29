defmodule AiFlow.Ollama.TextGeneration do
  @moduledoc """
  Handles text generation requests to the Ollama API using the `/api/generate` endpoint.

  This module provides a simple interface for sending prompts to Ollama and receiving
  generated text completions. It supports standard `{:ok, result} | {:error, reason}`
  return patterns as well as raising variants (`!`). It integrates with the shared
  `AiFlow.Ollama.HTTPClient` for consistent error handling, logging, retries, and
  response formatting via `:short` and `:field` options.

  ## Features

  - Generate text completions from prompts.
  - Support for specifying different Ollama models.
  - Automatic model pulling if a model is not found.
  - Standard and raising function variants.
  - Configurable response formatting with `:short` and `:field` options.
  - Debug logging and retry mechanisms.

  ## Examples

      # Basic text generation
      {:ok, response} = AiFlow.Ollama.TextGeneration.query("Why is the sky blue?")

      # Specify a model
      {:ok, response} = AiFlow.Ollama.TextGeneration.query("Write a haiku", model: "llama3.1")

      # Get the full API response
      {:ok, full_response} = AiFlow.Ollama.TextGeneration.query("Hello!", short: false)
      # full_response is %{"model" => "...", "response" => "...", "done" => true, ...}

      # Extract a specific field from the API response
      {:ok, model_name} = AiFlow.Ollama.TextGeneration.query("Hello!", short: false, field: "model")

      # Use raising variant
      response = AiFlow.Ollama.TextGeneration.query!("What is Elixir?")

      # Enable debug logging
      {:ok, _} = AiFlow.Ollama.TextGeneration.query("Debug this", debug: true)
  """

  require Logger
  alias AiFlow.Ollama.{Config, Error, HTTPClient}

  @doc """
  Sends a prompt to the Ollama API to generate a text completion.

  This function makes a POST request to the `/api/generate` endpoint. If the specified
  model is not found, it will automatically attempt to pull the model and retry the request.

  ## Parameters

  - `prompt`: The input text prompt to send to the model (string).
  - `opts`: Keyword list of options:
    - `:model` (string): The Ollama model to use (defaults to the model configured in `AiFlow.Ollama.Config`).
    - `:debug` (boolean): If `true`, logs detailed debug information about the request and response (default: `false`).
    - `:retries` (non_neg_integer): Number of times to retry the request on failure (default: `0`).
    - `:short` (boolean): Controls the format of the returned response.
      - `true` (default): Returns the generated text content (from the "response" field).
      - `false`: Returns the full JSON response map from the Ollama API.
    - `:field` (String.t()): When `:short` is `true`, specifies which field from the API response
                             to extract and return. If `:short` is `false`, this option is typically
                             ignored by `HTTPClient.handle_response/4` (default: `{:body, "response"}`).

  ## Returns

  - `{:ok, term()}`: The value of the specified `:field` from the API response when `:short` is `true`.
  - `{:error, Error.t()}`: An error struct if the request fails (network error, API error, etc.).

  ## Examples
    # Generate a completion for a prompt
    {:ok, response_text} = AiFlow.Ollama.TextGeneration.query("Explain quantum computing in simple terms.")
    # response_text is a string like "Quantum computing is a type of computing that uses..."

    # Use a specific model
    {:ok, response_text} = AiFlow.Ollama.TextGeneration.query("Write a poem", model: "mistral")

    # Get the full API response including metadata
    {:ok, full_api_response} = AiFlow.Ollama.TextGeneration.query("Hello!", short: false)
    # full_api_response is a map like:
    # %{
    #   "model" => "llama3.1",
    #   "created_at" => "2023-08-04T19:22:45.499127Z",
    #   "response" => "Hello! How can I assist you today?",
    #   "done" => true,
    #   "context" => [1, 2, 3, ...], # (if keep_alive or context was used)
    #   "total_duration" => 1234567890,
    #   "load_duration" => 543210987,
    #   ...
    # }

    # Extract a specific field from the API response (e.g., the model name used)
    {:ok, model_used} = AiFlow.Ollama.TextGeneration.query("Hi", short: false, field: "model")
    # model_used is "llama3.1"

    # Extract a different field from the main response content when short=true (less common for /api/generate)
    # This would depend on the structure of the "response" field if it were JSON, which it usually isn't for /api/generate.
    # More commonly, you'd use short: false, field: "model" as above.

    # Enable debug logging to see the raw request and response
    {:ok, _} = AiFlow.Ollama.TextGeneration.query("Debug me", debug: true)
    # This will log details like:
    # [debug] Sending request to http://localhost:11434/api/generate
    # [debug] Request body: %{...}
    # [debug] Ollama query response: Status=200, Body=%{...}

    # Handle a potential model not found error (will trigger auto-pull if configured in HTTPClient)
    case AiFlow.Ollama.TextGeneration.query("Hello", model: "non-existent-model") do
      {:ok, response} ->
        IO.puts("Response: \#{response}")
      {:error, %AiFlow.Ollama.Error{type: :http, status: 404}} ->
        IO.puts("Model not found, check if auto-pull is working or pull manually.")
      {:error, error} ->
        IO.puts("An error occurred: \#{inspect(error)}")
    end
  """
  @spec query(String.t(), keyword()) :: {:ok, term()} | {:error, Error.t()}
  def query(prompt, opts \\ []) do
    config = AiFlow.Ollama.get_config()
    debug = Keyword.get(opts, :debug, false)
    model = Keyword.get(opts, :model, config.model)
    field = Keyword.get(opts, :field, {:body, "response"})
    stream = Keyword.get(opts, :stream, false)
    url = Config.build_url(config, "/api/generate")

    body = %{model: model, prompt: prompt, stream: stream}

    HTTPClient.request(:post, url, body, config.timeout, debug, 0, :query)
    |> HTTPClient.handle_response(field, :query, opts)
  end

  @doc """
  Sends a prompt to generate a completion, raising an exception on error.

  This function behaves identically to `query/2`, but instead of returning
  `{:error, Error.t()}`, it raises a `RuntimeError` (or the specific error type
  handled by `HTTPClient.handle_result/2`) if the request fails.

  ## Parameters

  - `prompt`: The input text prompt to send to the model (string).
  - `opts`: Keyword list of options (same as `query/2`).

  ## Returns

  - `String.t()` or `term()`: The generated text completion or processed response based on `:short`/`:field`.
  - `Error.t()`: An error struct if the request fails.

  ## Examples

    # Successful generation
    AiFlow.Ollama.TextGeneration.query!("What are the benefits of using Elixir?")
    ["Elixir is a modern, dynamic language that runs on the Erlang VM (BEAM), and it has several benefits when compared to other programming languages. Here are some of the key advantages of using Elixir:",..

    # Raise on API error
    # response_text = AiFlow.Ollama.TextGeneration.query!("...", model: "invalid-model")
    nil

    # Get full response and raise on error
    full_response = AiFlow.Ollama.TextGeneration.query!("Hello!", short: false)
  """
  @spec query!(String.t(), keyword()) :: term() | Error.t()
  def query!(prompt, opts \\ []) do
    case query(prompt, opts) do
      {:ok, result} -> result
      {:error, error} -> error
      other -> other
    end
  end
end
