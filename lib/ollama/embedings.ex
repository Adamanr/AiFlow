defmodule AiFlow.Ollama.Embeddings do
  @moduledoc """
  Handles embedding generation for the Ollama API.

  This module provides functions to generate vector embeddings from text using
  Ollama's embedding models. It supports both the modern `/api/embed` endpoint
  for batch processing and the legacy `/api/embeddings` endpoint for single texts.

  ## Features

  - Automatic model pulling when a model is not found
  - Support for single and batch embedding generation
  - Both safe (`{:ok, result} | {:error, reason}`) and raising (`!`) variants
  - Debug mode for inspecting raw API responses

  ## Examples

  Generate embeddings for a single text:

      {:ok, embeddings} = AiFlow.Ollama.Embeddings.generate_embeddings("Hello world")

  Generate embeddings for multiple texts:

      {:ok, embeddings} = AiFlow.Ollama.Embeddings.generate_embeddings([
        "First sentence",
        "Second sentence"
      ])

  Generate embeddings with a specific model:

      {:ok, embeddings} = AiFlow.Ollama.Embeddings.generate_embeddings(
        "Hello world",
        model: "nomic-embed-text"
      )

  Use debug mode to see the full response logs:

      {:ok, embeddings} = AiFlow.Ollama.Embeddings.generate_embeddings(
        "Hello world",
        debug: true
      )

  Get full API response instead of just embeddings

      {:ok, full_response} = AiFlow.Ollama.Embeddings.generate_embeddings(
        "Hello world",
        short: false
      )

  Extract a specific field from the response

      {:ok, model_info} = AiFlow.Ollama.Embeddings.generate_embeddings(
        "Hello world",
        field: "model"
      )

  Use the legacy endpoint for single text embeddings:

      {:ok, embedding} = AiFlow.Ollama.Embeddings.generate_embeddings_legacy(
        "Hello world"
      )
  """

  require Logger
  alias AiFlow.Ollama.{Config, Error, HTTPClient, Model}

  @doc """
  Generates embeddings for input text(s).

  This function uses the modern `/api/embed` endpoint which supports both single
  strings and lists of strings for batch processing. If the specified model is
  not found, it will automatically attempt to pull the model and retry.

  ## Parameters

  - `input` - A string or list of strings to generate embeddings for
  - `opts` - Keyword list of options:
    - `:model` - The embedding model to use (default: `"llama3.1"`)
    - `:debug` - If `true`, logs debug information (default: `false`)
    - `:short` - If `false`, returns the full API response (default: `true`)
    - `:field` - The field name to extract from response (default: `"embeddings"`)

  ## Returns

  - `{:ok, list()}` - Success case with a list of embeddings
  - `{:error, Error.t()}` - Error case with detailed error information

  ## Examples

      # Single text embedding
      {:ok, embeddings} = AiFlow.Ollama.Embeddings.generate_embeddings("Hello world")

      # Multiple texts embedding
      {:ok, embeddings} = AiFlow.Ollama.Embeddings.generate_embeddings([
        "First text",
        "Second text"
      ])

      # With custom model
      {:ok, embeddings} = AiFlow.Ollama.Embeddings.generate_embeddings(
        "Hello world",
        model: "nomic-embed-text"
      )

      # Debug mode
      {:ok, response} = AiFlow.Ollama.Embeddings.generate_embeddings(
        "Hello world",
        debug: true
      )

      # View full response
      {:ok, response} = AiFlow.Ollama.Embeddings.generate_embeddings(
        "Hello world",
        short: false
      )

      # View field from response
      {:ok, response} = AiFlow.Ollama.Embeddings.generate_embeddings(
        "Hello world",
        field: "total_duration"
      )
  """
  @spec generate_embeddings(String.t() | [String.t()], keyword()) :: {:ok, list()} | {:error, Error.t()}
  def generate_embeddings(input, opts \\ []) do
    config = AiFlow.Ollama.get_config()
    debug = Keyword.get(opts, :debug, false)
    short = Keyword.get(opts, :short, true)
    model = Keyword.get(opts, :model, "llama3.1")
    field = Keyword.get(opts, :field, "embeddings")
    url = Config.build_url(config, "/api/embed")
    body = Map.merge(%{model: model, input: input}, Enum.into(Keyword.delete(opts, :debug), %{}))

    HTTPClient.request(:post, url, body, config.timeout, false, 0, :generate_embeddings)
    |> HTTPClient.handle_response(field, :generate_embeddings, opts)
  end

  @doc """
  Generates embeddings, raising on error.

  Same as `generate_embeddings/2` but raises an exception instead of returning
  `{:error, reason}`. Useful when you expect the operation to succeed and want
  to avoid pattern matching.

  ## Parameters

  - `input` - A string or list of strings to generate embeddings for
  - `opts` - Keyword list of options:
    - `:model` - The embedding model to use (default: `"llama3.1"`)
    - `:debug` - If `true`, logs debug information (default: `false`)
    - `:short` - If `false`, returns the full API response (default: `true`)
    - `:field` - The field name to extract from response (default: `"embeddings"`)

  ## Returns

  - `list()` - List of embeddings on success
  - Raises `AiFlow.Ollama.Error` on failure

  ## Examples

      # Single text embedding
      embeddings = AiFlow.Ollama.Embeddings.generate_embeddings!("Hello world")

      # Multiple texts embedding
      embeddings = AiFlow.Ollama.Embeddings.generate_embeddings!([
        "First text",
        "Second text"
      ])

      # With custom model
      embeddings = AiFlow.Ollama.Embeddings.generate_embeddings!(
        "Hello world",
        model: "nomic-embed-text"
      )

      # View full response
      embeddings = AiFlow.Ollama.Embeddings.generate_embeddings!(
        "Hello world",
        short: false
      )

      # View field from response
      total_duration = AiFlow.Ollama.Embeddings.generate_embeddings!(
        "Hello world",
        field: "total_duration"
      )
  """
  @spec generate_embeddings!(String.t() | [String.t()], keyword()) :: list()
  def generate_embeddings!(input, opts \\ []), do: HTTPClient.handle_result(generate_embeddings(input, opts), :generate_embeddings)

  @doc """
  Generates embeddings using the legacy endpoint.

  Uses the legacy `/api/embeddings` endpoint which only accepts a single string
  (prompt) and returns one embedding. This endpoint is maintained for backward
  compatibility with older versions of Ollama.

  Note: This endpoint does not support batch processing or automatic model pulling.

  ## Parameters

  - `prompt` - A single string to generate embedding for
  - `opts` - Keyword list of options:
    - `:model` - The embedding model to use (default: `"llama3.1"`)
    - `:debug` - If `true`, logs debug information (default: `false`)
    - `:short` - If `false`, returns the full API response (default: `true`)
    - `:field` - The field name to extract from response (default: `"embedding"`)

  ## Returns

  - `{:ok, list()}` - Success case with a list containing one embedding
  - `{:error, Error.t()}` - Error case with detailed error information

  ## Examples

      # Generate single embedding
      {:ok, embedding} = AiFlow.Ollama.Embeddings.generate_embeddings_legacy("Hello world")

      # With custom model
      {:ok, embedding} = AiFlow.Ollama.Embeddings.generate_embeddings_legacy(
        "Hello world",
        model: "llama3.1"
      )

      # Debug mode
      {:ok, embedding} = AiFlow.Ollama.Embeddings.generate_embeddings_legacy(
        "Hello world",
        debug: true,
      )

      # View full response
      {:ok, embedding} = AiFlow.Ollama.Embeddings.generate_embeddings_legacy(
        "Hello world",
        short: false
      )
  """
  @spec generate_embeddings_legacy(String.t(), keyword()) :: {:ok, list()} | {:error, Error.t()}
  def generate_embeddings_legacy(prompt, opts \\ []) do
    config = AiFlow.Ollama.get_config()
    debug = Keyword.get(opts, :debug, false)
    short = Keyword.get(opts, :short, true)
    model = Keyword.get(opts, :model, "llama3.1")
    field = Keyword.get(opts, :field, "embedding")
    url = Config.build_url(config, "/api/embeddings")
    body = Map.merge(%{model: model, prompt: prompt}, Enum.into(Keyword.delete(opts, :debug), %{}))

    HTTPClient.request(:post, url, body, config.timeout, debug, 0, :generate_embeddings_legacy)
    |> HTTPClient.handle_response(field, :generate_embeddings_legacy, opts)
  end

  @doc """
  Generates embeddings (legacy), raising on error.

  Same as `generate_embeddings_legacy/2` but raises an exception instead of returning
  `{:error, reason}`. Useful when you expect the operation to succeed and want
  to avoid pattern matching.

  ## Parameters

  - `prompt` - A single string to generate embedding for
  - `opts` - Keyword list of options:
    - `:model` - The embedding model to use (default: `"llama3.1"`)
    - `:debug` - If `true`, logs debug information (default: `false`)
    - `:short` - If `false`, returns the full API response (default: `true`)
    - `:field` - The field name to extract from response (default: `"embedding"`)

  ## Returns

  - `list()` - List containing one embedding on success
  - Raises `AiFlow.Ollama.Error` on failure

  ## Examples

      # Generate single embedding
      embedding = AiFlow.Ollama.Embeddings.generate_embeddings_legacy!("Hello world")

      # With custom model
      embedding = AiFlow.Ollama.Embeddings.generate_embeddings_legacy!(
        "Hello world",
        model: "llama3.1"
      )

      # With custom model
      embedding = AiFlow.Ollama.Embeddings.generate_embeddings_legacy!(
        "Hello world",
        model: "llama3.1"
      )

      # Debug mode
      embedding = AiFlow.Ollama.Embeddings.generate_embeddings_legacy!(
        "Hello world",
        debug: true,
      )

      # View full response
      embedding = AiFlow.Ollama.Embeddings.generate_embeddings_legacy!(
        "Hello world",
        short: false
      )
  """
  @spec generate_embeddings_legacy!(String.t(), keyword()) :: list()
  def generate_embeddings_legacy!(prompt, opts \\ []), do: HTTPClient.handle_result(generate_embeddings_legacy(prompt, opts), :generate_embeddings_legacy)
end
