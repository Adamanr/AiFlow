defmodule AiFlow.Ollama.Blob do
  @moduledoc """
  Manages blob operations for the Ollama API, providing functions to check and upload blobs.

  This module handles interactions with the Ollama API's blob endpoints, allowing users to check
  if a blob exists by its SHA256 digest and upload new blobs from local files. It uses the
  `AiFlow.Ollama.HTTPClient` for HTTP requests and supports configuration options like debugging,
  retries, and response formatting with `:short` and `:field`.

  ## Configuration
  The module relies on `AiFlow.Ollama.Config` for endpoint and timeout settings. Ensure that
  the configuration is properly set up before using these functions.

  ## Error Handling
  Functions return `{:ok, result}` on success or `{:error, Error.t()}` on failure. The bang (`!`)
  variants raise an `AiFlow.Ollama.Error` exception on failure for convenience in contexts where
  errors are not explicitly handled.

  ## Examples
      # Check if a blob exists
      iex> AiFlow.Ollama.Blob.check_blob("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2")
      {:ok, :exists}

      # Check blob and raise on error
      iex> AiFlow.Ollama.Blob.check_blob!("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2")
      :exists

      # Upload a blob
      iex> AiFlow.Ollama.Blob.create_blob("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2", "/path/to/file.bin")
      {:ok, :created}

      # Upload a blob and raise on error
      iex> AiFlow.Ollama.Blob.create_blob!("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2", "/path/to/file.bin")
      :created

      # Get raw HTTP response for check_blob
      iex> AiFlow.Ollama.Blob.check_blob("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2", short: false)
      {:ok, %Req.Response{status: 200, ...}}

      # Get raw HTTP response for create_blob
      iex> AiFlow.Ollama.Blob.create_blob("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2", "/path/to/file.bin", short: false)
      {:ok, %Req.Response{status: 201, ...}}
  """

  require Logger
  alias AiFlow.Ollama.{Config, Error, HTTPClient}

  @sha256_regex ~r/^[a-fA-F0-9]{64}$/

  @doc """
  Checks if a blob exists by its SHA256 digest.

  Sends a HEAD request to the Ollama API to verify the existence of a blob identified by its
  SHA256 digest. Returns `:exists` if the blob is found, `:not_found` if it does not exist, or
  an error if the request fails.

  ## Parameters
    - `digest`: A string representing the SHA256 digest of the blob (64 hex characters).
    - `opts`: Optional keyword list of options:
      - `:debug` (boolean): Enables debug logging (default: `false`).
      - `:retries` (integer): Number of retry attempts for failed requests (default: `0`).
      - `:short` (boolean): If `false`, returns the raw `%Req.Response{}` instead of the atom
        (`:exists` or `:not_found`). (default: `true`).
      - `:field` (String.t()): When `:short` is `false`, specifies a field to extract from the
        response body (if any). For HEAD requests, the body is usually empty. (default: `"body"`).

  ## Returns
    - `{:ok, :exists | :not_found}`: Indicates whether the blob exists (when `:short` is `true`).
    - `{:ok, term()}`: The processed response based on `:short` and `:field` options.
    - `{:error, Error.t()}`: Contains error details if the request fails.

  ## Examples
      iex> AiFlow.Ollama.Blob.check_blob("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2fewfew")
      {:error, %AiFlow.Ollama.Error{reason: :invalid, message: "Digest must be a valid SHA256 hash (64 hex characters)"}

      iex> AiFlow.Ollama.Blob.check_blob("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2", debug: true)
      {:ok, :exists}

      iex> AiFlow.Ollama.Blob.check_blob("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd4", retries: 2)
      {:ok, :not_found}

      # Get raw response
      iex> AiFlow.Ollama.Blob.check_blob("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2", short: false)
      {:ok, %Req.Response{status: 200, headers: %{"content-length" => "12345"}, body: "", ...}}
  """
  @spec check_blob(String.t(), keyword()) :: {:ok, :exists | :not_found | term()} | {:error, Error.t()}
  def check_blob(digest, opts \\ []) do
    if is_binary(digest) and Regex.match?(@sha256_regex, digest) do
      config = AiFlow.Ollama.get_config()
      debug = Keyword.get(opts, :debug, false)
      url = Config.build_url(config, "/api/blobs/sha256:#{digest}")
      retries = Keyword.get(opts, :retries, 0)

      http_response = HTTPClient.request(:head, url, nil, config.timeout, debug, retries, :check_blob)

      HTTPClient.handle_response(http_response, "body", :check_blob, opts)
      |> case do
        {:ok, %Req.Response{status: 200}} -> {:ok, :exists}
        {:ok, %Req.Response{status: 404}} -> {:ok, :not_found}
        other -> other
      end
    else
      {:error, Error.invalid("Digest must be a valid SHA256 hash (64 hex characters)")}
    end
  end

  @doc """
  Checks if a blob exists by its SHA256 digest, raising on error.

  Similar to `check_blob/2`, but returns `:exists` if the blob exists, `:not_found` if it does not,
  and raises an `AiFlow.Ollama.Error` for other errors.

  ## Parameters
    - `digest`: A string representing the SHA256 digest of the blob (64 hex characters).
    - `opts`: Optional keyword list of options (same as `check_blob/2`).

  ## Returns
    - `:exists`: If the blob exists.
    - `:not_found`: If the blob does not exist (HTTP 404).
    - `term()`: The processed response based on `:short` and `:field` options.
    - Raises `AiFlow.Ollama.Error`: For other errors.

  ## Examples
      iex> AiFlow.Ollama.Blob.check_blob!("a" * 64)
      :exists

      iex> AiFlow.Ollama.Blob.check_blob!("b" * 64)
      :not_found

      iex> AiFlow.Ollama.Blob.check_blob!("invalid")
      ** (AiFlow.Ollama.Error) check_blob failed: invalid digest

      # Get raw response and raise on error
      iex> AiFlow.Ollama.Blob.check_blob!("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2", short: false)
      %Req.Response{status: 200, ...}
  """
  @spec check_blob!(String.t(), keyword()) :: :exists | :not_found | term()
  def check_blob!(digest, opts \\ []) do
    check_blob(digest, opts)
    |> HTTPClient.handle_result(:check_blob)
    |> case do
      %Req.Response{status: 200} -> :exists
      %Req.Response{status: 404} -> :not_found
      other -> other
    end
  end

  @doc """
  Uploads a blob from a file to the Ollama API.

  Reads the content of a file and sends a POST request to the Ollama API to create a blob
  identified by its SHA256 digest. Returns `:created` for HTTP 201 (new blob created) or
  `:success` for HTTP 200 (blob already exists or other success case), or an error if the
  file cannot be read or the request fails.

  ## Parameters
    - `digest`: A string representing the SHA256 digest of the blob (64 hex characters).
    - `file_path`: A string representing the path to the file to upload.
    - `opts`: Optional keyword list of options:
      - `:debug` (boolean): Enables debug logging (default: `false`).
      - `:retries` (integer): Number of retry attempts for failed requests (default: `0`).
      - `:short` (boolean): If `false`, returns the raw `%Req.Response{}` instead of the atom
        (`:created` or `:success`). (default: `true`).
      - `:field` (String.t()): When `:short` is `false`, specifies a field to extract from the
        response body. (default: `"body"`).

  ## Returns
    - `{:ok, :created | :success}`: Indicates successful upload (when `:short` is `true`).
    - `{:ok, term()}`: The processed response based on `:short` and `:field` options.
    - `{:error, Error.t()}`: Contains error details if the file read or request fails.

  ## Examples
      iex> AiFlow.Ollama.Blob.create_blob("a" * 64, "/path/to/file.txt")
      {:ok, :created}

      iex> AiFlow.Ollama.Blob.create_blob("a" * 64, "/invalid/path")
      {:error, %AiFlow.Ollama.Error{reason: :enoent}}

      iex> AiFlow.Ollama.Blob.create_blob("invalid", "/path/to/file.txt")
      {:error, %AiFlow.Ollama.Error{reason: :invalid, message: "Digest must be a valid SHA256 hash (64 hex characters)"}}

      # Get raw response
      iex> AiFlow.Ollama.Blob.create_blob("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2", "/path/to/file.bin", short: false)
      {:ok, %Req.Response{status: 201, headers: %{"content-type" => "application/json"}, body: %{"status" => "success"}, ...}}
  """
  @spec create_blob(String.t(), String.t(), keyword()) :: {:ok, :created | :success | term()} | {:error, Error.t()}
  def create_blob(digest, file_path, opts \\ []) do
    with {:digest, true} <- {:digest, is_binary(digest) and Regex.match?(@sha256_regex, digest)},
         {:file_path, true} <- {:file_path, is_binary(file_path)},
         {:file_exists, true} <- {:file_exists, File.exists?(file_path)} do
      config = AiFlow.Ollama.get_config()
      debug = Keyword.get(opts, :debug, false)
      url = Config.build_url(config, "/api/blobs/sha256:#{digest}")
      retries = Keyword.get(opts, :retries, 0)

      case File.read(file_path) do
        {:ok, content} ->
          http_response = HTTPClient.request(:post, url, content, config.timeout, debug, retries, :create_blob)

          HTTPClient.handle_response(http_response, "body", :create_blob, opts)
          |> case do
            {:ok, %Req.Response{status: 201}} -> {:ok, :created}
            {:ok, %Req.Response{status: 200}} -> {:ok, :success}
            other -> other
          end
        {:error, reason} ->
          {:error, Error.file(reason)}
      end
    else
      {:digest, _} ->
        {:error, Error.invalid("Digest must be a valid SHA256 hash (64 hex characters)")}
      {:file_path, _} ->
        {:error, Error.invalid("File path must be a string")}
      {:file_exists, _} ->
        {:error, Error.file(:enoent)}
    end
  end

  @doc """
  Uploads a blob from a file to the Ollama API, raising on error.

  Similar to `create_blob/3`, but returns `:created` or `:success` on success and raises an
  `AiFlow.Ollama.Error` on failure.

  ## Parameters
    - `digest`: A string representing the SHA256 digest of the blob (64 hex characters).
    - `file_path`: A string representing the path to the file to upload.
    - `opts`: Optional keyword list of options (same as `create_blob/3`).

  ## Returns
    - `:created | :success`: Indicates successful upload.
    - `term()`: The processed response based on `:short` and `:field` options.
    - Raises `AiFlow.Ollama.Error`: If the file read or request fails.

  ## Examples
      iex> AiFlow.Ollama.Blob.create_blob!("a" * 64, "/path/to/file.txt")
      :created

      iex> AiFlow.Ollama.Blob.create_blob!("a" * 64, "/invalid/path")
      ** (AiFlow.Ollama.Error) create_blob failed: :enoent

      # Get raw response and raise on error
      iex> AiFlow.Ollama.Blob.create_blob!("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2", "/path/to/file.bin", short: false)
      %Req.Response{status: 201, ...}
  """
  @spec create_blob!(String.t(), String.t(), keyword()) :: :created | :success | term()
  def create_blob!(digest, file_path, opts \\ []) do
    create_blob(digest, file_path, opts)
    |> HTTPClient.handle_result(:create_blob)
    |> case do
      %Req.Response{status: 201} -> :created
      %Req.Response{status: 200} -> :success
      other -> other
    end
  end
end
