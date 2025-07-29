defmodule AiFlow.Ollama.Blob do
  @moduledoc """
  Manages blob operations for the Ollama API.

  Provides functions to check and upload blobs. This module handles interactions with the
  Ollama API's blob endpoints, allowing users to check if a blob exists by its SHA256 digest
  and upload new blobs from local files. It uses `AiFlow.Ollama.HTTPClient` for HTTP requests.

  ## Configuration

  The module relies on `AiFlow.Ollama.Config` for endpoint and timeout settings. Ensure that
  the configuration is properly set up before using these functions.

  ## Error Handling

  Functions return `{:ok, result}` on success or `{:error, Error.t()}` on failure. The bang (`!`)
  variants return Error.t() exception on failure

  ## Examples

      # Check if a blob exists
      iex> AiFlow.Ollama.Blob.check_blob("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2")
      {:ok, 200}

      # Check blob and raise on error
      iex> AiFlow.Ollama.Blob.check_blob!("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2")
      200

      # Upload a blob
      iex> AiFlow.Ollama.Blob.create_blob("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2", "/path/to/file.bin")
      {:ok, 200} depending on the server's response

      # Upload a blob and raise on error
      iex> AiFlow.Ollama.Blob.create_blob!("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2", "/path/to/file.bin")
      200

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
  SHA256 digest.

  ## Parameters

  - `digest`: A string representing the SHA256 digest of the blob (64 hex characters).
  - `opts`: Optional keyword list of options:
    - `:debug` (boolean): Enables debug logging (default: `false`).
    - `:retries` (integer): Number of retry attempts for failed requests (default: `0`).
    - `:short` (boolean): If `false`, returns the raw `%Req.Response{}` instead of the atom
      (`:exists` or `:not_found`). (default: `true`).
    - `:field` (atom() | String.t()): When `:short` is `false`, specifies a field to extract from the
      response. For HEAD requests, the body is usually empty. (default: `nil`).

  ## Returns

  - `{:ok, 200}`: Indicates whether the blob exists (when `:short` is `true`).
  - `{:ok, term()}`: The processed response based on `:short` and `:field` options.
  - `{:error, Error.t()}`: An error struct if the request fails.

  ## Examples

      iex> AiFlow.Ollama.Blob.check_blob("invalid_digest")
      {:error, %AiFlow.Ollama.Error{reason: :invalid, message: "Digest must be a valid SHA256 hash (64 hex characters)"}}

      iex> AiFlow.Ollama.Blob.check_blob("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2", debug: true)
      {:ok, 200}

      iex> AiFlow.Ollama.Blob.check_blob("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd4", retries: 2)
      {:error, 404}

      # Get raw response
      iex> AiFlow.Ollama.Blob.check_blob("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2", short: false)
      {:ok, %Req.Response{status: 200, headers: [{"content-length", "12345"}], body: "", ...}}

      # Get specific field from raw response
      iex> AiFlow.Ollama.Blob.check_blob("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2", field: :headers)
      {:ok, [{"content-length", "12345"}]}

      # Get specific field from raw response
      iex> AiFlow.Ollama.Blob.check_blob("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2", field: {:headers, "content-length"})
      {:ok, ["102"]}
  """
  @spec check_blob(String.t(), keyword()) :: {:ok, term()} | {:error, Error.t()}
  def check_blob(digest, opts \\ []) do
    with :ok <- validate_digest(digest) do
      config = AiFlow.Ollama.get_config()
      debug = Keyword.get(opts, :debug, false)
      field = Keyword.get(opts, :field, :status)
      retries = Keyword.get(opts, :retries, 0)
      url = Config.build_url(config, "/api/blobs/sha256:#{digest}")

      HTTPClient.request(:head, url, nil, config.timeout, debug, retries, :check_blob)
      |> HTTPClient.handle_response(field, :check_blob, opts)
    end
  end

  @doc """
  Checks if a blob exists by its SHA256 digest, raising on error.

  Similar to `check_blob/2`, but returns `200` if the blob exists, `404` if it does not and return `AiFlow.Ollama.Error` or `Req.Request` for other errors.

  ## Parameters

  - `digest`: A string representing the SHA256 digest of the blob (64 hex characters).
  - `opts`: Optional keyword list of options (same as `check_blob/2`).

  ## Returns

  - `term()`: The processed response based on `:short` and `:field` options.
  - `Error.t()`: An error struct if the request fails.

  ## Examples

      iex> AiFlow.Ollama.Blob.check_blob!("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2")
      200

      iex> AiFlow.Ollama.Blob.check_blob!("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd4")
      404

      iex> AiFlow.Ollama.Blob.check_blob!("invalid_digest")
      %AiFlow.Ollama.Error{
        message: "Digest must be a valid SHA256 hash (64 hex characters)",
        type: :invalid,
        reason: :invalid_input,
        status: nil
      }
      ** (AiFlow.Ollama.Error) Digest must be a valid SHA256 hash (64 hex characters)

      # Get raw response and raise on error
      iex> AiFlow.Ollama.Blob.check_blob!("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2", short: false)
      %Req.Response{status: 200, ...}
  """
  @spec check_blob!(String.t(), keyword()) :: term() | Error.t()
  def check_blob!(digest, opts \\ []) do
    case check_blob(digest, opts) do
      {:ok, result} -> result
      {:error, error} -> error
      other -> other
    end
  end

  @doc """
  Uploads a blob from a file to the Ollama API.

  Reads the content of a file and sends a POST request to the Ollama API to create a blob
  identified by its SHA256 digest.

  ## Parameters

  - `digest`: A string representing the SHA256 digest of the blob (64 hex characters).
  - `file_path`: A string representing the path to the file to upload.
  - `opts`: Optional keyword list of options:
    - `:debug` (boolean): Enables debug logging (default: `false`).
    - `:retries` (integer): Number of retry attempts for failed requests (default: `0`).
    - `:short` (boolean): If `false`, returns the raw `%Req.Response{}` instead of the atom
      (`:created` or `:success`). (default: `true`).
    - `:field` (atom() | String.t()): When `:short` is `false`, specifies a field to extract from the
      response body. (default: `:status`).

  ## Returns

  - `{:ok, 200}`: Indicates successful upload (when `:short` is `true`).
    - `201` is returned for HTTP 201.
    - `200` is returned for HTTP 200.
  - `{:ok, term()}`: The processed response based on `:short` and `:field` options.
  - `{:error, Error.t()}`: Contains error details if the file read or request fails.

  ## Examples

      iex> AiFlow.Ollama.Blob.create_blob("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2", "/path/to/file.txt")
      {:ok, 200} # or {:ok, 201}

      iex> AiFlow.Ollama.Blob.create_blob("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2", "/invalid/path")
      {:error, %AiFlow.Ollama.Error{reason: :enoent}}

      iex> AiFlow.Ollama.Blob.create_blob("invalid", "/path/to/file.txt")
      {:error, %AiFlow.Ollama.Error{reason: :invalid, message: "Digest must be a valid SHA256 hash (64 hex characters)"}}

      # Get raw response
      iex> AiFlow.Ollama.Blob.create_blob("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2", "/path/to/file.bin", short: false)
      {:ok, %Req.Response{status: 200, ...}
  """
  @spec create_blob(String.t(), String.t(), keyword()) :: {:ok, String.t()} | {:error, Error.t()}
  def create_blob(digest, file_path, opts \\ []) do
    with :ok <- validate_digest(digest),
         :ok <- validate_file_path(file_path) do

      config = AiFlow.Ollama.get_config()
      debug = Keyword.get(opts, :debug, false)
      field = Keyword.get(opts, :field, :status)
      retries = Keyword.get(opts, :retries, 0)
      url = Config.build_url(config, "/api/blobs/sha256:#{digest}")

      case File.read(file_path) do
        {:ok, content} ->
          HTTPClient.request(:post, url, content, config.timeout, debug, retries, :create_blob)
          |> HTTPClient.handle_response(field, :create_blob, opts)

        {:error, reason} ->
          {:error, Error.file(reason)}
      end
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

  - `term()`: The processed response based on `:short` and `:field` options.
  - `Error.t()`: An error struct if the request fails.

  ## Examples

      iex> AiFlow.Ollama.Blob.create_blob!("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2", "/path/to/file.txt")
      200

      iex> AiFlow.Ollama.Blob.create_blob!("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2", "/invalid/path")
      %AiFlow.Ollama.Error{reason: :enoent}

      # Get raw response and raise on error
      iex> AiFlow.Ollama.Blob.create_blob!("29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2", "/path/to/file.bin", short: false)
      %Req.Response{status: 200, ...}
  """
  @spec create_blob!(String.t(), String.t(), keyword()) :: term() | Error.t()
  def create_blob!(digest, file_path, opts \\ []) do
    case create_blob(digest, file_path, opts) do
      {:ok, result} -> result
      {:error, error} -> error
      other -> other
    end
  end

  @spec validate_digest(String.t()) :: :ok | {:error, Error.t()}
  defp validate_digest(digest) do
    if is_binary(digest) and Regex.match?(@sha256_regex, digest) do
      :ok
    else
      {:error, Error.invalid("Digest must be a valid SHA256 hash (64 hex characters)")}
    end
  end

  @spec validate_file_path(String.t()) :: :ok | {:error, Error.t()}
  defp validate_file_path(file_path) do
    cond do
      not is_binary(file_path) ->
        {:error, Error.invalid("File path must be a string")}

      not File.exists?(file_path) ->
        {:error, Error.file(:enoent)}

      true ->
        :ok
    end
  end
end
