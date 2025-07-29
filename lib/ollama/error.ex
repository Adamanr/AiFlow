defmodule AiFlow.Ollama.Error do
  @moduledoc """
  Defines an error structure and helper functions for handling errors in the AiFlow.Ollama module.

  This module provides a consistent way to represent and handle errors that occur during interactions
  with the Ollama API, such as HTTP request failures, file operation errors, invalid inputs, or server
  issues. Errors are categorized by type (e.g., `:http`, `:network`, `:file`, `:invalid`), and each
  error includes a human-readable message for better debugging and user feedback.

  ## Error Structure
  The `%Error{}` struct contains:
    - `type`: An atom indicating the error category (`:http`, `:network`, `:file`, `:invalid`, `:pull`, `:server`, `:client`, `:unknown`).
    - `reason`: The underlying cause of the error (e.g., HTTP status, file error code, or custom term).
    - `message`: A human-readable description of the error.
    - `status`: An optional HTTP status code (for `:http` errors).

  ## Examples
      iex> AiFlow.Ollama.Error.invalid("Digest must be a valid SHA256 hash")
      %AiFlow.Ollama.Error{
        type: :invalid,
        reason: :invalid_input,
        message: "Digest must be a valid SHA256 hash",
        status: nil
      }

      iex> AiFlow.Ollama.Error.http(404, "Not found")
      %AiFlow.Ollama.Error{
        type: :http,
        reason: :not_found,
        message: "HTTP request failed with status 404: Not found",
        status: 404
      }

      iex> error = AiFlow.Ollama.Error.file(:enoent)
      iex> AiFlow.Ollama.Error.to_string(error)
      "File operation failed: file not found"
  """

  # Use defexception which will create the struct with :message field
  # and add additional fields we need
  defexception [:message, :type, :reason, :status]

  @type error_type ::
          :http | :network | :file | :invalid | :pull | :server | :client | :unknown
  @type error_reason ::
          {:http_error, integer(), term()}
          | :timeout
          | :conn_refused
          | :enoent
          | :eacces
          | :invalid_input
          | term()

  @type t :: %__MODULE__{
          type: error_type(),
          reason: error_reason(),
          message: String.t(),
          status: integer() | nil
        }

  @doc """
  Creates an error for invalid input parameters.

  ## Parameters
    - `message`: A string describing the invalid input.

  ## Returns
    - `%Error{}` with `type: :invalid`, `reason: :invalid_input`, and the provided `message`.

  ## Examples
      iex> AiFlow.Ollama.Error.invalid("Digest must be a valid SHA256 hash")
      %AiFlow.Ollama.Error{
        type: :invalid,
        reason: :invalid_input,
        message: "Digest must be a valid SHA256 hash",
        status: nil
      }
  """
  @spec invalid(String.t()) :: t()
  def invalid(message) when is_binary(message) do
    %__MODULE__{
      type: :invalid,
      reason: :invalid_input,
      message: message,
      status: nil
    }
  end

  def invalid(_), do: raise(ArgumentError, "message must be a string")

  @doc """
  Creates an error for HTTP-related issues.

  ## Parameters
    - `status`: An integer representing the HTTP status code (e.g., 404, 500).
    - `reason`: A term describing the HTTP error (e.g., "Not found", `:bad_request`).

  ## Returns
    - `%Error{}` with `type: :http`, the provided `status`, and a formatted `message`.

  ## Examples
      iex> AiFlow.Ollama.Error.http(404, "Not found")
      %AiFlow.Ollama.Error{
        type: :http,
        reason: :not_found,
        message: "HTTP request failed with status 404: Not found",
        status: 404
      }
  """
  @spec http(integer(), term()) :: t()
  def http(status, reason) when is_integer(status) do
    message = "HTTP request failed with status #{status}: #{format_reason(reason)}"

    %__MODULE__{
      type: :http,
      reason: normalize_http_reason(reason),
      message: message,
      status: status
    }
  end

  def http(_, _), do: raise(ArgumentError, "status must be an integer")

  @doc """
  Creates an error for network-related issues.

  ## Parameters
    - `reason`: A term describing the network error (e.g., `:timeout`, `:conn_refused`).

  ## Returns
    - `%Error{}` with `type: :network` and a formatted `message`.

  ## Examples
      iex> AiFlow.Ollama.Error.network(:timeout)
      %AiFlow.Ollama.Error{
        type: :network,
        reason: :timeout,
        message: "Network error: timeout",
        status: nil
      }
  """
  @spec network(term()) :: t()
  def network(reason) do
    message =
      case reason do
        :timeout -> "Network error: timeout"
        :conn_refused -> "Network error: connection refused"
        other -> "Network error: #{format_reason(other)}"
      end

    %__MODULE__{
      type: :network,
      reason: reason,
      message: message,
      status: nil
    }
  end

  @doc """
  Creates an error for file-related issues.

  ## Parameters
    - `reason`: An atom or term representing the file error (e.g., `:enoent`, `:eacces`).

  ## Returns
    - `%Error{}` with `type: :file` and a formatted `message`.

  ## Examples
      iex> AiFlow.Ollama.Error.file(:enoent)
      %AiFlow.Ollama.Error{
        type: :file,
        reason: :enoent,
        message: "File operation failed: file not found",
        status: nil
      }
  """
  @spec file(term()) :: t()
  def file(reason) do
    message =
      case reason do
        :enoent -> "File operation failed: file not found"
        :eacces -> "File operation failed: permission denied"
        other -> "File operation failed: #{format_reason(other)}"
      end

    %__MODULE__{
      type: :file,
      reason: reason,
      message: message,
      status: nil
    }
  end

  @doc """
  Creates an error for pull operation issues (e.g., pulling a model from the Ollama API).

  ## Parameters
    - `reason`: A term describing the pull error.

  ## Returns
    - `%Error{}` with `type: :pull` and a formatted `message`.

  ## Examples
      iex> AiFlow.Ollama.Error.pull("Model not found")
      %AiFlow.Ollama.Error{
        type: :pull,
        reason: "Model not found",
        message: "Pull operation failed: Model not found",
        status: nil
      }
  """
  @spec pull(term()) :: t()
  def pull(reason) do
    %__MODULE__{
      type: :pull,
      reason: reason,
      message: "Pull operation failed: #{format_reason(reason)}",
      status: nil
    }
  end

  @doc """
  Creates an error for server-related issues.

  ## Parameters
    - `reason`: A term describing the server error.

  ## Returns
    - `%Error{}` with `type: :server` and a formatted `message`.

  ## Examples
      iex> AiFlow.Ollama.Error.server("Internal server error")
      %AiFlow.Ollama.Error{
        type: :server,
        reason: "Internal server error",
        message: "Server error: Internal server error",
        status: nil
      }
  """
  @spec server(term()) :: t()
  def server(reason) do
    %__MODULE__{
      type: :server,
      reason: reason,
      message: "Server error: #{format_reason(reason)}",
      status: nil
    }
  end

  @doc """
  Creates an error for client-related issues.

  ## Parameters
    - `reason`: A term describing the client error.

  ## Returns
    - `%Error{}` with `type: :client` and a formatted `message`.

  ## Examples
      iex> AiFlow.Ollama.Error.client("Internal client error")
      %AiFlow.Ollama.Error{
        type: :client,
        reason: "Internal client error",
        message: "Client error: Internal client error",
        status: nil
      }
  """
  @spec client(term()) :: t()
  def client(reason) do
    %__MODULE__{
      type: :client,
      reason: reason,
      message: "Client error: #{format_reason(reason)}",
      status: nil
    }
  end

  @doc """
  Creates an error for unknown or uncategorized issues.

  ## Parameters
    - `reason`: A term describing the unknown error.

  ## Returns
    - `%Error{}` with `type: :unknown` and a formatted `message`.

  ## Examples
      iex> AiFlow.Ollama.Error.unknown(:unexpected_error)
      %AiFlow.Ollama.Error{
        type: :unknown,
        reason: :unexpected_error,
        message: "Unknown error: unexpected_error",
        status: nil
      }
  """
  @spec unknown(term()) :: t()
  def unknown(reason) do
    %__MODULE__{
      type: :unknown,
      reason: reason,
      message: "Unknown error: #{format_reason(reason)}",
      status: nil
    }
  end

  @doc """
  Raises an error with a formatted message based on the operation and error struct.

  ## Parameters
    - `operation`: An atom representing the operation that failed (e.g., `:check_blob`, `:create_blob`).
    - `error`: An `%Error{}` struct containing error details.

  ## Raises
    - Raises an `%AiFlow.Ollama.Error{}` with a formatted message.

  ## Examples
      iex> error = AiFlow.Ollama.Error.invalid("Invalid digest")
      iex> AiFlow.Ollama.Error.raise_error(:check_blob, error)
      ** (AiFlow.Ollama.Error) check_blob failed: Invalid digest

      iex> error = AiFlow.Ollama.Error.http(500, "Server error")
      iex> AiFlow.Ollama.Error.raise_error(:create_blob, error)
      ** (AiFlow.Ollama.Error) create_blob failed: HTTP request failed with status 500: Server error
  """
  @spec raise_error(atom(), t()) :: no_return()
  def raise_error(operation, %__MODULE__{message: message}) when is_atom(operation) do
    raise(__MODULE__, "#{operation} failed: #{message}")
  end

  def raise_error(_, _),
    do:
      raise(
        ArgumentError,
        "operation must be an atom and error must be an %AiFlow.Ollama.Error{}"
      )

  @doc """
  Converts an error struct to a string for logging or display.

  ## Parameters
    - `error`: An `%Error{}` struct.

  ## Returns
    - A string representing the error's message.

  ## Examples
      iex> error = AiFlow.Ollama.Error.file(:enoent)
      iex> AiFlow.Ollama.Error.to_string(error)
      "File operation failed: file not found"
  """
  @spec to_string(t()) :: String.t()
  def to_string(%__MODULE__{message: message}) do
    message
  end

  def to_string(_), do: raise(ArgumentError, "argument must be an %AiFlow.Ollama.Error{}")

  # Private helper to format reason for readable messages
  defp format_reason(reason) do
    case reason do
      reason when is_binary(reason) -> reason
      reason when is_atom(reason) -> Atom.to_string(reason)
      reason -> inspect(reason)
    end
  end

  # Private helper to normalize HTTP reason
  defp normalize_http_reason(reason) do
    case reason do
      "Not found" -> :not_found
      "Bad request" -> :bad_request
      "Unauthorized" -> :unauthorized
      "Forbidden" -> :forbidden
      "Internal server error" -> :internal_server_error
      other -> other
    end
  end
end
