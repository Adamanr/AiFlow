defmodule AiFlow.Ollama.HTTPClient do
  @moduledoc """
  Handles HTTP requests to the Ollama API using the Req library.
  Supports GET, POST, and DELETE methods with retry logic, logging, and telemetry.
  """

  require Logger
  alias AiFlow.Ollama.Error

  @doc """
  Makes an HTTP request to the specified URL with the given method, body, and options.

  ## Parameters
  - `method`: HTTP method (`:get`, `:post`, `:delete`, `:head`).
  - `url`: The target URL for the request.
  - `body`: The request body (nil for GET/HEAD, binary or map for POST/DELETE).
  - `timeout`: Request timeout in milliseconds.
  - `debug`: Boolean to enable debug logging.
  - `retries`: Number of retries on network error.
  - `action`: Telemetry action identifier (e.g., `:check_blob`).

  ## Returns
  - `{:ok, term()}` on successful request with status 200.
  - `{:error, Error.t()}` on failure (HTTP error, network error, or unknown error).
  """
  @spec request(atom(), String.t(), term(), integer(), boolean(), integer(), atom()) :: {:ok, term()} | {:error, Error.t()}
  def request(:get, url, _body, timeout, debug, retries, action) do
    :telemetry.execute([:ai_flow, :ollama, action, :request], %{retries: retries}, %{url: url})
    if debug, do: Logger.debug("Ollama :get #{action} request: URL=#{url}, Body=nil")

    case Req.get(url, receive_timeout: timeout) do
      {:ok, req} when req.status in [200, 201] ->
        {:ok, req}

      {:ok, %Req.Response{status: 404} = req} ->
        {:error, req}

      {:ok, %Req.Response{status: status, body: body}} ->
        {:error, Error.http(status, body)}

      {:error, %Req.TransportError{reason: reason}} when retries > 0 ->
        Logger.debug("Transport error (retries left: #{retries}): #{inspect(reason)}")
        :timer.sleep(200)
        request(:get, url, nil, timeout, debug, retries - 1, action)

      {:error, %Req.Response{status: status} = resp} when status in 400..499 ->
        {:error, Error.client(resp)}

      {:error, %Req.Response{status: status} = resp} when status >= 500 ->
        {:error, Error.server(resp)}

      {:error, %Req.TransportError{reason: reason}} ->
        {:error, Error.network(reason)}

      {:error, reason} ->
        {:error, Error.unknown(reason)}
    end
  end

  def request(:head, url, _body, timeout, debug, retries, action) do
    :telemetry.execute([:ai_flow, :ollama, action, :request], %{retries: retries}, %{url: url})
    if debug, do: Logger.debug("Ollama :head #{action} request: URL=#{url}, Body=nil")

    case Req.head(url, receive_timeout: timeout) do
      {:ok, req} when req.status in [200, 201] ->
        {:ok, req}

      {:ok, %Req.Response{status: 404} = req} ->
        {:error, req}

      {:ok, %Req.Response{status: status, body: body}} ->
        {:error, Error.http(status, body)}

      {:error, %AiFlow.Ollama.Error{} = error} ->
        {:error, error}

      {:error, %Req.TransportError{reason: reason}} when retries > 0 ->
        Logger.debug("Transport error (retries left: #{retries}): #{inspect(reason)}")
        :timer.sleep(200)
        request(:head, url, nil, timeout, debug, retries - 1, action)

      {:error, %Req.Response{status: status} = resp} when status in 400..499 ->
        {:error, Error.client(resp)}

      {:error, %Req.Response{status: status} = resp} when status >= 500 ->
        {:error, Error.server(resp)}

      {:error, %Req.TransportError{reason: reason}} ->
        {:error, Error.network(reason)}

      {:error, reason} ->
        {:error, Error.unknown(reason)}
    end
  end

  def request(:post, url, body, timeout, debug, retries, action) do
    :telemetry.execute([:ai_flow, :ollama, action, :request], %{retries: retries}, %{url: url})
    if debug, do: Logger.debug("Ollama :post #{action} request: URL=#{url}, Body=#{inspect(body)}")

    case Req.post(url, json: body, receive_timeout: timeout) do
      {:ok, req} when req.status in [200, 201] ->
        {:ok, req}

      {:ok, %Req.Response{status: 404} = req} ->
        {:error, req}

      {:ok, %Req.Response{status: status, body: body}} ->
        {:error, Error.http(status, body)}

      {:error, %AiFlow.Ollama.Error{} = error} ->
        {:error, error}

      {:error, %Req.TransportError{reason: reason}} when retries > 0 ->
        Logger.debug("Transport error (retries left: #{retries}): #{inspect(reason)}")
        :timer.sleep(200)
        request(:post, url, body, timeout, debug, retries - 1, action)

      {:error, %Req.TransportError{reason: reason}} ->
        {:error, Error.network(reason)}

      {:error, %Req.Response{status: status} = resp} when status in 400..499 ->
        {:error, Error.client(resp)}

      {:error, %Req.Response{status: status} = resp} when status >= 500 ->
        {:error, Error.server(resp)}

      {:error, reason} ->
        {:error, Error.unknown(reason)}
    end
  end

  def request(:delete, url, body, timeout, debug, retries, action) do
    :telemetry.execute([:ai_flow, :ollama, action, :request], %{retries: retries}, %{url: url})
    if debug, do: Logger.debug("Ollama :delete #{action} request: URL=#{url}, Body=#{inspect(body)}")

    case Req.delete(url, json: body, receive_timeout: timeout) do
      {:ok, req} when req.status in [200, 201] ->
        {:ok, req}

      {:ok, %Req.Response{status: 404} = req} ->
        {:error, req}

      {:ok, %Req.Response{status: status, body: body}} ->
        {:error, Error.http(status, body)}

      {:error, %AiFlow.Ollama.Error{} = error} ->
        {:error, error}

      {:error, %Req.TransportError{reason: reason}} when retries > 0 ->
        Logger.debug("Transport error (retries left: #{retries}): #{inspect(reason)}")
        :timer.sleep(200)
        request(:post, url, body, timeout, debug, retries - 1, action)

      {:error, %Req.TransportError{reason: reason}} ->
        {:error, Error.network(reason)}

      {:error, %Req.Response{status: status} = resp} when status in 400..499 ->
        {:error, Error.client(resp)}

      {:error, %Req.Response{status: status} = resp} when status >= 500 ->
        {:error, Error.server(resp)}

      {:error, reason} ->
        {:error, Error.unknown(reason)}
    end
  end

  def handle_response(response, field, action, opts \\ []) do
    debug = Keyword.get(opts, :debug, false)
    short = Keyword.get(opts, :short, true)

    case response do
      {:ok, %Req.Response{body: body} = resp} ->
        handle_success_response(body, resp, field, action, debug, short, :ok)

      {:error, %Req.Response{status: 404, body: body} = resp} ->
        handle_success_response(body, resp, :body, action, debug, short, :error)

      {:error, reason} ->
        handle_generic_error(reason, action, debug)

      unexpected ->
        handle_unexpected_response(unexpected, action, debug)
    end
  end

  defp handle_success_response(body, req, field, action, debug, short, status) do
    if debug, do: Logger.debug("Ollama #{action} response: Status=200, Body=#{inspect(body)}")
    :telemetry.execute([:ai_flow, :ollama, action], %{}, %{result: :ok})

    if short do
      handle_short_response(req, field, action, debug, status)
    else
      {:ok, req}
    end
  end

  defp handle_short_response(%Req.Response{} = resp, field_spec, action, debug, status) do
    parse_if_json_string = fn
      value when is_binary(value) ->
        case Jason.decode(value) do
          {:ok, decoded_map} when is_map(decoded_map) ->
            decoded_map
          _ ->
            lines = String.split(value, "\n", trim: true)
            if length(lines) > 1 do
              decoded_list =
                Enum.map(lines, fn line ->
                  case Jason.decode(line) do
                    {:ok, decoded_map} when is_map(decoded_map) ->
                      decoded_map
                    _ ->
                      line
                  end
                end)

              decoded_list
            else
              value
            end
        end
      value ->
        value
    end

    result =
      case field_spec do
        field_name when is_atom(field_name) ->
          case Map.get(resp, field_name) do
            nil -> nil
            value -> parse_if_json_string.(value)
          end

        {parent_key, child_key} when is_atom(parent_key) and is_binary(child_key) ->
          case Map.get(resp, parent_key) do
            parent_value when is_map(parent_value) or is_list(parent_value) ->
              case get_in(parent_value, [child_key]) do
                nil -> nil
                value -> parse_if_json_string.(value)
              end
            _ ->
              nil
          end

        {parent_key, child_path} when is_atom(parent_key) and is_list(child_path) ->
          case Map.get(resp, parent_key) do
            parent_value when is_map(parent_value) or is_list(parent_value) ->
              case get_in(parent_value, child_path) do
                nil -> nil
                value -> parse_if_json_string.(value)
              end
            _ ->
              nil
          end

        field_name when is_binary(field_name) and is_map(resp.body) ->
          case Map.get(resp.body, field_name) do
            nil ->
              content = get_in(resp.body, ["message", field_name])
              parse_if_json_string.(content)
            value ->
              parse_if_json_string.(value)
          end

        nil when is_map(resp.body) ->
          case get_in(resp.body, ["message", "content"]) do
            nil -> resp.body
            content -> parse_if_json_string.(content)
          end

        _ when is_map(resp.body) ->
          case get_in(resp.body, ["message", "content"]) do
            nil -> resp.body
            content -> parse_if_json_string.(content)
          end

        _ ->
          if is_atom(field_spec) do
            case Map.get(resp, field_spec) do
              nil -> resp.body
              value -> parse_if_json_string.(value)
            end
          else
            parse_if_json_string.(resp.body)
          end
      end

    {status, result}
  rescue
    _ ->
      if debug do
        Logger.warning("Error while processing short response for action #{action}, returning raw body.")
        Logger.flush()
      end
      {status, resp.body}
  end

  defp handle_short_response(_body, field_spec, action, debug, _status) do
    message = "Invalid response body structure for Ollama #{action}, cannot extract field #{inspect(field_spec)}"
    if debug do
      Logger.warning(message)
      Logger.flush()
    end
    {:error, message}
  end

  defp handle_generic_error(reason, action, debug) do
    Logger.error("Error in Ollama #{action}: #{inspect(reason)}")
    if debug, do: Logger.debug("Ollama #{action} error: Reason=#{inspect(reason)}")
    :telemetry.execute([:ai_flow, :ollama, action], %{}, %{result: :error, reason: reason})
    {:error, reason}
  end

  defp handle_unexpected_response(unexpected, action, debug) do
    Logger.error("Unexpected response format in Ollama #{action}: #{inspect(unexpected)}")
    if debug, do: Logger.debug("Ollama #{action} unexpected response: #{inspect(unexpected)}")

    :telemetry.execute([:ai_flow, :ollama, action], %{}, %{
      result: :error,
      reason: :unexpected_format
    })

    {:error, {:unexpected_response, unexpected}}
  end

  @doc """
  Handles the result of an HTTP request, returning the result for success or the error tuple.

  ## Parameters
    - `result`: The result of the HTTP request (`{:ok, term()} | {:error, Error.t()}`).
    - `action`: The telemetry action identifier (e.g., `:check_blob`).

  ## Returns
    - `term()` on success.
    - `{:error, Error.t()}` on error.
  """
  @spec handle_result_safe({:ok, term()} | {:error, Error.t()}, atom()) :: term() | {:error, Error.t()}
  def handle_result_safe({:ok, result}, _action), do: result
  def handle_result_safe({:error, err}, _action), do: {:error, err}
end
