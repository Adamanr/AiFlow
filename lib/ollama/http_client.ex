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
    if debug, do: Logger.debug("Ollama #{action} request: URL=#{url}, Body=nil")

    case Req.get(url, receive_timeout: timeout) do
      {:ok, resp} when resp.status in [404, 200] ->
        {:ok, resp}
      {:ok, %Req.Response{status: status, body: body}} ->
        {:error, Error.http(status, body)}
      {:error, %Req.TransportError{reason: reason}} when retries > 0 ->
        Logger.debug("Transport error (retries left: #{retries}): #{inspect(reason)}")
        :timer.sleep(200)
        request(:get, url, nil, timeout, debug, retries - 1, action)
      {:error, %Req.TransportError{reason: reason}} ->
        {:error, Error.network(reason)}
      {:error, reason} ->
        {:error, Error.unknown(reason)}
    end
  end

  def request(:head, url, _body, timeout, debug, retries, action) do
    :telemetry.execute([:ai_flow, :ollama, action, :request], %{retries: retries}, %{url: url})
    if debug, do: Logger.debug("Ollama #{action} request: URL=#{url}, Body=nil")

    case Req.head(url, receive_timeout: timeout) do
      {:ok, %Req.Response{status: 200}} ->
        {:ok, :exists}
      {:ok, %Req.Response{status: 404}} ->
        {:ok, :not_found}
      {:ok, %Req.Response{status: status, body: body}} ->
        {:error, Error.http(status, body)}
      {:error, %Req.TransportError{reason: reason}} when retries > 0 ->
        Logger.debug("Transport error (retries left: #{retries}): #{inspect(reason)}")
        :timer.sleep(200)
        request(:head, url, nil, timeout, debug, retries - 1, action)
      {:error, %Req.TransportError{reason: reason}} ->
        {:error, Error.network(reason)}
      {:error, reason} ->
        {:error, Error.unknown(reason)}
    end
  end

  def request(:post, url, body, timeout, debug, retries, action) do
    :telemetry.execute([:ai_flow, :ollama, action, :request], %{retries: retries}, %{url: url})
    if debug, do: Logger.debug("Ollama #{action} request: URL=#{url}, Body=#{inspect(body)}")

    case Req.post(url, json: body, receive_timeout: timeout) do
      {:ok, req} when req.status in [200, 201] ->
        {:ok, req}
      {:ok, %Req.Response{status: status, body: body}} ->
        {:error, Error.http(status, body)}
      {:error, %Req.TransportError{reason: reason}} when retries > 0 ->
        Logger.debug("Transport error (retries left: #{retries}): #{inspect(reason)}")
        :timer.sleep(200)
        request(:post, url, body, timeout, debug, retries - 1, action)
      {:error, %Req.TransportError{reason: reason}} ->
        {:error, Error.network(reason)}
      {:error, reason} ->
        {:error, Error.unknown(reason)}
    end
  end

  def request(:delete, url, body, timeout, debug, retries, action) do
    :telemetry.execute([:ai_flow, :ollama, action, :request], %{retries: retries}, %{url: url})
    if debug, do: Logger.debug("Ollama #{action} request: URL=#{url}, Body=#{inspect(body)}")

    case Req.delete(url, json: body, receive_timeout: timeout) do
      {:ok, %Req.Response{status: 200, body: body}} ->
        {:ok, body}
      {:ok, %Req.Response{status: status, body: body}} ->
        {:error, Error.http(status, dynamic_message(status, body))}
      {:error, %Req.TransportError{reason: reason}} when retries > 0 ->
        Logger.debug("Transport error (retries left: #{retries}): #{inspect(reason)}")
        :timer.sleep(200)
        request(:delete, url, body, timeout, debug, retries - 1, action)
      {:error, %Req.TransportError{reason: reason}} ->
        {:error, Error.network(reason)}
      {:error, reason} ->
        {:error, Error.unknown(reason)}
    end
  end

  defp dynamic_message(404, _body), do: "Resource not found"
  defp dynamic_message(400, _body), do: "Bad request"
  defp dynamic_message(status, body), do: "HTTP error: #{status} - #{inspect(body)}"

  def handle_response(response, field, action, opts \\ []) do
    debug = Keyword.get(opts, :debug, false)
    short = Keyword.get(opts, :short, true)

    case response do
      {:ok, %Req.Response{status: 200, body: body} = req} ->
        handle_success_response(body, req, field, action, debug, short)

      {:ok, %Req.Response{status: 404} = resp} ->
        handle_404_response(resp, action, debug)

      {:error, %Req.TransportError{} = error} ->
        handle_network_error(error, action, debug)

      {:error, reason} ->
        handle_generic_error(reason, action, debug)

      unexpected ->
        handle_unexpected_response(unexpected, action, debug)
    end
  end

  def handle_response({:error, error}, debug, action, _success_handler) do
    case error do
      %Error{type: :http, status: status, reason: body} ->
        handle_http_error(status, body, action, debug)

      %Error{type: :network, reason: reason} ->
        handle_network_error(reason, action, debug)

      %Error{type: :unknown, reason: reason} ->
        handle_unknown_error(reason, action, debug)
    end
  end

  defp handle_success_response(body, req, field, action, debug, short) do
    if debug, do: Logger.debug("Ollama #{action} response: Status=200, Body=#{inspect(body)}")
    :telemetry.execute([:ai_flow, :ollama, action], %{}, %{result: :ok})

    if short do
      handle_short_response(body, field, action, debug)
    else
      {:ok, req}
    end
  end

  defp handle_short_response(body, field, action, debug) do
    case Map.fetch(body, field) do
      {:ok, value} ->
        {:ok, value}
      :error when field == nil ->
        {:ok, body}
      :error ->
        message = "Field '#{field}' not found in Ollama #{action} response"
        if debug do
          Logger.warning("#{message}: #{inspect(body)}")
          Logger.flush()
        end
        {:error, message}
    end
  end

  defp handle_404_response(resp, action, debug) do
    Logger.warning("Ollama #{action} returned 404")
    if debug, do: Logger.debug("Ollama #{action} response: Status=404, Body=#{inspect(resp)}")
    {:error, {:http_error, 404, resp}}
  end

  defp handle_network_error(error, action, debug) do
    Logger.error("Network error in Ollama #{action}: #{Exception.message(error)}")
    if debug, do: Logger.debug("Ollama #{action} error: Reason=#{inspect(error)}")
    :telemetry.execute([:ai_flow, :ollama, action], %{}, %{result: :error, reason: :network})
    {:error, {:network_error, error}}
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
    :telemetry.execute([:ai_flow, :ollama, action], %{}, %{result: :error, reason: :unexpected_format})
    {:error, {:unexpected_response, unexpected}}
  end

  defp handle_http_error(status, body, action, debug) do
    Logger.error("Ollama #{action} request failed with status #{status}: #{inspect(body)}")
    if debug, do: Logger.debug("Ollama #{action} response: Status=#{status}, Body=#{inspect(body)}")
    :telemetry.execute([:ai_flow, :ollama, action], %{}, %{result: :error, status: status})
    {:error, :status}
  end

  defp handle_unknown_error(reason, action, debug) do
    Logger.error("Ollama #{action} request failed: #{inspect(reason)}")
    if debug, do: Logger.debug("Ollama #{action} error: Reason=#{inspect(reason)}")
    :telemetry.execute([:ai_flow, :ollama, action], %{}, %{result: :unknown_error})
    {:error, reason}
  end

  @doc """
  Handles the result of an HTTP request, returning the result for success or raising an error.

  ## Parameters
    - `result`: The result of the HTTP request (`{:ok, term()} | {:error, Error.t()}`).
    - `action`: The telemetry action identifier (e.g., `:check_blob`).

  ## Returns
    - `term()` on success.
    - Raises `AiFlow.Ollama.Error` on error.
  """
  @spec handle_result({:ok, term()} | {:error, Error.t()}, atom()) :: term()
  def handle_result({:ok, result}, _action), do: result
  def handle_result({:error, err}, action), do: Error.raise_error(action, err)

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
