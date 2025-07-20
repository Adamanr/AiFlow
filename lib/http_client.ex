defmodule AiFlow.HTTPClient do
  @moduledoc """
  HTTP client behaviour for AiFlow.Ollama.
  """

  @callback get(String.t(), keyword()) :: any()
  @callback post(String.t(), keyword()) :: any()
  @callback delete(String.t(), keyword()) :: any()
  @callback head(String.t(), keyword()) :: any()

  def get(url, opts \\ []), do: Req.get(url, opts)
  def post(url, opts \\ []), do: Req.post(url, opts)
  def delete(url, opts \\ []), do: Req.delete(url, opts)
  def head(url, opts \\ []), do: Req.head(url, opts)
end
