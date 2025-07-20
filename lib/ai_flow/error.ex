defmodule AiFlow.Ollama.Error do
  @moduledoc """
  Custom error types for AiFlow.Ollama
  """

  defexception [:message, :type, :details]

  @type t :: %__MODULE__{
          message: String.t(),
          type: atom(),
          details: any()
        }

  def new(type, message, details \\ nil) do
    %__MODULE__{type: type, message: message, details: details}
  end

  def network(reason), do: new(:network, "Network error", reason)
  def http(status, body), do: new(:http, "HTTP error: #{status}", %{status: status, body: body})
  def file(reason), do: new(:file, "File error", reason)
  def config(reason), do: new(:config, "Configuration error", reason)
  def unknown(reason), do: new(:unknown, "Unknown error", reason)
end
