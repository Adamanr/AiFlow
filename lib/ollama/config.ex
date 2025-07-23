defmodule AiFlow.Ollama.Config do
  @moduledoc false
  defstruct hostname: "127.0.0.1", port: 11_434, model: "llama3.1", timeout: 60_000, chat_file: "chats.json"

  @type t :: %__MODULE__{
          hostname: String.t(),
          port: integer(),
          model: String.t(),
          timeout: integer(),
          chat_file: String.t()
        }

  @spec load(keyword()) :: t()
  def load(opts \\ []) do
    defaults = Application.get_env(:ai_flow_ollama, AiFlow.Ollama, [])
    hostname = Keyword.get(opts, :hostname, Keyword.get(defaults, :hostname, "127.0.0.1"))
    port = Keyword.get(opts, :port, Keyword.get(defaults, :port, 11_434))
    model = Keyword.get(opts, :model, Keyword.get(defaults, :model, "llama3.1"))
    timeout = Keyword.get(opts, :timeout, Keyword.get(defaults, :timeout, 60_000))
    chat_file = Keyword.get(opts, :chat_file, Keyword.get(defaults, :chat_file, "chats.json"))

    unless File.exists?(chat_file) do
      File.write!(chat_file, Jason.encode!(%{chats: %{}, created_at: DateTime.utc_now()}))
    end

    %__MODULE__{hostname: hostname, port: port, model: model, timeout: timeout, chat_file: chat_file}
  end

  @spec build_url(t(), String.t()) :: String.t()
  def build_url(config, endpoint), do: "http://#{config.hostname}:#{config.port}#{endpoint}"
end
