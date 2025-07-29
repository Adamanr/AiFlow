defmodule AiFlow.Ollama.HttpClient do
  @callback get(String.t(), keyword()) :: {:ok, Req.Response.t()} | {:error, term()}
  @callback post(String.t(), keyword()) :: {:ok, Req.Response.t()} | {:error, term()}
  @callback head(String.t(), keyword()) :: {:ok, Req.Response.t()} | {:error, term()}
  @callback delete(String.t(), keyword()) :: {:ok, Req.Response.t()} | {:error, term()}
end

Mox.defmock(AiFlow.Ollama.MockReq, for: AiFlow.Ollama.HttpClient)
ExUnit.start()

Logger.configure(level: :warning)

{:ok, _pid} = AiFlow.Ollama.start_link(
  hostname: "localhost",
  port: 11434,
  timeout: 1000,
  chat_file: "test_global_chat.json"
)
