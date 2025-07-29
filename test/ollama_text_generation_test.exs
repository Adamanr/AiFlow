defmodule AiFlow.Ollama.TextGenerationTest do
  use ExUnit.Case, async: true
  import Mox

  alias AiFlow.Ollama.TextGeneration

  setup :verify_on_exit!

  setup do
    temp_chat_file = "test_textgen_#{System.system_time()}.json"
    on_exit(fn -> File.rm(temp_chat_file) end)
    %{chat_file: temp_chat_file}
  end

  describe "query/1,2 and query!/1,2" do
    test "returns error if config or model is invalid" do
      assert {:error, %{"error" => "model 'invalid-model' not found"}} = TextGeneration.query("Hello", model: "invalid-model")
      assert %{"error" => "model 'invalid-model' not found"} = TextGeneration.query!("Hello", model: "invalid-model")
    end
  end
end
