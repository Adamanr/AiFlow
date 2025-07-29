defmodule AiFlow.Ollama.EmbeddingsTest do
  use ExUnit.Case, async: true
  import Mox

  alias AiFlow.Ollama.Embeddings

  setup :verify_on_exit!

  setup do
    temp_chat_file = "test_embeddings_#{System.system_time()}.json"
    on_exit(fn -> File.rm(temp_chat_file) end)
    %{chat_file: temp_chat_file}
  end

  describe "generate_embeddings/1,2 and generate_embeddings!/1,2" do
    test "returns error if model is invalid" do
      assert {:error, %Req.Response{}} = Embeddings.generate_embeddings("Hello", model: "invalid-model")
      assert %Req.Response{} = Embeddings.generate_embeddings!("Hello", model: "invalid-model")
    end
  end

  describe "generate_embeddings_legacy/1,2 and generate_embeddings_legacy!/1,2" do
    test "returns error if model is invalid" do
      assert {:error, %Req.Response{}} = Embeddings.generate_embeddings_legacy("Hello", model: "invalid-model")
      assert %Req.Response{} = Embeddings.generate_embeddings_legacy!("Hello", model: "invalid-model")
    end
  end
end
