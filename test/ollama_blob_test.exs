defmodule AiFlow.Ollama.BlobTest do
  use ExUnit.Case, async: true
  import Mox

  alias AiFlow.Ollama.Blob

  setup :verify_on_exit!

  setup do
    temp_file = "test_blob_#{System.system_time()}.bin"
    File.write!(temp_file, "test content")
    on_exit(fn -> File.rm(temp_file) end)
    %{temp_file: temp_file}
  end

  describe "check_blob/1,2 and check_blob!/1,2" do
    test "returns error for invalid digest" do
      assert {:error, _} = Blob.check_blob("invalid-digest")
      assert %AiFlow.Ollama.Error{} = Blob.check_blob!("invalid-digest")
    end
  end

  describe "create_blob/2,3 and create_blob!/2,3" do
    test "returns error for invalid digest or file path", %{temp_file: temp_file} do
      assert {:error, _} = Blob.create_blob("invalid-digest", temp_file)
      assert %AiFlow.Ollama.Error{} = Blob.create_blob!("invalid-digest", temp_file)
      assert {:error, _} = Blob.create_blob(String.duplicate("a", 64), "/nonexistent/path.txt")
      assert %AiFlow.Ollama.Error{} = Blob.create_blob!(String.duplicate("a", 64), "/nonexistent/path.txt")
    end
  end
end
