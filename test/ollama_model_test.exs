defmodule AiFlow.Ollama.ModelTest do
  use ExUnit.Case, async: true
  import Mox

  alias AiFlow.Ollama.Model

  setup :verify_on_exit!

  setup do
    temp_chat_file = "test_model_#{System.system_time()}.json"
    on_exit(fn -> File.rm(temp_chat_file) end)
    %{chat_file: temp_chat_file}
  end

  describe "list_models/1 and list_models!/1" do
    test "returns ok for valid config" do
      assert {:ok, _} = Model.list_models(debug: true)
    end
  end

  describe "show_model/2 and show_model!/2" do
    test "returns error for invalid name" do
      assert {:error, _} = Model.show_model("")
      assert %AiFlow.Ollama.Error{} = Model.show_model!("")
    end
  end

  describe "pull_model/2 and pull_model!/2" do
    test "returns error for invalid name" do
      assert {:error, _} = Model.pull_model("")
      assert %AiFlow.Ollama.Error{} = Model.pull_model!("")
    end
  end

  describe "push_model/2 and push_model!/2" do
    test "returns error for invalid name" do
      assert {:error, _} = Model.push_model("")
      assert %AiFlow.Ollama.Error{} = Model.push_model!("")
    end
  end

  describe "list_running_models/1 and list_running_models!/1" do
    test "returns ok for valid config" do
      assert {:ok, _} = Model.list_running_models(debug: true)
    end
  end

  describe "load_model/2 and load_model!/2" do
    test "returns error for invalid name" do
      assert {:error, _} = Model.load_model("")
      assert %AiFlow.Ollama.Error{} = Model.load_model!("")
    end
  end

  describe "create_model/4 and create_model!/4" do
    test "returns error for invalid name" do
      assert {:error, _} = Model.create_model("", "base", "sys")
      assert %AiFlow.Ollama.Error{} = Model.create_model!("", "base", "sys")
    end
  end

  describe "copy_model/3 and copy_model!/3" do
    test "returns error for invalid names" do
      assert {:error, %AiFlow.Ollama.Error{}} = Model.copy_model("", "dest")
      assert {:error, %AiFlow.Ollama.Error{}} = Model.copy_model("src", "")
      assert %AiFlow.Ollama.Error{} = Model.copy_model!("", "dest")
      assert %AiFlow.Ollama.Error{} = Model.copy_model!("src", "")
    end
  end

  describe "delete_model/2 and delete_model!/2" do
    test "returns error for invalid name" do
      assert {:error, %AiFlow.Ollama.Error{}} = Model.delete_model("")
      assert %AiFlow.Ollama.Error{} = Model.delete_model!("")
    end
  end
end
