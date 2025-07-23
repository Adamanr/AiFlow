defmodule AiFlowIntegrationTest do
  use ExUnit.Case, async: false
  import Mox
  setup :verify_on_exit!

  setup do
    if Process.whereis(AiFlow.Ollama) do
      Agent.stop(AiFlow.Ollama)
    end

    temp_chat_file = "test_chats_integration_#{System.system_time()}.json"

    {:ok, _pid} = AiFlow.Ollama.start_link(
      hostname: "localhost",
      port: 11_434,
      timeout: 5000,
      chat_file: temp_chat_file
    )

    on_exit(fn ->
      if Process.whereis(AiFlow.Ollama) do
        try do
          Agent.stop(AiFlow.Ollama)
        catch
          :exit, _ -> :ok
        end
      end

      try do
        File.rm(temp_chat_file)
      catch
        :error, _ -> :ok
      end
    end)

    %{chat_file: temp_chat_file}
  end

  describe "configuration functions" do
    test "get_hostname returns configured value" do
      hostname = AiFlow.Ollama.get_hostname()
      assert is_binary(hostname)
      assert hostname == "localhost"
    end

    test "get_port returns configured value" do
      port = AiFlow.Ollama.get_port()
      assert is_integer(port)
      assert port == 11_434
    end

    test "get_timeout returns configured value" do
      timeout = AiFlow.Ollama.get_timeout()
      assert is_integer(timeout)
      assert timeout == 5000
    end
  end

  describe "chat file operations" do
    test "clear_chat_history creates empty file" do
      AiFlow.Ollama.clear_chat_history()

      {:ok, chat_data} = AiFlow.Ollama.debug_load_chat_data()
      assert map_size(chat_data.chats) == 0
    end

    test "show_chat_history returns empty list for new chat" do
      messages = AiFlow.Ollama.show_chat_history("new_chat", "new_user")
      assert is_list(messages)
      assert Enum.empty?(messages)
    end

    test "show_all_chats returns empty structure" do
      AiFlow.Ollama.clear_chat_history()

      all_chats = AiFlow.Ollama.show_all_chats()
      assert Map.has_key?(all_chats, :chats)
      assert map_size(all_chats.chats) == 0
    end

    test "debug_load_chat_data works with empty file" do
      AiFlow.Ollama.clear_chat_history()

      {:ok, chat_data} = AiFlow.Ollama.debug_load_chat_data()
      assert Map.has_key?(chat_data, :chats)
      assert map_size(chat_data.chats) == 0
    end

    test "check_chat_file works" do
      {:ok, chat_data} = AiFlow.Ollama.check_chat_file()
      assert Map.has_key?(chat_data, "chats")
    end

    test "show_chat_file_content works" do
      content = AiFlow.Ollama.show_chat_file_content()
      assert is_map(content)
      assert Map.has_key?(content, "chats")
    end

    test "debug_show_chat_history works" do
      messages = AiFlow.Ollama.debug_show_chat_history("test_chat", "test_user")
      assert is_list(messages)
      assert Enum.empty?(messages)
    end
  end

  describe "file operations" do
    test "create_blob with nonexistent file" do
      result = AiFlow.Ollama.create_blob("test-digest", "nonexistent_file.txt")
      assert {:error, {:file_error, :enoent}} = result
    end
  end

  describe "network error handling" do
    test "list_models returns error when server is not available" do
      expect(AiFlow.HTTPClientMock, :get, fn _, _ -> {:error, %Req.TransportError{reason: :nxdomain}} end)
      result = AiFlow.Ollama.list_models()
      assert match?({:error, _}, result)
    end

    test "query returns error when server is not available" do
      expect(AiFlow.HTTPClientMock, :post, fn _, _ -> {:error, %Req.TransportError{reason: :nxdomain}} end)
      result = AiFlow.Ollama.query("Hello", "test-model")
      assert match?({:error, _}, result)
    end

    test "chat returns error when server is not available" do
      expect(AiFlow.HTTPClientMock, :post, fn _, _ -> {:error, %Req.TransportError{reason: :nxdomain}} end)
      result = AiFlow.Ollama.chat("Hello", "test_chat", "test_user")
      assert match?({:error, _}, result)
    end
  end

  describe "bang versions with errors" do
    test "list_models! raises on network error" do
      expect(AiFlow.HTTPClientMock, :get, fn _, _ -> {:error, %Req.TransportError{reason: :nxdomain}} end)
      assert_raise RuntimeError, ~r/Ollama list models request failed/, fn ->
        AiFlow.Ollama.list_models!()
      end
    end

    test "query! raises on network error" do
      expect(AiFlow.HTTPClientMock, :post, fn _, _ -> {:error, %Req.TransportError{reason: :nxdomain}} end)
      assert_raise RuntimeError, ~r/Ollama query request failed/, fn ->
        AiFlow.Ollama.query!("Hello", "test-model")
      end
    end

    test "chat! raises on network error" do
      expect(AiFlow.HTTPClientMock, :post, fn _, _ -> {:error, %Req.TransportError{reason: :nxdomain}} end)
      assert_raise RuntimeError, ~r/Ollama chat request failed/, fn ->
        AiFlow.Ollama.chat!("Hello", "test_chat", "test_user")
      end
    end
  end
end
