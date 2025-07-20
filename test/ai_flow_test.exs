defmodule AiFlowTest do
  use ExUnit.Case, async: true
  import Mox
  Mox.defmock(AiFlow.HTTPClientMock, for: AiFlow.HTTPClient)
  setup :verify_on_exit!

  setup do
    Application.put_env(:ai_flow, :http_client, AiFlow.HTTPClientMock)
    if Process.whereis(AiFlow.Ollama) do
      Agent.stop(AiFlow.Ollama)
    end

    temp_chat_file = "test_chats_#{System.system_time()}.json"

    {:ok, _pid} = AiFlow.Ollama.start_link(
      hostname: "localhost",
      port: 11434,
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

  describe "start_link/1" do
    test "starts with default configuration" do
      if Process.whereis(AiFlow.Ollama) do
        Agent.stop(AiFlow.Ollama)
      end

      {:ok, pid} = AiFlow.Ollama.start_link()

      assert is_pid(pid)
      assert AiFlow.Ollama.get_hostname() == "127.0.0.1"
      assert AiFlow.Ollama.get_port() == 11434
      assert AiFlow.Ollama.get_timeout() == 60_000
    end

    test "starts with custom configuration" do
      if Process.whereis(AiFlow.Ollama) do
        Agent.stop(AiFlow.Ollama)
      end

      {:ok, pid} = AiFlow.Ollama.start_link(
        hostname: "test-host",
        port: 12345,
        timeout: 30000
      )

      assert is_pid(pid)
      assert AiFlow.Ollama.get_hostname() == "test-host"
      assert AiFlow.Ollama.get_port() == 12345
      assert AiFlow.Ollama.get_timeout() == 30000
    end
  end

  describe "init/1" do
    test "initializes with default configuration" do
      state = AiFlow.Ollama.init()
      assert state.hostname == "127.0.0.1"
      assert state.port == 11434
      assert state.timeout == 60_000
    end

    test "initializes with custom configuration" do
      state = AiFlow.Ollama.init(
        hostname: "custom-host",
        port: 54321,
        timeout: 15000
      )

      assert state.hostname == "custom-host"
      assert state.port == 54321
      assert state.timeout == 15000
    end
  end

  describe "chat history management" do
    test "clears chat history" do
      AiFlow.Ollama.clear_chat_history()

      messages = AiFlow.Ollama.show_chat_history("test_chat", "test_user")

      assert length(messages) == 0
    end

    test "shows all chats" do
      AiFlow.Ollama.clear_chat_history()

      all_chats = AiFlow.Ollama.show_all_chats()
      assert Map.has_key?(all_chats, :chats)
      assert map_size(all_chats.chats) == 0
    end
  end

  describe "debug functions" do
    test "debug_load_chat_data works" do
      AiFlow.Ollama.clear_chat_history()

      {:ok, chat_data} = AiFlow.Ollama.debug_load_chat_data()

      assert Map.has_key?(chat_data, :chats)
      assert map_size(chat_data.chats) == 0
    end

    test "check_chat_file works" do
      {:ok, chat_data} = AiFlow.Ollama.check_chat_file()

      assert Map.has_key?(chat_data, "chats")
    end
  end

  describe "file operations" do
    test "create_blob handles file read error" do
      {:error, {:file_error, :enoent}} = AiFlow.Ollama.create_blob(
        "digest",
        "nonexistent_file.txt"
      )
    end
  end

  describe "configuration" do
    test "get_hostname returns configured hostname" do
      hostname = AiFlow.Ollama.get_hostname()

      assert is_binary(hostname)
    end

    test "get_port returns configured port" do
      port = AiFlow.Ollama.get_port()

      assert is_integer(port)
    end

    test "get_timeout returns configured timeout" do
      timeout = AiFlow.Ollama.get_timeout()

      assert is_integer(timeout)
    end
  end

  describe "chat file operations" do
    test "show_chat_file_content works" do
      content = AiFlow.Ollama.show_chat_file_content()

      assert is_map(content) || is_binary(content)
    end

    test "debug_show_chat_history works" do
      messages = AiFlow.Ollama.debug_show_chat_history("test_chat", "test_user")

      assert is_list(messages)
    end
  end

  describe "list_models/1" do
    test "returns list of models successfully" do
      expect(AiFlow.HTTPClientMock, :get, fn _url, _opts ->
        {:ok, %Req.Response{status: 200, body: %{"models" => [%{"name" => "llama3.1"}]}}}
      end)

      {:ok, models} = AiFlow.Ollama.list_models()

      assert is_list(models)
      assert length(models) > 0
    end

    test "retries on temporary network error and succeeds" do
      parent = self()
      ref = make_ref()

      Agent.start_link(fn -> 0 end, name: :retry_counter)

      expect(AiFlow.HTTPClientMock, :get, 2, fn url, _opts ->
        count = Agent.get_and_update(:retry_counter, fn c -> {c, c + 1} end)

        if count == 0 do
          {:error, %Req.TransportError{reason: :timeout}}
        else
          send(parent, {:retried, ref})
          {:ok, %Req.Response{status: 200, body: %{"models" => [%{"name" => "llama3.1"}]}}}
        end
      end)

      {:ok, models} = AiFlow.Ollama.list_models(retries: 1)

      assert models == [%{"name" => "llama3.1"}]
      assert_receive {:retried, ^ref}
    end

    test "uses cache on repeated call" do
      expect(AiFlow.HTTPClientMock, :get, 1, fn _url, _opts ->
        {:ok, %Req.Response{status: 200, body: %{"models" => [%{"name" => "cached"}]}}}
      end)

      {:ok, models1} = AiFlow.Ollama.list_models()
      {:ok, models2} = AiFlow.Ollama.list_models()

      assert models1 == models2
      assert models1 == [%{"name" => "cached"}]
    end

    test "emits telemetry events" do
      :telemetry.attach_many("test-list-models", [
        [:ai_flow, :ollama, :list_models],
        [:ai_flow, :ollama, :list_models, :request]
      ], fn event, measurements, metadata, _ ->
        send(self(), {:telemetry, event, measurements, metadata})
      end, nil)

      expect(AiFlow.HTTPClientMock, :get, fn _url, _opts ->
        {:ok, %Req.Response{status: 200, body: %{"models" => [%{"name" => "llama3.1"}]}}}
      end)

      {:ok, _} = AiFlow.Ollama.list_models()

      assert_receive {:telemetry, [:ai_flow, :ollama, :list_models, :request], %{retries: 0}, _}
      assert_receive {:telemetry, [:ai_flow, :ollama, :list_models], %{cache: false}, %{result: :ok}}

      :telemetry.detach("test-list-models")
    end
  end
end
