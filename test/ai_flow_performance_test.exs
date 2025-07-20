defmodule AiFlowPerformanceTest do
  use ExUnit.Case, async: false

  setup do
    if Process.whereis(AiFlow.Ollama) do
      try do
        Agent.stop(AiFlow.Ollama)
      catch
        :exit, _ -> :ok
      end
    end

    temp_chat_file = "test_chats_performance_#{System.system_time()}.json"

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

  describe "performance tests" do
    test "configuration getters performance" do
      iterations = 1000

      start_time = System.monotonic_time(:microsecond)

      Enum.each(1..iterations, fn _ ->
        AiFlow.Ollama.get_hostname()
        AiFlow.Ollama.get_port()
        AiFlow.Ollama.get_timeout()
      end)

      end_time = System.monotonic_time(:microsecond)
      duration = end_time - start_time

      avg_time = duration / (iterations * 3)
      IO.puts("Average time per configuration getter: #{avg_time} microseconds")

      assert avg_time < 1000
    end

    test "chat history operations performance" do
      iterations = 100

      start_time = System.monotonic_time(:microsecond)

      Enum.each(1..iterations, fn _ ->
        AiFlow.Ollama.show_chat_history("test_chat", "test_user")
      end)

      end_time = System.monotonic_time(:microsecond)
      duration = end_time - start_time

      avg_time = duration / iterations
      IO.puts("Average time per show_chat_history: #{avg_time} microseconds")

      assert avg_time < 10000
    end

    test "debug functions performance" do
      iterations = 50

      start_time = System.monotonic_time(:microsecond)

      Enum.each(1..iterations, fn _ ->
        AiFlow.Ollama.debug_load_chat_data()
      end)

      end_time = System.monotonic_time(:microsecond)
      duration = end_time - start_time

      avg_time = duration / iterations
      IO.puts("Average time per debug_load_chat_data: #{avg_time} microseconds")

      assert avg_time < 50000
    end

    test "file operations performance" do
      iterations = 100

      start_time = System.monotonic_time(:microsecond)

      Enum.each(1..iterations, fn _ ->
        AiFlow.Ollama.create_blob("test-digest", "nonexistent_file.txt")
      end)

      end_time = System.monotonic_time(:microsecond)
      duration = end_time - start_time

      avg_time = duration / iterations
      IO.puts("Average time per create_blob (nonexistent): #{avg_time} microseconds")

      assert avg_time < 5000
    end

    test "clear_chat_history performance" do
      iterations = 10

      start_time = System.monotonic_time(:microsecond)

      Enum.each(1..iterations, fn _ ->
        AiFlow.Ollama.clear_chat_history()
      end)

      end_time = System.monotonic_time(:microsecond)
      duration = end_time - start_time

      avg_time = duration / iterations
      IO.puts("Average time per clear_chat_history: #{avg_time} microseconds")

      assert avg_time < 100000
    end

    test "show_all_chats performance" do
      iterations = 100

      start_time = System.monotonic_time(:microsecond)

      Enum.each(1..iterations, fn _ ->
        AiFlow.Ollama.show_all_chats()
      end)

      end_time = System.monotonic_time(:microsecond)
      duration = end_time - start_time

      avg_time = duration / iterations
      IO.puts("Average time per show_all_chats: #{avg_time} microseconds")

      assert avg_time < 10000
    end

    test "check_chat_file performance" do
      iterations = 100

      start_time = System.monotonic_time(:microsecond)

      Enum.each(1..iterations, fn _ ->
        AiFlow.Ollama.check_chat_file()
      end)

      end_time = System.monotonic_time(:microsecond)
      duration = end_time - start_time

      avg_time = duration / iterations
      IO.puts("Average time per check_chat_file: #{avg_time} microseconds")

      assert avg_time < 10000
    end

    test "show_chat_file_content performance" do
      iterations = 50

      start_time = System.monotonic_time(:microsecond)

      Enum.each(1..iterations, fn _ ->
        AiFlow.Ollama.show_chat_file_content()
      end)

      end_time = System.monotonic_time(:microsecond)
      duration = end_time - start_time

      avg_time = duration / iterations
      IO.puts("Average time per show_chat_file_content: #{avg_time} microseconds")

      assert avg_time < 20000
    end

    test "debug_show_chat_history performance" do
      iterations = 100

      start_time = System.monotonic_time(:microsecond)

      Enum.each(1..iterations, fn _ ->
        AiFlow.Ollama.debug_show_chat_history("test_chat", "test_user")
      end)

      end_time = System.monotonic_time(:microsecond)
      duration = end_time - start_time

      avg_time = duration / iterations
      IO.puts("Average time per debug_show_chat_history: #{avg_time} microseconds")

      assert avg_time < 15000
    end
  end

  describe "memory usage tests" do
    test "memory usage after multiple operations" do
      Enum.each(1..100, fn i ->
        AiFlow.Ollama.show_chat_history("chat_#{i}", "user_#{i}")
        AiFlow.Ollama.debug_load_chat_data()
        AiFlow.Ollama.show_all_chats()
      end)

      assert Process.alive?(Process.whereis(AiFlow.Ollama))

      result = AiFlow.Ollama.get_hostname()
      assert is_binary(result)
    end

    test "memory usage after clear operations" do
      Enum.each(1..10, fn _ ->
        AiFlow.Ollama.clear_chat_history()
      end)

      assert Process.alive?(Process.whereis(AiFlow.Ollama))

      result = AiFlow.Ollama.get_hostname()
      assert is_binary(result)
    end
  end

  describe "concurrent access tests" do
    test "concurrent configuration access" do
      tasks = Enum.map(1..100, fn _ ->
        Task.async(fn ->
          AiFlow.Ollama.get_hostname()
          AiFlow.Ollama.get_port()
          AiFlow.Ollama.get_timeout()
        end)
      end)

      results = Enum.map(tasks, &Task.await/1)

      assert length(results) == 100

      assert Process.alive?(Process.whereis(AiFlow.Ollama))
    end

    test "concurrent chat history access" do
      tasks = Enum.map(1..50, fn i ->
        Task.async(fn ->
          AiFlow.Ollama.show_chat_history("chat_#{i}", "user_#{i}")
          AiFlow.Ollama.debug_load_chat_data()
        end)
      end)

      results = Enum.map(tasks, &Task.await/1)

      assert length(results) == 50

      assert Process.alive?(Process.whereis(AiFlow.Ollama))
    end
  end
end
