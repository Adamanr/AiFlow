defmodule AiFlow.Ollama.ChatTest do
  use ExUnit.Case, async: true
  import Mox

  alias AiFlow.Ollama.Chat

  setup :verify_on_exit!

  setup do
    temp_chat_file = "test_chat_#{System.system_time()}.json"
    on_exit(fn -> File.rm(temp_chat_file) end)
    %{chat_file: temp_chat_file}
  end

  describe "show_chat_history/2 and show_chat_history!/2" do
    test "returns empty list for new chat" do
      assert {:error, :chat_not_found} = Chat.show_chat_history(chat_id: "new_chat", user_id: "new_user")
      assert :chat_not_found = Chat.show_chat_history!(chat_id: "new_chat", user_id: "new_user")
    end
  end

  describe "show_all_chats/0 and show_all_chats!/0" do
    test "returns empty chats structure" do
      assert {:ok, data} = Chat.show_all_chats()
      assert Map.has_key?(data, :chats)
      assert is_map(data.chats)
      assert %{} == data.chats
      assert is_map(Chat.show_all_chats!())
    end
  end

  describe "clear_chat_history/1" do
    test "requires confirm: true to delete all chats" do
      assert {:error, _} = Chat.clear_chat_history()
      assert {:ok, :success} = Chat.clear_chat_history(confirm: true)
    end
  end

  describe "debug_load_chat_data/0" do
    test "returns atomized chat data" do
      assert {:ok, data} = Chat.debug_load_chat_data()
      assert is_map(data)
      assert Map.has_key?(data, :chats)
    end
  end

  describe "debug_show_chat_history/2" do
    test "returns empty list for new chat" do
      assert [] = Chat.debug_show_chat_history("debug_chat", "debug_user")
    end
  end

  describe "check_chat_file/0" do
    test "returns error or empty chat data if file is missing" do
      result = Chat.check_chat_file()
      assert result == {:error, :file_not_found} or match?({:ok, %{"chats" => _}}, result)
    end
  end
end
