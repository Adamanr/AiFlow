defmodule AiFlow.MixProject do
  use Mix.Project

  def project do
    [
      app: :ai_flow,
      version: "0.1.0",
      elixir: "~> 1.18",
      build_embedded: Mix.env == :prod,
      start_permanent: Mix.env == :prod,
      description: "A simple Elixir library for interacting with the Ollama API",
      package: package(),
      deps: deps(),
      name: "AiFlow",
      source_url: "https://github.com/Adamanr/AiFlow",
      docs: &docs/0
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp package do
    [
      licenses: ["MIT"],
      links: %{"GitHub" => "https://github.com/Adamanr/AiFlow"}
    ]
  end

  defp docs do
    [
      main: "AiFlow",
      logo: "assets/aiflow.png",
      extras: ["README.md"]
    ]
  end

  defp deps do
    [
      {:req, "~> 0.5.14"},
      {:jason, "~> 1.4.4"},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false, warn_if_outdated: true},
      {:makeup_html, ">= 0.0.0", only: :dev, runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:mox, "~> 1.2", only: :test}
    ]
  end
end
