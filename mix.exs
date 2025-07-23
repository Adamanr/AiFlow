defmodule AiFlow.MixProject do
  use Mix.Project

  def project do
    [
      app: :ai_flow,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      description: "A simple Elixir library for interacting with the Ollama API",
      package: package(),
      deps: deps(),

      name: "AiFlow",
      source_url: "https://github.com/Adamanr/AiFlow",
      docs: &docs/0
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp package do
    [
      licenses: ["MIT"],
      links: %{"GitHub" => "https://github.com/yourusername/ollama_ex"}
    ]
  end

  defp docs do
  [
    main: "AiFlow",
    logo: "assets/aiflow.png",
    extras: ["README.md"]
  ]
end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:req, "~> 0.5.14"},
      {:mox, "~> 1.2"},
      {:jason, "~> 1.4.4"},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false, warn_if_outdated: true},
      {:makeup_html, ">= 0.0.0", only: :dev, runtime: false},
      {:bumblebee, "~> 0.5.0"},
      {:nx, "~> 0.7.0"},
      {:axon, "~> 0.6.0"},
      {:exla, "~> 0.7.0", optional: true},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false}
    ]
  end
end
