{
  description = "AiFlow - Unified interface for working with various AI models";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        # Elixir and Erlang versions
        elixir = pkgs.beam.packages.erlang_27.elixir_1_18;
        
        # Project dependencies
        projectDeps = with pkgs; [
          elixir
          erlang
          git
          
          # For building native dependencies
          gcc
          gnumake
          pkg-config
        ];
        
        # Development dependencies (Ollama for local AI development)
        devDeps = with pkgs; [
          ollama
        ];
        
      in
      {
        packages.default = pkgs.buildElixirProject {
          pname = "ai_flow";
          version = "0.1.0";
          src = ./.;
          beamPackages = pkgs.beam.packages.erlang_27;
          nativeBuildInputs = with pkgs; [ git ];
        };

        devShells.default = pkgs.mkShell {
          buildInputs = projectDeps ++ devDeps;
          
          shellHook = ''
            echo "🎨 AiFlow Development Environment"
            echo "📚 Elixir $(elixir --version | grep Elixir)"
            echo "🟢 Erlang/OTP $(erl -eval '{ok, Version} = file:read_file(filename:join([code:root_dir(), "releases", erlang:system_info(otp_release), "OTP_VERSION"])), io:fwrite(Version), halt().' -noshell 2>/dev/null)"
            echo "📦 Mix $(mix --version | grep Mix)"
            echo ""
            echo "🚀 Ready to code! Run 'mix deps.get' to get started"
          '';
          
          # Environment variables
          MIX_ENV = "dev";
          LANG = "en_US.UTF-8";
        };
        
        # Test-only environment
        devShells.test = pkgs.mkShell {
          buildInputs = projectDeps;
          MIX_ENV = "test";
        };
      }
    );
}