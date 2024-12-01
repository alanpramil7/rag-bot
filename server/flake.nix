{
  description = "RAG Bot project with Python and dependencies";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          # Dependencies from requirements.txt
          # Add these based on your requirements.txt content
          # For example:
          # transformers
          # torch
          # pypdf
          # langchain
          # Add the specific packages you're using
        ]);
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.git
          ];

          shellHook = ''
            echo "🤖 RAG Bot Development Environment 🚀"
            echo "Python version: $(python --version)"
            python -m venv .venv
            source .venv/bin/activate
            pip install -r ./server/requirements.txt
          '';
        };

        # Optional: define a package if you want to build the project
        # packages.default = pkgs.stdenv.mkDerivation {
        #   name = "rag-bot";
        #   src = ./.;

        #   buildInputs = [ pythonEnv ];

        #   buildPhase = ''
        #     mkdir -p $out/bin
        #     cp main.py $out/bin/rag-bot
        #     chmod +x $out/bin/rag-bot
        #   '';
        # };
      }
    );
}
