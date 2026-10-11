{
  description = "take-my-layer: read decisions out of local Qwen instead of generating them (AnyJev on MLX)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            # Native arm64 Python (mlx-lm 0.32.0 needs >= 3.11; MLX needs macOS >= 14)
            python312
            # Installs the Metal-enabled mlx/mlx-lm PyPI wheels.
            # NOTE: do NOT use python312Packages.mlx-lm — nixpkgs builds MLX
            # without Metal (CPU-only), useless for inference.
            uv

            git
          ];

          shellHook = ''
            export VIRTUAL_ENV="$PWD/.venv"
            if [ ! -d "$VIRTUAL_ENV" ]; then
              uv venv --python "$(command -v python3)" "$VIRTUAL_ENV"
            fi
            export PATH="$VIRTUAL_ENV/bin:$PATH"

            # keep everything (wheels, HF model cache, XDG caches) inside the project
            export XDG_CACHE_HOME="$PWD/.cache"
            export UV_CACHE_DIR="$PWD/.uv-cache"
            export HF_HOME="$PWD/.hf"
            export HF_HUB_CACHE="$HF_HOME/hub"

            # uv resolves from cache, so this is fast and keeps deps fresh
            uv pip install -q -r requirements.txt

            echo "Shell ready."
            echo "Python: $($VIRTUAL_ENV/bin/python --version) ($($VIRTUAL_ENV/bin/python -c 'import platform; print(platform.machine())'))"
            echo "MLX Metal: $($VIRTUAL_ENV/bin/python -c 'import mlx.core as mx; print(mx.metal.is_available())' 2>/dev/null || echo 'not installed yet')"
            echo "HF cache: $HF_HUB_CACHE"
          '';
        };
      });
}
