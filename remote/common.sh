#!/usr/bin/env bash

activate_pbp_if_needed() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "pbp" ]]; then
    return 0
  fi

  if [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "/root/miniconda3/etc/profile.d/conda.sh"
  elif [[ -x "/root/miniconda3/bin/conda" ]]; then
    eval "$(/root/miniconda3/bin/conda shell.bash hook)"
  else
    echo "Could not find conda initialization under /root/miniconda3" >&2
    return 1
  fi

  conda activate pbp
}

