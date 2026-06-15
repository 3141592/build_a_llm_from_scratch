#!/usr/bin/env bash
set -e

cp /workspaces/build_a_llm_from_scratch/.devcontainer/.vimrc /root/.vimrc

apt-get update
apt-get install -y tmux

pip install -e /workspaces/ai_shared_utilities
