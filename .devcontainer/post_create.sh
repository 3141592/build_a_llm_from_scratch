#!/usr/bin/env bash
set -e

cp /workspaces/build_a_llm_from_scratch/.devcontainer/.vimrc /root/.vimrc

apt-get update
apt-get install -y tmux

pip install -e /workspaces/ai_shared_utilities
pip install black

ln -sf /usr/bin/python3 /usr/bin/python

cat > /root/.bashrc_devcontainer <<'EOF'
# ---- Bash history ----

export HISTSIZE=100000
export HISTFILESIZE=200000
export HISTCONTROL=ignoredups:erasedups

shopt -s histappend

# Save history immediately and import commands from other shells
export PROMPT_COMMAND='history -a; history -n'

# ---- Up/down arrow prefix search ----

bind '"\e[A": history-search-backward'
bind '"\e[B": history-search-forward'
EOF

grep -qxF 'source /root/.bashrc_devcontainer' /root/.bashrc \
      || echo 'source /root/.bashrc_devcontainer' >> /root/.bashrc

