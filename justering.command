#!/bin/zsh
# Åbner justeringsvinduet (samme som "Justér grænser..." i 🦐-menuen)
"$HOME/Library/Application Support/Rejefangeren/.venv/bin/python" "$HOME/Library/Application Support/Rejefangeren/rejefangeren.py" --preview || read "?Noget gik galt. Tryk Enter for at lukke."
