#!/bin/zsh
# Stopper Rejefangeren og fjerner automatisk start.
# Kalibrering, grænser og log i ~/.rejefangeren bliver liggende (slet mappen, hvis du vil af med dem).
LABEL="dk.rejefangeren.app"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/$LABEL.plist"
rm -rf "$HOME/Library/Application Support/Rejefangeren"
echo "Rejefangeren er afinstalleret."
