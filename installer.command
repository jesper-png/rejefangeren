#!/bin/zsh
# Installerer Rejefangeren, så den starter automatisk ved login og kører i menulinjen (🦐).
# Kør igen efter en opdatering for at installere den nye version.
set -e
SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="$HOME/Library/Application Support/Rejefangeren"
LABEL="dk.rejefangeren.app"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

if [ "$(uname -m)" != "arm64" ]; then
  echo "Rejefangeren kræver en Mac med Apple-chip (M1 eller nyere)."
  exit 1
fi

# Find en Python mellem 3.10 og 3.12 (MediaPipe understøtter ikke ældre)
PY=""
for p in python3.12 python3.11 python3.10 /opt/homebrew/bin/python3.12; do
  if command -v "$p" >/dev/null 2>&1; then PY="$(command -v "$p")"; break; fi
done
if [ -z "$PY" ]; then
  echo "Fandt ingen Python 3.10-3.12. Installér den med Homebrew (https://brew.sh):"
  echo "  brew install python@3.12"
  echo "og kør så installer.command igen."
  exit 1
fi

echo "Kopierer Rejefangeren til $DEST"
mkdir -p "$DEST/models" "$HOME/.rejefangeren" "$HOME/Library/LaunchAgents"
cp "$SRC/rejefangeren.py" "$SRC/requirements.txt" "$DEST/"
if [ ! -f "$DEST/models/pose_landmarker_lite.task" ]; then
  echo "Henter Googles kropssporings-model (ca. 6 MB)"
  curl -sSfL -o "$DEST/models/pose_landmarker_lite.task" "$MODEL_URL"
fi
[ -x "$DEST/.venv/bin/python" ] || "$PY" -m venv "$DEST/.venv"
echo "Installerer biblioteker (første gang tager det et par minutter)"
"$DEST/.venv/bin/python" -m pip install -q --disable-pip-version-check -r "$DEST/requirements.txt"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$DEST/.venv/bin/python</string>
    <string>$DEST/rejefangeren.py</string>
  </array>
  <key>WorkingDirectory</key><string>$DEST</string>
  <key>EnvironmentVariables</key><dict><key>PYTHONUNBUFFERED</key><string>1</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>$HOME/.rejefangeren/log.txt</string>
  <key>StandardErrorPath</key><string>$HOME/.rejefangeren/log.txt</string>
</dict>
</plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
# vent til den gamle udgave er helt stoppet, ellers fejler genstart
for i in {1..20}; do launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || break; sleep 0.5; done
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo ""
echo "Færdig. Rejefangeren kører nu (🦐 i menulinjen) og starter automatisk ved login."
echo "Første gang spørger macOS, om 'Python' må bruge kameraet. Tryk Tillad."
echo "Sid pænt og kig på midten af skærmen de første 3 sekunder, mens den kalibrerer."
