#!/bin/bash
set -e

# --- Konfiguration ---
ORT_VERSION="1.22.0"
OS="linux"
ARCH="x64"
FILE_NAME="onnxruntime-${OS}-${ARCH}-${ORT_VERSION}.tgz"
DIR_NAME="onnxruntime-${OS}-${ARCH}-${ORT_VERSION}"
DOWNLOAD_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${FILE_NAME}"

# --- Schritt 1: Herunterladen ---
echo "--- Starte Download von ONNX Runtime v${ORT_VERSION} ---"
if [ -f "$FILE_NAME" ]; then
    echo "Datei $FILE_NAME existiert bereits. Überspringe Download."
else
    wget -q --show-progress "$DOWNLOAD_URL"
    echo "Download abgeschlossen."
fi

# --- Schritt 2: Entpacken ---
echo "--- Entpacke Archiv ---"
tar -xzf "$FILE_NAME"

# --- Schritt 3: Installation ---
echo "--- Installiere Bibliotheken (sudo erforderlich) ---"

# In das Verzeichnis wechseln
cd "$DIR_NAME"

# Prüfen, ob Verzeichnisse existieren und kopieren
if [ -d "include" ] && [ -d "lib" ]; then
    echo "Kopiere Header-Dateien nach /usr/local/include/..."
    sudo cp -r include/* /usr/local/include/

    echo "Kopiere Bibliotheken nach /usr/local/lib/..."
    sudo cp -r lib/* /usr/local/lib/
else
    echo "FEHLER: Entpacktes Verzeichnis scheint unvollständig zu sein."
    exit 1
fi

# --- Schritt 4: Cache aktualisieren ---
echo "--- Aktualisiere System-Linker-Cache ---"
sudo ldconfig

cd ..
rm -rf "$DIR_NAME" "$FILE_NAME"

echo "--- Installation von ONNX Runtime v${ORT_VERSION} erfolgreich abgeschlossen! ---"