#!/bin/bash
# Setup script — AI Portfolio
echo "🚀 Configuration de l'environnement AI Portfolio..."

# Vérifier Python
python3 --version || { echo "❌ Python 3 requis"; exit 1; }

# Créer l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# Copier .env
if [ ! -f .env ]; then
  cp .env.example .env
  echo "📝 Fichier .env créé — remplir les clés API"
fi

echo ""
echo "✅ Environnement configuré !"
echo "   Activer : source .venv/bin/activate"
echo "   Jupyter : jupyter lab"
echo ""
echo "🎯 Première action : ouvrir deeplearning.ai/courses"
echo "   Cours : How LLMs Work (1h)"
