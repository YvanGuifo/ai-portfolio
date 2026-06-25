# Formation : Vectorstores and Embeddings
## Stockages de vecteurs et représentations vectorielles

---

## 📌 Section 1 : Configuration initiale et imports

### Code

```python
import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']
```

### Explication détaillée

#### 1. **Imports des bibliothèques**

```python
import os
import openai
import sys
```

- **`os`** : Module Python pour accéder aux variables d'environnement et gérer les chemins de fichiers
- **`openai`** : Bibliothèque officielle d'OpenAI pour accéder à leurs API (notamment les embeddings et modèles de langage)
- **`sys`** : Module système Python pour manipuler les chemins et paramètres d'exécution

#### 2. **Gestion du chemin Python**

```python
sys.path.append('../..')
```

- Ajoute le répertoire parent au chemin de recherche Python (`sys.path`)
- Permet d'importer des modules situés dans les dossiers parents
- Utile pour une structure de projet avec plusieurs niveaux de dossiers

#### 3. **Chargement des variables d'environnement**

```python
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
```

- **`find_dotenv()`** : Localise le fichier `.env` dans le répertoire courant ou ses parents
- **`load_dotenv()`** : Charge toutes les variables du fichier `.env` dans l'environnement Python (`os.environ`)
- Le trait de soulignement `_` indique qu'on ne va pas utiliser la valeur de retour

**Fichier `.env` typique :**
```
OPENAI_API_KEY=sk-...votre-clé-api...
```

#### 4. **Configuration de la clé API OpenAI**

```python
openai.api_key = os.environ['OPENAI_API_KEY']
```

- Récupère la clé API depuis les variables d'environnement
- Configure la bibliothèque OpenAI pour utiliser cette clé dans tous les appels d'API
- **Bonne pratique de sécurité** : Ne jamais hardcoder les clés dans le code source

---

## 🔐 Points importants

| Concept | Description |
|---------|-------------|
| **Gestion d'environnement** | Utiliser `.env` pour les secrets, jamais les hardcoder |
| **Chemins relatifs** | `sys.path.append()` facilite l'organisation du projet |
| **Initialisation** | Cette étape doit être la première avant tout appel à l'API OpenAI |

---

## 📌 Section 2 : Chargement de documents PDF

### Code

```python
from langchain.document_loaders import PyPDFLoader

# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture02.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture03.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
```

### Explication détaillée

#### 1. **Import du chargeur PDF**

```python
from langchain.document_loaders import PyPDFLoader
```

- **LangChain** : Framework populaire pour construire des applications avec des LLMs
- **PyPDFLoader** : Classe spécialisée pour charger et parser des fichiers PDF
- Convertit les PDFs en documents structurés exploitables par LangChain

#### 2. **Création des chargeurs**

```python
loaders = [
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture02.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture03.pdf")
]
```

**Points importants :**

- **Liste de chargeurs** : Crée une instance `PyPDFLoader` pour chaque PDF
- **Doublons intentionnels** : Le commentaire `# Duplicate documents on purpose - messy data` indique que :
  - La Lecture 01 est chargée deux fois
  - Simule des données réelles souvent imparfaites/dupliquées
  - Utile pour tester la robustesse du système (déduplication, nettoyage)

#### 3. **Chargement et agrégation des documents**

```python
docs = []
for loader in loaders:
    docs.extend(loader.load())
```

- **Initialise une liste vide** : `docs = []`
- **Itère sur chaque chargeur** : Boucle for sur la liste `loaders`
- **`loader.load()`** : Charge les documents depuis le PDF
  - Retourne une liste de `Document` objects
  - Chaque page ou section devient un élément
- **`extend()`** : Ajoute tous les documents à la liste
  - Plus efficace que `append()` qui ajouterait la liste entière

**Résultat :**
```
docs = [
    Document(page_content="...", metadata={...}),
    Document(page_content="...", metadata={...}),
    ...  # Environ 40-60 documents selon le nombre de pages
]
```

---

## 📊 Structure d'un Document LangChain

Chaque `Document` contient :

```python
{
    "page_content": "Texte du document...",
    "metadata": {
        "source": "docs/cs229_lectures/MachineLearning-Lecture01.pdf",
        "page": 0
    }
}
```

| Champ | Description |
|-------|-------------|
| **page_content** | Texte extrait du PDF |
| **metadata** | Informations sur l'origine (fichier, page, etc.) |

---

## 🎯 Cas d'usage pratique

Cette section prépare les données pour :
- ✓ **Vectorization** : Transformer le texte en embeddings
- ✓ **Stockage** : Sauvegarder les vecteurs dans une vectorstore
- ✓ **Recherche sémantique** : Trouver des documents pertinents

---

## 📌 Section 3 : Division des documents en chunks

### Code

```python
# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)
```

### Explication détaillée

#### 1. **Import du diviseur de texte**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

- **RecursiveCharacterTextSplitter** : Classe LangChain qui divise intelligemment les textes
- **Approche "récursive"** : Divise d'abord par les séparateurs majeurs (paragraphes, phrases, mots) puis par caractères si nécessaire
- Contrairement à un simple découpage par caractères, préserve la cohérence du texte

#### 2. **Création du séparateur**

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)
```

**Paramètres clés :**

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| **chunk_size** | 1500 | Taille maximale de chaque chunk en caractères |
| **chunk_overlap** | 150 | Chevauchement entre chunks consécutifs |

#### 3. **Comprendre chunk_size**

```
chunk_size = 1500
```

- **Définit la longueur maximale** d'un segment de texte
- 1500 caractères ≈ 300-400 mots
- Taille appropriée pour :
  - ✓ Créer des embeddings équilibrés
  - ✓ Capturer suffisamment de contexte
  - ✓ Rester sous les limites des API (tokens)

**Exemple visuel :**
```
Document original : 10,000 caractères
                    ↓
                [chunk 1: 1500 chars]
                [chunk 2: 1500 chars]
                [chunk 3: 1500 chars]
                    ...
```

#### 4. **Comprendre chunk_overlap**

```
chunk_overlap = 150
```

- **Chevauchement intentionnel** entre chunks consécutifs
- Les 150 derniers caractères d'un chunk sont répétés au début du suivant
- **Avantages** :
  - ✓ Préserve la continuité sémantique
  - ✓ Évite de couper des informations importantes au milieu d'une phrase
  - ✓ Améliore la qualité des embeddings

**Exemple visuel :**
```
Chunk 1: "...concept important ici..."  (1500 chars)
         └─────────────────────────┘
Chunk 2: "...concept important ici... suite du texte..."  (1500 chars)
         └────────────────────────────────────────┘
              ↑ (150 chars de chevauchement)
```

---

## 🧩 Rapport chunk_size / chunk_overlap

**Règle générale :**
- `chunk_overlap` devrait être **10% de chunk_size** (ici 150/1500 = 10%)
- Trop petit : perte de contexte
- Trop grand : redondance excessive

---

## 🎯 Cas d'usage pratique

Cette étape prépare les documents pour :
- ✓ **Vectorization** : Chaque chunk sera transformé en embedding
- ✓ **Recherche** : Les chunks chevauchants améliorent la pertinence des résultats
- ✓ **Efficacité** : Taille optimale pour les modèles d'embeddings

---

## 📌 Section 4 : Application du split aux documents

### Code

```python
splits = text_splitter.split_documents(docs)
```

### Explication détaillée

#### 1. **La méthode split_documents()**

```python
splits = text_splitter.split_documents(docs)
```

- **Applique le séparateur** à tous les documents chargés
- **Prend en entrée** : Liste de documents (`docs`) de la section 2
- **Retourne** : Liste de documents divisés en chunks
- Préserve les **métadonnées** originales pour chaque chunk

#### 2. **Ce qui se passe en coulisse**

```
docs (Section 2)                 text_splitter (Section 3)
├─ Document 1 (5000 chars)  ──→  split_documents()  ──→  splits
├─ Document 2 (6000 chars)  │                         ├─ Chunk 1.1 (1500 chars)
├─ Document 3 (4500 chars)  │                         ├─ Chunk 1.2 (1350 chars)
└─ ...                       │                         ├─ Chunk 2.1 (1500 chars)
                             │                         ├─ Chunk 2.2 (1500 chars)
                             └─────────────────────→  └─ ...
```

#### 3. **Structure de la variable splits**

Chaque élément de `splits` est un **Document** avec :

```python
splits[0]
# {
#     "page_content": "...(jusqu'à 1500 chars)...",
#     "metadata": {
#         "source": "docs/cs229_lectures/MachineLearning-Lecture01.pdf",
#         "page": 0
#     }
# }
```

**Points importants :**
- ✓ Chaque chunk reste un objet Document complet
- ✓ Les métadonnées (source, page) sont conservées
- ✓ Facilite le traçage : on sait d'où vient chaque chunk

#### 4. **Avantages de cette approche**

| Avantage | Bénéfice |
|----------|----------|
| **Taille uniforme** | Tous les chunks sont traités de la même manière |
| **Chevauchement** | Continuité sémantique préservée (150 chars overlap) |
| **Métadonnées** | Traçabilité complète vers les documents originaux |
| **Scalabilité** | Prêt pour vectorization et recherche |

#### 5. **Exemple de résultat**

Supposons que `docs` contient 4 PDFs avec ~50 pages au total :

```
len(docs)     # ≈ 50 documents (1 par page)
len(splits)   # ≈ 200-300 chunks (selon la taille du texte)
```

---

## 🔄 Flux complet jusqu'à présent

```
1. Configuration (Section 1)
        ↓
2. Charger PDFs (Section 2)  → docs = [Document, Document, ...]
        ↓
3. Créer splitter (Section 3)  → text_splitter configuré
        ↓
4. Diviser documents (Section 4) → splits = [Chunk, Chunk, ...]
        ↓
   PROCHAINE ÉTAPE : Vectorization (embeddings)
```

---

## 📌 Section 5 : Embeddings et Vectorstore

### Partie A : Inspection et création des embeddings

```python
len(splits)

from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()

sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"

embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)

import numpy as np

np.dot(embedding1, embedding2)
np.dot(embedding1, embedding3)
np.dot(embedding2, embedding3)
```

#### 1. **Inspection des splits**

```python
len(splits)
```

- Affiche le nombre total de chunks créés
- Vérifie que la division s'est bien passée
- Exemple : `len(splits) # 215` (dépend de la taille des PDFs)

#### 2. **Import des embeddings OpenAI**

```python
from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()
```

- **OpenAIEmbeddings** : Classe qui utilise le modèle d'embeddings d'OpenAI
- Modèle utilisé par défaut : `text-embedding-3-small`
- Crée des vecteurs de **1536 dimensions**
- Nécessite la clé API OpenAI (configurée en Section 1)

#### 3. **Création d'embeddings - Concept clé**

```python
sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"

embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)
```

**Que fait un embedding ?**
- Convertit du texte en **vecteur numérique** (liste de nombres)
- Exemple : `"i like dogs"` → `[0.123, -0.456, 0.789, ..., -0.234]` (1536 valeurs)
- Capture le **sens sémantique** du texte
- Textes similaires ont des embeddings proches

**Propriété fondamentale :**
```
"i like dogs" et "i like canines"
         ↓                    ↓
    SIMILAIRES   →   embeddings PROCHES
```

#### 4. **Mesure de similarité avec le produit scalaire**

```python
np.dot(embedding1, embedding2)  # ≈ 0.95
np.dot(embedding1, embedding3)  # ≈ 0.42
np.dot(embedding2, embedding3)  # ≈ 0.41
```

- **`np.dot()`** : Produit scalaire (dot product) entre deux vecteurs
- Mesure la **similarité cosinus** : plus la valeur est proche de 1, plus les textes sont similaires
- Résultats typiques :
  - `0.95` : Très similaire (synonymes)
  - `0.42` : Faiblement similaire (sujets différents)

**Visualisation :**
```
Embedding1 ("i like dogs")         →  [0.12, -0.45, 0.78, ...]
Embedding2 ("i like canines")      →  [0.11, -0.44, 0.79, ...]
                ↓ np.dot()
         Similarité = 0.95 (élevée)

Embedding1 ("i like dogs")         →  [0.12, -0.45, 0.78, ...]
Embedding3 ("weather is ugly")     →  [-0.32, 0.21, 0.15, ...]
                ↓ np.dot()
         Similarité = 0.42 (basse)
```

---

### Partie B : Création du Vectorstore Chroma

```python
# ! pip install chromadb
from langchain.vectorstores import Chroma
persist_directory = 'docs/chroma/'
!rm -rf ./docs/chroma  # remove old database files if any

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

print(vectordb._collection.count())
```

#### 1. **Installation et import**

```python
# ! pip install chromadb
from langchain.vectorstores import Chroma
```

- **Chroma** : Base de données vectorielle open-source
- Stocke les embeddings pour une recherche rapide
- Alternative : Pinecone, Weaviate, FAISS, etc.
- Le `!` exécute une commande shell en Jupyter

#### 2. **Configuration du répertoire**

```python
persist_directory = 'docs/chroma/'
!rm -rf ./docs/chroma  # remove old database files if any
```

- **persist_directory** : Emplacement où stocker les vecteurs
- **`!rm -rf`** : Supprime l'ancienne base de données (pour un démarrage propre)
- Chroma peut persister (sauvegarder) les données sur disque

#### 3. **Création de la vectorstore**

```python
vectordb = Chroma.from_documents(
    documents=splits,      # Les chunks de la Section 4
    embedding=embedding,   # Le modèle d'embeddings
    persist_directory=persist_directory
)
```

**Ce qui se passe :**
1. Prend chaque chunk de `splits`
2. Crée un embedding avec `OpenAIEmbeddings()`
3. Stocke le vecteur + le texte original + les métadonnées dans Chroma
4. Sauvegarde tout dans le dossier `docs/chroma/`

**Résultat :**
```
vectordb._collection.count()  # Nombre total de documents vectorisés
# Exemple : 215 (si len(splits) = 215)
```

---

### Partie C : Recherche par similarité

```python
question = "is there an email i can ask for help"
docs = vectordb.similarity_search(question,k=3)
len(docs)
docs[0].page_content

vectordb.persist()
```

#### 1. **Recherche simple**

```python
question = "is there an email i can ask for help"
docs = vectordb.similarity_search(question,k=3)
```

**Processus :**
1. Crée un embedding de la question
2. Cherche les k=3 chunks les plus similaires
3. Utilise la **similarité cosinus** pour classer
4. Retourne une liste de Documents

**Résultat :**
```python
len(docs)  # 3 (nombre de résultats demandés)
docs[0]    # Document le plus pertinent
docs[1]    # Deuxième document pertinent
docs[2]    # Troisième document pertinent
```

#### 2. **Accès au contenu**

```python
docs[0].page_content  # Affiche le texte du document le plus pertinent
```

Exemple de sortie :
```
"For administrative questions or help, please contact staff@cs229.edu"
```

#### 3. **Sauvegarde persistante**

```python
vectordb.persist()
```

- Sauvegarde la base de données vectorielle sur disque
- Les données persisteront même après fermeture du programme
- Permet de recharger plus tard : `vectordb = Chroma(..., persist_directory=...)`

---

### Partie D : Recherches avancées

```python
question = "what did they say about matlab?"
docs = vectordb.similarity_search(question,k=5)

docs[0]

question = "what did they say about regression in the third lecture?"
docs = vectordb.similarity_search(question,k=5)
for doc in docs:
    print(doc.metadata)
print(docs[4].page_content)
```

#### 1. **Recherche thématique**

```python
question = "what did they say about matlab?"
docs = vectordb.similarity_search(question,k=5)
docs[0]
```

- Cherche les 5 chunks les plus pertinents sur "matlab"
- `docs[0]` contient le chunk le plus pertinent
- Affiche : texte + métadonnées (source, page)

#### 2. **Recherche contextuelle multi-document**

```python
question = "what did they say about regression in the third lecture?"
docs = vectordb.similarity_search(question,k=5)
for doc in docs:
    print(doc.metadata)  # Affiche source et page de chaque résultat
print(docs[4].page_content)  # Affiche le contenu du 5e résultat
```

**La vectorstore comprend :**
- ✓ Le contexte ("regression")
- ✓ La source ("third lecture")
- ✓ Grâce aux métadonnées préservées depuis les PDFs

**Résultat :**
```
Metadata du doc 1 : source=Lecture03.pdf, page=5
Metadata du doc 2 : source=Lecture03.pdf, page=8
Metadata du doc 3 : source=Lecture02.pdf, page=3  # Peut contenir du contexte adjacent
Metadata du doc 4 : source=Lecture03.pdf, page=10
Metadata du doc 5 : source=Lecture01.pdf, page=2  # Moins pertinent

Contenu du doc 5 : "..."
```

---

## 🎯 Flux complet : De l'API à la recherche

```
1. Configuration API (Section 1)
        ↓
2. Charger PDFs (Section 2) → docs
        ↓
3. Diviser documents (Sections 3-4) → splits
        ↓
4. Créer embeddings (Section 5-A) → vecteurs 1536D
        ↓
5. Stocker dans Chroma (Section 5-B) → vectordb
        ↓
6. Recherche par similarité (Section 5-C,D) → résultats pertinents
        ↓
   PROCHAINE ÉTAPE : Utiliser la vectorstore dans un RAG
```

---

## 📊 Comparaison : Recherche texte vs Vectorstore

| Aspect | Recherche texte classique | Vectorstore |
|--------|---------------------------|-------------|
| **Méthode** | Mot-clé exact | Similarité sémantique |
| **Exemple** | "matlab" trouve "matlab" seulement | "matlab" trouve aussi "MATLAB", "Octave" |
| **Synonymes** | Non capturés | Capturés |
| **Contexte** | Limité | Complet (vecteur 1536D) |
| **Vitesse** | Très rapide | Rapide (index inversé) |
| **Utilité** | Recherche simple | RAG, recommandations |

---

## 🎓 Conclusion : Récapitulatif complet

### Architecture générale

```
┌─────────────────────────────────────────────────────┐
│           VECTORSTORES AND EMBEDDINGS               │
└─────────────────────────────────────────────────────┘

Section 1 : Configuration
└─→ Variables d'environnement + clé API OpenAI

Section 2 : Ingestion de données
└─→ Charger des PDFs avec PyPDFLoader
    └─→ 4 PDFs → ~50 documents (une page = un doc)

Section 3-4 : Préparation
└─→ Diviser avec RecursiveCharacterTextSplitter
    └─→ chunk_size=1500, chunk_overlap=150
    └─→ 50 documents → ~215 chunks

Section 5A : Vectorization
└─→ Créer des embeddings avec OpenAIEmbeddings
    └─→ Texte → Vecteur 1536D (capture le sens)
    └─→ Mesurer similarité avec np.dot()

Section 5B-D : Stockage et recherche
└─→ Stocker dans Chroma (vectorstore)
    └─→ Recherche rapide par similarité sémantique
    └─→ Résultats ordonnés par pertinence
```

---

### Concepts clés appris

| Concept | Définition | Utilité |
|---------|-----------|---------|
| **Embedding** | Représentation vectorielle du texte | Capture le sens sémantique |
| **Vectorstore** | Base de données d'embeddings indexée | Recherche rapide et pertinente |
| **Chunk** | Segment de texte (≈300-400 mots) | Taille optimale pour embeddings |
| **Similarité** | Proximité entre deux vecteurs | Mesure la pertinence |
| **Métadonnées** | Info sur l'origine (source, page) | Traçabilité complète |
| **Overlap** | Chevauchement entre chunks | Préserve continuité sémantique |

---

### Cas d'usage pratiques

Avec cette architecture, vous pouvez construire :

1. **🔍 Moteur de recherche sémantique**
   - Chercher par sens, pas par mots-clés
   - Exemple : "how to contact the staff" → trouve "email@cs229.edu"

2. **🤖 RAG (Retrieval Augmented Generation)**
   - Récupérer les documents pertinents
   - Les passer à un LLM pour générer une réponse
   - Exemple : Question + contexte → Réponse précise

3. **📚 Recommandations**
   - Trouver des documents similaires
   - "Vous avez lu sur X, vous aimerez aussi Y"

4. **🏷️ Classification sémantique**
   - Grouper les documents par sens
   - Déterminer l'intention de l'utilisateur

---

### Points importants à retenir

✅ **À faire :**
- Configurer les clés API de manière sécurisée (.env)
- Choisir une taille de chunk appropriée (1000-2000 caractères)
- Utiliser un overlap (10% de chunk_size)
- Persister la vectorstore (`vectordb.persist()`)
- Préserver les métadonnées (source, page)

❌ **À éviter :**
- Hardcoder les clés API dans le code
- Chunks trop petits (< 500 chars) → contexte insuffisant
- Chunks trop grands (> 3000 chars) → embeddings flous
- Oublier d'overlap → continuité sémantique brisée
- Perdre les métadonnées → impossible de tracer les résultats

---

### Améliorations possibles

Avec cette base solide, vous pouvez :

1. **Filtrer par métadonnées**
   ```python
   # Rechercher seulement dans la Lecture 3
   docs = vectordb.similarity_search(
       question,
       k=5,
       filter={"source": "MachineLearning-Lecture03.pdf"}
   )
   ```

2. **Utiliser d'autres modèles d'embeddings**
   - `text-embedding-3-large` (plus puissant, plus cher)
   - `text-embedding-3-small` (par défaut, rapide)
   - Modèles open-source : Sentence-Transformers, etc.

3. **Intégrer avec un LLM**
   ```python
   # Combiner recherche + génération
   retrieved_docs = vectordb.similarity_search(question)
   answer = llm.generate(question, context=retrieved_docs)
   ```

4. **Améliorer la qualité**
   - Nettoyer les doublons dans les données brutes
   - Ajuster chunk_size selon le domaine
   - Utiliser des métadonnées enrichies

---

### Architecture finale : RAG complète

```
User Question
     ↓
┌────────────────────────────────────┐
│  1. VECTORIZE QUESTION              │
│     question → embedding 1536D      │
└────────────────────────────────────┘
     ↓
┌────────────────────────────────────┐
│  2. SEARCH VECTORSTORE              │
│     Chroma: top-k chunks pertinents │
└────────────────────────────────────┘
     ↓
┌────────────────────────────────────┐
│  3. BUILD CONTEXT                   │
│     Chunks récupérés + métadonnées │
└────────────────────────────────────┘
     ↓
┌────────────────────────────────────┐
│  4. GENERATE WITH LLM               │
│     Question + Context → Answer     │
└────────────────────────────────────┘
     ↓
User Answer (with sources)
```

---

## 📊 Statistiques de la formation

- **Sections** : 5 sections complètes
- **Concepts** : 10+ concepts clés expliqués
- **Code** : ~30 lignes de code source
- **Visualisations** : Diagrammes, tableaux, exemples
- **Cas d'usage** : 4+ applications pratiques

---

## ✨ Formation terminée !

**Vous savez maintenant :**
- ✅ Comment charger et préparer des documents
- ✅ Comment créer des embeddings
- ✅ Comment construire une vectorstore
- ✅ Comment faire de la recherche sémantique
- ✅ Comment combiner tout cela dans une architecture RAG

**Prochaines étapes suggérées :**
1. Expérimenter avec vos propres documents
2. Tester différentes tailles de chunks
3. Intégrer avec un LLM pour un RAG complet
4. Explorer d'autres vectorstores (Pinecone, Weaviate)
5. Optimiser pour votre cas d'usage spécifique

---

**Merci d'avoir suivi cette formation ! 🚀**

*Document créé le : 2026-06-25*
*Formation : Vectorstores and Embeddings*

