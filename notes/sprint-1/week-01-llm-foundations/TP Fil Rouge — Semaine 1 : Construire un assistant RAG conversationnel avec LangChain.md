# 🧪 TP Fil Rouge — Semaine 1 : Construire un assistant RAG conversationnel avec LangChain

> **Type** : Travail Pratique guidé · **Projet fil rouge Sprint 1**  
> **Durée** : 2 heures (8 étapes calibrées)  
> **Niveau** : M1 / M2 ingénieurs · Prérequis : Python 3.10+, bases d'API REST  
> **Stack** : Python, LangChain ≥ 0.1, OpenAI GPT-3.5-turbo, Chroma  
> **Auteur** : Dr. Yvan GUIFO FODJO — EFREI Paris · 2025–2026

---

## Table des matières

- [Contexte et mise en situation](#contexte-et-mise-en-situation)
- [Objectifs pédagogiques](#objectifs-pédagogiques)
- [Architecture cible](#architecture-cible)
- [Prérequis et installation](#étape-0--prérequis-et-installation-10-min)
- [Étape 1 — Charger les documents](#étape-1--charger-les-documents-15-min)
- [Étape 2 — Découper en chunks](#étape-2--découper-en-chunks-15-min)
- [Étape 3 — Créer le vectorstore](#étape-3--créer-le-vectorstore-15-min)
- [Étape 4 — Retrieval basique](#étape-4--retrieval-basique-10-min)
- [Étape 5 — Retrieval avancé avec MMR](#étape-5--retrieval-avancé-avec-mmr-10-min)
- [Étape 6 — Pipeline Question Answering](#étape-6--pipeline-question-answering-15-min)
- [Étape 7 — Chat conversationnel avec mémoire](#étape-7--chat-conversationnel-avec-mémoire-15-min)
- [Étape 8 — Analyse critique et extension](#étape-8--analyse-critique-et-extension-15-min)
- [Grille d'auto-évaluation](#grille-dauto-évaluation)
- [Références](#références)

---

## Contexte et mise en situation

### Le scénario

Vous êtes ingénieur IA dans une startup EdTech. Un professeur vous confie **trois supports de cours en PDF** (slides de machine learning de Stanford, disponibles en ligne) et vous demande de construire un **assistant conversationnel** capable de répondre aux questions des étudiants à partir de ces documents — sans halluciner et en citant ses sources.

Les étudiants doivent pouvoir poser des questions de suivi naturelles, comme :

- *"Quels sont les prérequis mathématiques du cours ?"*
- *"Et pourquoi sont-ils nécessaires ?"* ← question de suivi anaphori­que

Votre mission : **construire un pipeline RAG complet** depuis le chargement des PDF jusqu'au chatbot conversationnel, en passant par l'indexation vectorielle et le retrieval avancé.

### Pourquoi ce projet ?

Ce TP consolide **l'intégralité** du cours *LangChain: Chat with Your Data* (Harrison Chase, DeepLearning.AI) en une seule session pratique, en suivant le même pipeline de 6 étapes dans un contexte réaliste.

```
PDF du cours
    │
    ▼  Étape 1 : Loading
Documents LangChain (page_content + metadata)
    │
    ▼  Étape 2 : Splitting
Chunks sémantiques
    │
    ▼  Étape 3 : Vectorstore + Embedding
Index Chroma persisté sur disque
    │
    ▼  Étape 4-5 : Retrieval (similarité → MMR)
Top-k chunks pertinents et diversifiés
    │
    ▼  Étape 6 : Question Answering
Réponse + sources citées (RetrievalQA)
    │
    ▼  Étape 7 : Chat avec mémoire
Chatbot conversationnel multi-tours (ConversationalRetrievalChain)
    │
    ▼  Étape 8 : Analyse critique
Bilan, limites, pistes d'amélioration
```

---

## Objectifs pédagogiques

À la fin de ce TP, vous serez capable de (taxonomie de Bloom) :

| Niveau | Objectif |
|---|---|
| **Comprendre** | Expliquer le rôle de chaque étape du pipeline RAG |
| **Appliquer** | Implémenter les 6 étapes avec LangChain et Python |
| **Analyser** | Comparer la similarité cosinus simple et MMR sur un même corpus |
| **Analyser** | Diagnostiquer l'échec de `RetrievalQA` sur les questions de suivi |
| **Évaluer** | Interpréter l'impact de `chunk_size` et `chunk_overlap` sur la qualité |
| **Créer** | Assembler un chatbot RAG conversationnel fonctionnel de bout en bout |

---

## Architecture cible

```
┌──────────────────────────────────────────────────────────────────┐
│                     PHASE HORS-LIGNE                             │
│                                                                  │
│  PDF ──► PyPDFLoader ──► [Document] ──► RecursiveCharSplitter   │
│                                               │                  │
│                                         [Chunks]                 │
│                                               │                  │
│                                     OpenAIEmbeddings             │
│                                               │                  │
│                                     Chroma (persist)             │
└───────────────────────────────────────────────┼──────────────────┘
                                                │
┌───────────────────────────────────────────────▼──────────────────┐
│                      PHASE EN LIGNE                              │
│                                                                  │
│  Question ──► Retriever (MMR) ──► Chunks pertinents             │
│      +                                   │                       │
│  Chat History ──► LLM (condensation) ──► Question autonome      │
│                                          │                       │
│                                LLM (génération, T=0)            │
│                                          │                       │
│                              Réponse + source_documents          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Étape 0 — Prérequis et installation (10 min)

### 0.1 Structure du projet

Créez l'arborescence suivante avant de commencer :

```bash
mkdir -p tp_rag/{docs/pdf,chroma_db,src}
cd tp_rag
touch src/pipeline.py
touch .env
```

```
tp_rag/
├── docs/
│   └── pdf/          ← vos PDFs ici
├── chroma_db/         ← base vectorielle persistée (auto-créé)
├── src/
│   └── pipeline.py   ← votre code
└── .env               ← clé API (jamais commitée)
```

### 0.2 Installation des dépendances

```bash
pip install langchain langchain-community langchain-openai \
            chromadb pypdf python-dotenv openai
```

> **Vérification** : `python -c "import langchain; print(langchain.__version__)"`  
> Version attendue : ≥ 0.1.x

### 0.3 Clé API OpenAI

Dans le fichier `.env` :

```bash
OPENAI_API_KEY=sk-...votre_clé_ici...
```

> ⚠️ **Ne jamais** versionner `.env`. Ajoutez-le à `.gitignore` immédiatement.

```bash
echo ".env" >> .gitignore
```

### 0.4 Documents sources

Téléchargez les PDF de démonstration (cours ML de Stanford, libres de droit) :

```bash
# Option A : télécharger depuis le web
curl -L "https://see.stanford.edu/materials/aimlcs229/MachineLearning-Lecture01.pdf" \
     -o docs/pdf/ML-Lecture01.pdf

curl -L "https://see.stanford.edu/materials/aimlcs229/MachineLearning-Lecture02.pdf" \
     -o docs/pdf/ML-Lecture02.pdf
```

> **Alternative** : si ces URLs ne sont plus disponibles, utilisez **n'importe quel PDF de cours** que vous avez (syllabus, notes de cours). L'exercice fonctionne avec tout corpus de texte technique.

---

## Étape 1 — Charger les documents (15 min)

> **Objectif** : transformer des PDF bruts en objets `Document` LangChain standardisés.

### 1.1 Code — Chargement multi-PDF

Créez `src/pipeline.py` et ajoutez :

```python
# src/pipeline.py
# ─────────────────────────────────────────────────────────────
# ÉTAPE 1 : DOCUMENT LOADING
# ─────────────────────────────────────────────────────────────

import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()  # charge OPENAI_API_KEY depuis .env

PDF_DIR = "docs/pdf/"

def load_documents(pdf_dir: str) -> list:
    """Charge tous les PDF d'un répertoire en objets Document LangChain."""
    all_docs = []
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"Aucun PDF trouvé dans {pdf_dir}")

    for pdf_path in sorted(pdf_files):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        all_docs.extend(docs)
        print(f"  ✓ {os.path.basename(pdf_path)} → {len(docs)} pages chargées")

    return all_docs


# ── Test de l'étape 1 ────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== ÉTAPE 1 : LOADING ===")
    documents = load_documents(PDF_DIR)

    print(f"\nTotal : {len(documents)} documents chargés")

    # Inspecter le premier document
    doc0 = documents[0]
    print(f"\n--- Premier document ---")
    print(f"Type       : {type(doc0)}")
    print(f"Métadonnées: {doc0.metadata}")
    print(f"Contenu    : {doc0.page_content[:300]}...")
    print(f"Longueur   : {len(doc0.page_content)} caractères")
```

### 1.2 Exécution et vérification

```bash
python src/pipeline.py
```

**Sortie attendue** :
```
=== ÉTAPE 1 : LOADING ===
  ✓ ML-Lecture01.pdf → 22 pages chargées
  ✓ ML-Lecture02.pdf → 20 pages chargées

Total : 42 documents chargés

--- Premier document ---
Type       : <class 'langchain_core.documents.base.Document'>
Métadonnées: {'source': 'docs/pdf/ML-Lecture01.pdf', 'page': 0}
Contenu    : Machine Learning...
Longueur   : 1847 caractères
```

### ✅ Point de vérification 1

Avant de continuer, vérifiez :

- [ ] Tous les PDF sont chargés sans erreur
- [ ] Chaque `Document` a un champ `metadata` avec `source` et `page`
- [ ] La longueur du contenu est > 0 pour chaque document

> **Question de réflexion** : que se passe-t-il si un PDF est scanné (image) ? Quel loader faudrait-il utiliser à la place ?

---

## Étape 2 — Découper en chunks (15 min)

> **Objectif** : fragmenter les documents en chunks sémantiques cohérents, avec chevauchement.

### 2.1 Problème illustré

```
Document complet (22 pages, ~40 000 caractères)
→ Envoyé en entier dans le contexte LLM : IMPOSSIBLE (limite ~16k tokens)
→ Découpé naïvement (tous les 500 car.) : coupe les phrases et le sens
→ Découpé intelligemment (RecursiveCharSplitter) : ✅
```

Le `RecursiveCharacterTextSplitter` essaie de couper dans l'ordre : `\n\n` → `\n` → `.` → ` ` → `""`. Il préserve les paragraphes quand c'est possible.

### 2.2 Code — Splitting avec expérimentation

Ajoutez à `src/pipeline.py` :

```python
# ─────────────────────────────────────────────────────────────
# ÉTAPE 2 : DOCUMENT SPLITTING
# ─────────────────────────────────────────────────────────────

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Paramètres à expérimenter (voir Exercice 2.3)
CHUNK_SIZE    = 1000   # taille max d'un chunk en caractères
CHUNK_OVERLAP = 150    # chevauchement entre chunks consécutifs

def split_documents(documents: list) -> list:
    """Découpe les documents en chunks sémantiques."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )
    splits = splitter.split_documents(documents)
    return splits


# ── Test de l'étape 2 ────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== ÉTAPE 1 : LOADING ===")
    documents = load_documents(PDF_DIR)

    print("\n=== ÉTAPE 2 : SPLITTING ===")
    splits = split_documents(documents)

    print(f"Documents initiaux : {len(documents)}")
    print(f"Chunks produits    : {len(splits)}")
    print(f"Ratio moyen        : {len(splits)/len(documents):.1f} chunks/page")

    # Inspecter quelques chunks
    print(f"\n--- Chunk 0 ---")
    print(f"Contenu    : {splits[0].page_content[:200]}...")
    print(f"Longueur   : {len(splits[0].page_content)} caractères")
    print(f"Métadonnées: {splits[0].metadata}")

    print(f"\n--- Chunk 1 (overlap visible) ---")
    # Le début du chunk 1 doit reprendre la fin du chunk 0
    end_chunk0   = splits[0].page_content[-80:]
    start_chunk1 = splits[1].page_content[:80]
    print(f"Fin chunk 0  : ...{end_chunk0}")
    print(f"Début chunk 1: {start_chunk1}...")
```

### 2.3 Exercice — Impact des paramètres (5 min)

Modifiez les paramètres et observez l'effet :

| `chunk_size` | `chunk_overlap` | Chunks produits | Taille moy. | Observation |
|---|---|---|---|---|
| 500 | 50 | ? | ? | |
| 1000 | 150 | ? | ? | ← valeur de référence |
| 2000 | 200 | ? | ? | |

```python
# Ajoutez cette fonction d'analyse dans votre script
def analyze_splits(splits: list) -> None:
    """Analyse statistique des chunks produits."""
    sizes = [len(s.page_content) for s in splits]
    print(f"\nAnalyse des chunks :")
    print(f"  Nombre total : {len(splits)}")
    print(f"  Taille min   : {min(sizes)} caractères")
    print(f"  Taille max   : {max(sizes)} caractères")
    print(f"  Taille moy.  : {sum(sizes)/len(sizes):.0f} caractères")
    # Chunks trop courts (probablement des pages vides ou en-têtes)
    trop_courts = sum(1 for s in sizes if s < 100)
    print(f"  Chunks < 100 : {trop_courts} (à vérifier)")
```

### ✅ Point de vérification 2

- [ ] Le nombre de chunks est > nombre de pages initiales
- [ ] L'overlap est visible entre deux chunks consécutifs (fin du chunk N ≈ début du chunk N+1)
- [ ] Aucun chunk n'est vide (`len(page_content) > 0`)
- [ ] Les métadonnées (`source`, `page`) sont conservées dans chaque chunk

---

## Étape 3 — Créer le vectorstore (15 min)

> **Objectif** : transformer les chunks en vecteurs denses et les indexer dans Chroma.

### 3.1 Code — Embedding et indexation

```python
# ─────────────────────────────────────────────────────────────
# ÉTAPE 3 : VECTORSTORE + EMBEDDING
# ─────────────────────────────────────────────────────────────

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIR = "chroma_db/"

def create_vectorstore(splits: list, persist_dir: str) -> Chroma:
    """
    Crée le vectorstore Chroma depuis les chunks.
    Si le répertoire existe déjà, recharge l'index existant.
    """
    embedding = OpenAIEmbeddings()  # text-embedding-ada-002 par défaut

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        # ── Rechargement (sessions suivantes : pas de re-vectorisation) ──
        print("  ↩ Index Chroma existant rechargé depuis le disque")
        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding
        )
    else:
        # ── Création initiale ──────────────────────────────────────────
        print("  🔄 Création de l'index Chroma (appel API Embeddings)...")
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=persist_dir
        )
        print("  ✓ Index persisté sur disque")

    count = vectordb._collection.count()
    print(f"  📦 {count} vecteurs indexés dans Chroma")
    return vectordb


# ── Test de l'étape 3 ────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== ÉTAPE 1 : LOADING ===")
    documents = load_documents(PDF_DIR)
    print("\n=== ÉTAPE 2 : SPLITTING ===")
    splits = split_documents(documents)
    analyze_splits(splits)

    print("\n=== ÉTAPE 3 : VECTORSTORE ===")
    vectordb = create_vectorstore(splits, PERSIST_DIR)
```

### 3.2 Ce qui se passe en coulisses

```
chunk.page_content
    → POST https://api.openai.com/v1/embeddings
        body: {"input": "...", "model": "text-embedding-ada-002"}
    → vecteur de 1536 dimensions [0.023, -0.14, 0.87, ...]
    → stocké dans Chroma avec l'ID du chunk et ses métadonnées
```

### ✅ Point de vérification 3

- [ ] Le répertoire `chroma_db/` est créé et non vide
- [ ] Le message "Index persisté sur disque" apparaît à la première exécution
- [ ] À la deuxième exécution, le message "rechargé depuis le disque" apparaît (pas de nouvel appel API)
- [ ] Le nombre de vecteurs correspond au nombre de chunks

> **Question de réflexion** : pourquoi est-il important de persister l'index ? Quel serait le coût en crédits OpenAI de re-vectoriser 200 chunks à chaque redémarrage ?

---

## Étape 4 — Retrieval basique (10 min)

> **Objectif** : interroger le vectorstore avec une question et observer les chunks retournés.

### 4.1 Code — Similarité cosinus

```python
# ─────────────────────────────────────────────────────────────
# ÉTAPE 4 : RETRIEVAL BASIQUE (similarité cosinus)
# ─────────────────────────────────────────────────────────────

def retrieval_basic(vectordb: Chroma, question: str, k: int = 3) -> list:
    """Récupère les k chunks les plus similaires à la question."""
    docs = vectordb.similarity_search(question, k=k)
    return docs


def display_chunks(docs: list, label: str = "Résultats") -> None:
    """Affiche les chunks récupérés de façon lisible."""
    print(f"\n{'─'*60}")
    print(f"  {label} ({len(docs)} chunks)")
    print(f"{'─'*60}")
    for i, doc in enumerate(docs, 1):
        src  = doc.metadata.get('source', 'inconnu')
        page = doc.metadata.get('page', '?')
        print(f"\n[{i}] {os.path.basename(src)}, page {page}")
        print(f"    {doc.page_content[:250]}...")


# ── Test de l'étape 4 ────────────────────────────────────────
if __name__ == "__main__":
    # [Étapes 1-3 ici ...]

    print("\n=== ÉTAPE 4 : RETRIEVAL BASIQUE ===")

    question = "What are the prerequisites for this course?"
    print(f"\nQuestion : {question}")

    docs_basic = retrieval_basic(vectordb, question, k=3)
    display_chunks(docs_basic, "Similarité cosinus (k=3)")
```

### 4.2 Exercice — Observer le problème de redondance

Essayez avec `k=5` et observez :

```python
docs_5 = retrieval_basic(vectordb, question, k=5)
display_chunks(docs_5, "Similarité cosinus (k=5)")

# Vérifiez : combien de chunks sont issus de la même page ?
pages = [d.metadata.get('page') for d in docs_5]
sources = [os.path.basename(d.metadata.get('source','')) for d in docs_5]
print(f"\nPages récupérées : {pages}")
print(f"Sources          : {sources}")
```

> **Observation attendue** : plusieurs chunks peuvent être issus de la même page ou du même paragraphe → redondance → la réponse sera répétitive.

### ✅ Point de vérification 4

- [ ] La recherche retourne des chunks dont le contenu est lié à la question
- [ ] Les métadonnées (source, page) sont accessibles
- [ ] Avec `k=5`, au moins 2 chunks proviennent de la même zone du document (redondance observable)

---

## Étape 5 — Retrieval avancé avec MMR (10 min)

> **Objectif** : comparer MMR (Maximal Marginal Relevance) et similarité cosinus sur la même question.

### 5.1 Le problème de la similarité pure

```
Question : "What are the prerequisites?"

Similarité cosinus (k=3) :
  [1] Lecture01, p.0 — "Prerequisites: probability, statistics..."
  [2] Lecture01, p.0 — "We assume knowledge of probability..."   ← quasi-identique à [1]
  [3] Lecture01, p.1 — "Refresher on probability needed..."      ← encore pareil

MMR (k=3, fetch_k=10) :
  [1] Lecture01, p.0 — "Prerequisites: probability, statistics..."
  [2] Lecture02, p.3 — "Linear algebra notions required..."      ← DIFFÉRENT
  [3] Lecture01, p.5 — "Python and NumPy assumed known..."       ← DIFFÉRENT
```

MMR sélectionne les documents qui sont à la fois **pertinents** et **diversifiés**  
(Carbonell & Goldstein, 1998 — DOI: 10.1145/290941.291025).

### 5.2 Code — Comparaison MMR vs similarité cosinus

```python
# ─────────────────────────────────────────────────────────────
# ÉTAPE 5 : RETRIEVAL AVANCÉ — MMR
# ─────────────────────────────────────────────────────────────

def retrieval_mmr(vectordb: Chroma, question: str,
                  k: int = 3, fetch_k: int = 10) -> list:
    """
    Récupère k chunks avec MMR (diversité maximale).
    fetch_k : nb de candidats initiaux avant sélection MMR.
    Règle pratique : fetch_k = 3 à 5 × k
    """
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k}
    )
    docs = retriever.invoke(question)
    return docs


# ── Comparaison côte à côte ───────────────────────────────────
if __name__ == "__main__":
    # [Étapes 1-4 ici ...]

    print("\n=== ÉTAPE 5 : RETRIEVAL AVANCÉ — MMR ===")

    question = "What are the prerequisites for this course?"

    docs_cosine = retrieval_basic(vectordb, question, k=3)
    docs_mmr    = retrieval_mmr(vectordb, question, k=3, fetch_k=10)

    display_chunks(docs_cosine, "Similarité cosinus")
    display_chunks(docs_mmr,    "MMR (fetch_k=10)")

    # Mesurer la diversité : nombre de pages UNIQUES
    pages_cosine = set(d.metadata.get('page') for d in docs_cosine)
    pages_mmr    = set(d.metadata.get('page') for d in docs_mmr)

    print(f"\n{'─'*60}")
    print(f"  Pages uniques — cosinus : {len(pages_cosine)}")
    print(f"  Pages uniques — MMR     : {len(pages_mmr)}")
    print(f"  → MMR est {'plus' if len(pages_mmr) > len(pages_cosine) else 'aussi'} diversifié")
    print(f"{'─'*60}")
```

### ✅ Point de vérification 5

- [ ] MMR retourne des chunks issus de pages **différentes** par rapport à la similarité cosinus
- [ ] La métrique "pages uniques" est ≥ pour MMR vs cosinus
- [ ] Les chunks MMR restent **pertinents** malgré la diversité (lire leur contenu)

---

## Étape 6 — Pipeline Question Answering (15 min)

> **Objectif** : connecter le retriever au LLM via `RetrievalQA` pour générer une réponse sourcée.

### 6.1 Code — RetrievalQA avec prompt personnalisé

```python
# ─────────────────────────────────────────────────────────────
# ÉTAPE 6 : QUESTION ANSWERING
# ─────────────────────────────────────────────────────────────

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── LLM ──────────────────────────────────────────────────────
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0   # déterministe : réponses factuelles, reproductibles
)

# ── Prompt système personnalisé ───────────────────────────────
QA_TEMPLATE = """Tu es un assistant pédagogique expert.
Utilise UNIQUEMENT les extraits de documents ci-dessous pour répondre.
Si l'information n'est pas dans les extraits, dis-le explicitement.
Ne fais aucune hypothèse. Réponds en français.

Extraits :
{context}

Question : {question}

Réponse (avec références aux extraits) :"""

QA_PROMPT = PromptTemplate(
    template=QA_TEMPLATE,
    input_variables=["context", "question"]
)

def build_qa_chain(vectordb: Chroma) -> RetrievalQA:
    """Construit la chaîne RetrievalQA avec MMR et traçabilité des sources."""
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10}
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",                  # stratégie : tous les chunks en 1 appel
        retriever=retriever,
        return_source_documents=True,         # traçabilité ← important !
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    return qa_chain


def ask(qa_chain: RetrievalQA, question: str) -> None:
    """Pose une question et affiche la réponse avec ses sources."""
    print(f"\n❓ Question : {question}")
    result = qa_chain.invoke({"query": question})

    print(f"\n💬 Réponse :\n{result['result']}")

    print(f"\n📚 Sources utilisées ({len(result['source_documents'])} chunks) :")
    for i, doc in enumerate(result['source_documents'], 1):
        src  = os.path.basename(doc.metadata.get('source', '?'))
        page = doc.metadata.get('page', '?')
        print(f"   [{i}] {src}, page {page}")


# ── Test de l'étape 6 ────────────────────────────────────────
if __name__ == "__main__":
    # [Étapes 1-5 ici ...]

    print("\n=== ÉTAPE 6 : QUESTION ANSWERING ===")
    qa_chain = build_qa_chain(vectordb)

    # Question 1 : factuelle directe
    ask(qa_chain, "What are the prerequisites for this machine learning course?")

    # Question 2 : synthèse
    ask(qa_chain, "What topics are covered in lecture 1?")

    # Question 3 : hors corpus — observer la limite
    ask(qa_chain, "Who won the FIFA World Cup in 2022?")
```

### 6.2 Observation clé — La limite de RetrievalQA

```python
    # ── Démonstration du problème de mémoire ──────────────────
    print("\n--- DÉMONSTRATION : absence de mémoire dans RetrievalQA ---")

    # Tour 1
    ask(qa_chain, "What is probability assumed to be in this course?")

    # Tour 2 : question de suivi — "those" fait référence aux prérequis
    ask(qa_chain, "Why are those prerequisites needed?")
    # ↳ Observez : la réponse ne fait PAS référence à la probabilité
    #   Elle répond sur des "prérequis généraux" car le contexte du tour 1 est perdu.
```

### ✅ Point de vérification 6

- [ ] La réponse à Q1 cite des informations présentes dans les PDFs
- [ ] `source_documents` liste les chunks réellement utilisés
- [ ] Q3 (hors corpus) reçoit une réponse avouant l'ignorance (pas d'hallucination grâce au prompt)
- [ ] La démo de mémoire montre que le tour 2 ne référence pas le tour 1

---

## Étape 7 — Chat conversationnel avec mémoire (15 min)

> **Objectif** : résoudre le problème de mémoire avec `ConversationalRetrievalChain` et `ConversationBufferMemory`.

### 7.1 Le mécanisme de condensation

```
Tour 1 : "What is probability assumed to be?"
         → Réponse : "Probability is assumed to be known by the students."

Tour 2 : "Why are those prerequisites needed?"
         ↓
    LLM (condensation) reçoit :
    ┌─────────────────────────────────────────────────────┐
    │ Historique : [("What is probability...", "Prob...")]│
    │ Question suivi : "Why are those prerequisites..."   │
    │ Instruction : "Reformule en question autonome"      │
    └─────────────────────────────────────────────────────┘
         ↓
    Question autonome :
    "Why is knowledge of probability required as a prerequisite?"
         ↓
    Retriever → chunks sur la probabilité ✅
         ↓
    LLM (génération) → réponse cohérente ✅
```

### 7.2 Code — Chatbot complet

```python
# ─────────────────────────────────────────────────────────────
# ÉTAPE 7 : CHAT AVEC MÉMOIRE CONVERSATIONNELLE
# ─────────────────────────────────────────────────────────────

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def build_chat_chain(vectordb: Chroma) -> ConversationalRetrievalChain:
    """Construit le chatbot RAG avec mémoire conversationnelle."""
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10}
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,   # liste de HumanMessage/AIMessage
        output_key="answer"     # nécessaire avec return_source_documents=True
    )
    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False
    )
    return chat_chain, memory


def chat(chain, memory, question: str) -> None:
    """Un tour de conversation avec affichage complet."""
    print(f"\n❓ [{len(memory.chat_memory.messages)//2 + 1}] {question}")
    result = chain.invoke({"question": question})
    print(f"💬 {result['answer']}")
    print(f"   📚 Sources : {[os.path.basename(d.metadata.get('source','?'))
                               + ' p.' + str(d.metadata.get('page','?'))
                               for d in result['source_documents']]}")


# ── Session de chat interactive ───────────────────────────────
if __name__ == "__main__":
    # [Étapes 1-6 ici ...]

    print("\n=== ÉTAPE 7 : CHAT CONVERSATIONNEL ===")
    chat_chain, memory = build_chat_chain(vectordb)

    print("\n── Session 1 : test de la mémoire ──")
    chat(chat_chain, memory,
         "What is probability assumed to be in this course?")
    chat(chat_chain, memory,
         "Why are those prerequisites needed?")     # ← "those" résolu grâce à la mémoire
    chat(chat_chain, memory,
         "Can you summarize the two points you just made?")

    # Inspecter l'état de la mémoire
    print(f"\n── État de la mémoire ({len(memory.chat_memory.messages)} messages) ──")
    for msg in memory.chat_memory.messages:
        role = "🧑" if msg.type == "human" else "🤖"
        print(f"   {role} {msg.content[:80]}...")

    print("\n── Session 2 : nouvelle thématique ──")
    chat(chat_chain, memory,
         "What supervised learning algorithms are introduced?")
    chat(chat_chain, memory,
         "How do those compare to unsupervised methods?")   # ← "those" = supervised
```

### 7.3 Exercice — Observer la condensation

Pour voir la question autonome générée, activez le mode verbose :

```python
# Remplacer verbose=False par verbose=True dans build_chat_chain()
# LangChain affichera le raisonnement intermédiaire, dont la question condensée.
```

Vous verrez apparaître quelque chose comme :
```
> Entering new chain...
Standalone question: "Why is knowledge of probability and statistics 
required as a prerequisite for this machine learning course?"
```

### ✅ Point de vérification 7

- [ ] Le tour 2 répond bien sur la **probabilité** (pas sur des prérequis généraux)
- [ ] Le tour 3 ("vous venez de mentionner") prouve que la mémoire est active
- [ ] L'inspection de `memory.chat_memory.messages` montre l'historique complet
- [ ] Les sources sont citées à chaque tour

---

## Étape 8 — Analyse critique et extension (15 min)

> **Objectif** : prendre du recul sur le pipeline construit, identifier ses limites et proposer des améliorations.

### 8.1 Bilan des composants

Complétez ce tableau au vu de vos expériences des étapes 1–7 :

| Composant | Choix fait | Alternative possible | Trade-off |
|---|---|---|---|
| Loader | `PyPDFLoader` | `UnstructuredPDFLoader` | PyPDF = rapide ; Unstructured = meilleur pour PDFs complexes |
| Splitter | `RecursiveCharSplitter` | `MarkdownHeaderTextSplitter` | Recursive = universel ; Markdown = structure préservée |
| Embedding | `text-embedding-ada-002` | `sentence-transformers` local | OpenAI = qualité ; local = gratuit, privé |
| Vectorstore | Chroma | FAISS, Qdrant | Chroma = simple ; FAISS = perf. ; Qdrant = production |
| Retrieval | MMR | SelfQueryRetriever | MMR = diversité ; Self-query = filtres métadonnées |
| QA Strategy | Stuff | Map-Reduce | Stuff = simple ; Map-Reduce = grands corpus |
| Mémoire | BufferMemory | SummaryMemory | Buffer = fidèle ; Summary = compact |

### 8.2 Limites identifiées

Testez chacun de ces cas limites et notez vos observations :

```python
# ── Test des limites ──────────────────────────────────────────

# Limite 1 : question très vague
chat(chat_chain, memory, "Tell me everything.")
# → Observez : que retourne le retriever ? La réponse est-elle utile ?

# Limite 2 : très longue conversation (mémoire illimitée)
for i in range(10):
    chat(chat_chain, memory, f"Give me one fact about lecture {i+1}.")
print(f"Taille mémoire : {len(memory.chat_memory.messages)} messages")
# → Observez : le contexte envoyé au LLM grandit à chaque tour.
#   À partir de combien de tours risque-t-on de dépasser la limite ?

# Limite 3 : question hors corpus (vérification des hallucinations)
chat(chat_chain, memory, "Who invented the internet?")
# → Le modèle répond-il depuis les documents ou depuis ses connaissances ?
```

### 8.3 Exercice d'extension — SummaryMemory (optionnel)

Si le temps le permet, remplacez `ConversationBufferMemory` par `ConversationSummaryMemory` :

```python
from langchain.memory import ConversationSummaryMemory

# Remplacer dans build_chat_chain() :
memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)
```

Relancez la session longue (boucle de 10 tours) et comparez :
- La taille du contenu de mémoire envoyé au LLM
- La cohérence des réponses après de nombreux tours

### 8.4 Questions de synthèse

Répondez par écrit dans votre notebook ou en commentaire dans le code :

1. **Chunking** : si vos documents sont des emails (courts, ~150 mots), quel `chunk_size` conseilleriez-vous ? Justifiez.

2. **MMR vs cosinus** : dans quel cas préféreriez-vous la similarité cosinus simple à MMR ?

3. **Mémoire** : avec `ConversationBufferMemory`, combien de tours maximum peut-on faire avant de risquer de dépasser la fenêtre de contexte de GPT-3.5-turbo (16k tokens) ? Montrez votre raisonnement.

4. **Prompt engineering** : modifiez le `QA_TEMPLATE` pour que le modèle réponde **toujours en anglais**, **avec une structure en 3 points**, et **cite explicitement la page source** dans sa réponse.

5. **Traçabilité** : en production, pourquoi est-il essentiel d'activer `return_source_documents=True` ? Donnez deux raisons concrètes liées à la responsabilité IA.

---

## Script final consolidé

Voici le `pipeline.py` complet pour référence :

```python
# src/pipeline.py — Pipeline RAG conversationnel complet
# TP Fil Rouge Semaine 1 · Dr. Yvan GUIFO FODJO · EFREI Paris

import os
import glob
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

load_dotenv()

# ── Constantes ────────────────────────────────────────────────
PDF_DIR      = "docs/pdf/"
PERSIST_DIR  = "chroma_db/"
CHUNK_SIZE   = 1000
CHUNK_OVERLAP= 150

QA_TEMPLATE = """Tu es un assistant pédagogique expert.
Utilise UNIQUEMENT les extraits de documents ci-dessous pour répondre.
Si l'information n'est pas dans les extraits, dis-le explicitement.
Ne fais aucune hypothèse. Réponds en français.

Extraits :
{context}

Question : {question}

Réponse (avec références aux extraits) :"""

# ── Pipeline ──────────────────────────────────────────────────
def run_pipeline():
    # 1. Loading
    print("=== ÉTAPE 1 : LOADING ===")
    all_docs = []
    for pdf_path in sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf"))):
        docs = PyPDFLoader(pdf_path).load()
        all_docs.extend(docs)
        print(f"  ✓ {os.path.basename(pdf_path)} → {len(docs)} pages")

    # 2. Splitting
    print("\n=== ÉTAPE 2 : SPLITTING ===")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""])
    splits = splitter.split_documents(all_docs)
    print(f"  {len(all_docs)} pages → {len(splits)} chunks")

    # 3. Vectorstore
    print("\n=== ÉTAPE 3 : VECTORSTORE ===")
    embedding = OpenAIEmbeddings()
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        vectordb = Chroma(persist_directory=PERSIST_DIR,
                          embedding_function=embedding)
        print("  ↩ Index rechargé")
    else:
        vectordb = Chroma.from_documents(splits, embedding,
                                          persist_directory=PERSIST_DIR)
        print("  ✓ Index créé et persisté")
    print(f"  📦 {vectordb._collection.count()} vecteurs")

    # 4-5. Retrieval (MMR)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10})

    # 6. QA Chain
    print("\n=== ÉTAPE 6 : QA CHAIN ===")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate(
            template=QA_TEMPLATE, input_variables=["context","question"])})

    # 7. Chat Chain
    print("\n=== ÉTAPE 7 : CHAT ===")
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer")
    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory,
        return_source_documents=True)

    # ── Session interactive ──
    print("\n" + "═"*60)
    print("  CHATBOT RAG — tapez 'quit' pour quitter")
    print("═"*60)
    while True:
        question = input("\n❓ Votre question : ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        result = chat_chain.invoke({"question": question})
        print(f"\n💬 {result['answer']}")
        sources = [f"{os.path.basename(d.metadata.get('source','?'))} "
                   f"p.{d.metadata.get('page','?')}"
                   for d in result['source_documents']]
        print(f"   📚 Sources : {', '.join(sources)}")

if __name__ == "__main__":
    run_pipeline()
```

---

## Grille d'auto-évaluation

Évaluez votre travail sur 20 points avant de passer à la semaine 2 :

| Critère | Barème | Votre score |
|---|---|---|
| **Étape 1** — Tous les PDFs chargés, métadonnées présentes | 2 pts | /2 |
| **Étape 2** — Overlap visible entre chunks consécutifs | 2 pts | /2 |
| **Étape 2** — Tableau d'impact des paramètres complété | 1 pt | /1 |
| **Étape 3** — Index persisté et rechargeable | 2 pts | /2 |
| **Étape 4** — Redondance observée et documentée | 1 pt | /1 |
| **Étape 5** — Comparaison MMR / cosinus argumentée | 2 pts | /2 |
| **Étape 6** — QA répond sans halluciner sur Q hors corpus | 2 pts | /2 |
| **Étape 6** — Limite de mémoire démontrée et comprise | 1 pt | /1 |
| **Étape 7** — Mémoire conversationnelle fonctionnelle | 3 pts | /3 |
| **Étape 8** — Au moins 3 questions de synthèse répondues | 3 pts | /3 |
| **Code** — Propre, commenté, `.env` non versionné | 1 pt | /1 |
| **TOTAL** | **20 pts** | **/20** |

### Interprétation

| Score | Niveau | Action recommandée |
|---|---|---|
| 18–20 | Excellent | Attaquer l'extension (SummaryMemory, Self-Query) |
| 14–17 | Bon | Relire les étapes < 100% et corriger les points manquants |
| 10–13 | Passable | Refaire les étapes 6 et 7 en mode verbose |
| < 10 | Insuffisant | Reprendre les notes de cours avant de continuer |

---

## Références

1. **Lewis, P., Perez, E., Piktus, A., et al.** (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS. DOI : [`10.48550/arXiv.2005.11401`](https://arxiv.org/abs/2005.11401)

2. **Carbonell, J. & Goldstein, J.** (1998). *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries*. SIGIR. DOI : [`10.1145/290941.291025`](https://dl.acm.org/doi/10.1145/290941.291025)

3. **Radford, A., Kim, J.W., Xu, T., et al.** (2022). *Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)*. arXiv. DOI : [`10.48550/arXiv.2212.04356`](https://arxiv.org/abs/2212.04356)

4. **Chase, H.** (2022). *LangChain*. Open-source. [`github.com/langchain-ai/langchain`](https://github.com/langchain-ai/langchain)

5. **Karpukhin, V., Oğuz, B., Min, S., et al.** (2020). *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP. DOI : [`10.48550/arXiv.2004.04906`](https://arxiv.org/abs/2004.04906)

---

> **Livrable attendu pour la semaine 2** : `src/pipeline.py` fonctionnel + notebook avec les observations des étapes 4, 5 et 8 documentées + réponses aux 5 questions de synthèse  
> **Rédigé par** : Dr. Yvan GUIFO FODJO · EFREI Paris · Sprint 1, Semaine 1 · Juin 2026
