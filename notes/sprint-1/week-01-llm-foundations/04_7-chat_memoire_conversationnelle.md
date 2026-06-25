# Leçon 7 — Chat et mémoire conversationnelle

> **Cours** : *LangChain: Chat with Your Data* — DeepLearning.AI × LangChain  
> **Instructeur** : Harrison Chase (co-fondateur et CEO de LangChain)  
> **Leçon** : 7/7 — Chat (`lesson/c1ngd/chat`)  
> **Prérequis** : Leçons 1–6 (Loading, Splitting, Vectorstore, Retrieval, QA)  
> **Stack** : Python 3.10+, LangChain ≥ 0.1, OpenAI GPT-3.5-turbo, Chroma  
> **Durée estimée** : 50 min

---

## Table des matières

1. [Objectifs d'apprentissage](#1-objectifs-dapprentissage)
2. [Contexte : la limite de `RetrievalQA`](#2-contexte--la-limite-de-retrievalqa)
3. [Le concept de Chat History](#3-le-concept-de-chat-history)
4. [`ConversationBufferMemory`](#4-conversationbuffermemory)
5. [`ConversationalRetrievalChain`](#5-conversationalretrievalchain)
6. [Le mécanisme de condensation — étape par étape](#6-le-mécanisme-de-condensation--étape-par-étape)
7. [Démonstration multi-tours — code complet](#7-démonstration-multi-tours--code-complet)
8. [Gestion manuelle de la mémoire (interface UI)](#8-gestion-manuelle-de-la-mémoire-interface-ui)
9. [Architecture complète de l'interface Panel/Gradio](#9-architecture-complète-de-linterface-panelgradio)
10. [Comparatif des types de mémoire LangChain](#10-comparatif-des-types-de-mémoire-langchain)
11. [Pièges courants et bonnes pratiques](#11-pièges-courants-et-bonnes-pratiques)
12. [Synthèse de la leçon](#12-synthèse-de-la-leçon)
13. [Références bibliographiques](#13-références-bibliographiques)

---

## 1. Objectifs d'apprentissage

À l'issue de cette leçon, l'étudiant sera capable de (taxonomie de Bloom) :

| Niveau Bloom | Objectif |
|---|---|
| **Comprendre** | Expliquer pourquoi `RetrievalQA` échoue sur les questions de suivi |
| **Comprendre** | Décrire le rôle de `ConversationBufferMemory` et de `chat_history` |
| **Appliquer** | Implémenter une `ConversationalRetrievalChain` fonctionnelle en Python |
| **Appliquer** | Gérer la mémoire manuellement pour une interface multi-onglets |
| **Analyser** | Distinguer les quatre types de mémoire LangChain selon leur compromis |
| **Évaluer** | Choisir la stratégie de mémoire adaptée à une contrainte de contexte donnée |
| **Créer** | Assembler un chatbot conversationnel complet sur documents privés |

---

## 2. Contexte : la limite de `RetrievalQA`

### 2.1 Le problème démontré dans le cours

Le cours démontre le problème sur un corpus de *slides de cours* (MachineLearning-Lecture01.pdf).

```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)

# Tour 1 — fonctionne correctement
question = "Is probability a class topic?"
result = qa_chain({"query": question})
print(result["result"])
# → "Yes, probability and statistics are listed as prerequisites."

# Tour 2 — question de suivi SANS rappel du contexte
question2 = "why are those prerequisites needed?"
result2 = qa_chain({"query": question2})
print(result2["result"])
# → Répond sur les prérequis INFORMATIQUES (Python, etc.)
#   et non sur la probabilité mentionnée au tour 1.
```

### 2.2 Diagnostic

`RetrievalQA` **n'a aucune notion d'état** :

- Chaque appel est traité de façon **totalement indépendante**.
- La chaîne ne conserve ni les questions précédentes, ni les réponses produites.
- La question *« Why are those prerequisites needed? »* est transmise telle quelle au retriever, qui retourne des chunks sur les prérequis informatiques — faute de savoir que *« those »* référence la probabilité du tour précédent.

> **Conséquence pédagogique** : un chatbot sans mémoire est inutilisable pour une conversation naturelle multi-tours.

---

## 3. Le concept de Chat History

### 3.1 Définition

Le **chat history** (`chat_history`) est la liste ordonnée de tous les échanges précédents entre l'utilisateur et le système :

```
chat_history = [
    ("Is probability a class topic?",   "Yes, probability is a prerequisite."),
    ("Why are those prerequisites needed?", "..."),
    ...
]
```

Chaque élément est un tuple `(question_utilisateur, réponse_système)`.

### 3.2 Rôle dans le pipeline

Le chat history est injecté à chaque nouveau tour pour permettre :

1. La **résolution des références anaphoriques** (*« those »*, *« it »*, *« they »*...).
2. La **reformulation autonome** de la question de suivi avant de contacter le retriever.
3. La **cohérence thématique** à travers toute la conversation.

> **Note de modularité** : tous les retrievers avancés vus dans les leçons précédentes (MMR, self-query, compression contextuelle) restent **entièrement compatibles** avec cette architecture. LangChain est conçu pour que les composants s'assemblent librement.

---

## 4. `ConversationBufferMemory`

### 4.1 Présentation

`ConversationBufferMemory` est le type de mémoire le plus simple disponible dans LangChain. Il conserve une **liste brute** (*buffer*) de tous les messages échangés, sans traitement ni compression.

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",   # clé attendue par ConversationalRetrievalChain
    return_messages=True          # renvoie une liste de messages structurés
)
```

### 4.2 Paramètres clés

| Paramètre | Valeur | Explication |
|---|---|---|
| `memory_key` | `"chat_history"` | Nom de la variable dans le prompt. Doit correspondre à la clé attendue par la chaîne. |
| `return_messages` | `True` | Renvoie l'historique sous forme de liste d'objets `HumanMessage`/`AIMessage` plutôt qu'une chaîne concaténée. Recommandé pour les modèles de chat. |

### 4.3 Fonctionnement interne

```
Après chaque tour :
  memory.save_context(
      {"input": "Is probability a class topic?"},
      {"output": "Yes, probability is a prerequisite."}
  )

Au tour suivant, memory.load_memory_variables({}) retourne :
  {
    "chat_history": [
        HumanMessage(content="Is probability a class topic?"),
        AIMessage(content="Yes, probability is a prerequisite.")
    ]
  }
```

Ce buffer est passé automatiquement à la chaîne si `memory=memory` est fourni lors de la construction.

---

## 5. `ConversationalRetrievalChain`

### 5.1 Présentation

`ConversationalRetrievalChain` est la chaîne centrale de cette leçon. Elle étend `RetrievalQA` en ajoutant **une étape de condensation** : avant d'interroger le vector store, elle reformule la question de suivi en une **question autonome** (*standalone question*) exploitable sans le contexte conversationnel.

```python
from langchain.chains import ConversationalRetrievalChain

qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectordb.as_retriever(),
    memory=memory
)
```

### 5.2 Différence architecturale avec `RetrievalQA`

```
RetrievalQA :
  Question  →  Retriever  →  LLM  →  Réponse
  (sans mémoire, chaque tour est indépendant)

ConversationalRetrievalChain :
  Question + Chat History  →  LLM (condensation)  →  Question autonome
                           →  Retriever            →  Documents pertinents
                           →  LLM (génération)     →  Réponse
```

La présence d'un **double appel LLM** (condensation + génération) est le prix à payer pour la cohérence conversationnelle.

---

## 6. Le mécanisme de condensation — étape par étape

### 6.1 Vue d'ensemble

```
┌─────────────────────────────────────────┐
│  Chat History (tours précédents)        │
│  + Question de suivi (tour N)           │
└──────────────────┬──────────────────────┘
                   │
                   ▼
     ┌─────────────────────────────┐
     │  LLM — Appel 1 : Condensation│
     │  Prompt système :            │
     │  "Given the conversation and │
     │   a follow-up question,      │
     │   rephrase as a standalone   │
     │   question."                 │
     └──────────────┬──────────────┘
                    │
                    ▼
        Question autonome (standalone)
                    │
                    ▼
          ┌──────────────────┐
          │    Retriever     │  ← top-k chunks pertinents
          └────────┬─────────┘
                   │
                   ▼
     ┌─────────────────────────────┐
     │  LLM — Appel 2 : Génération │
     │  Chunks + Question autonome │
     │  → Réponse finale           │
     └─────────────────────────────┘
```

### 6.2 Exemple concret du cours

**Conversation :**

| Tour | Rôle | Message |
|---|---|---|
| 1 | Humain | *"Is probability a class topic?"* |
| 1 | IA | *"Yes, probability and statistics are listed as prerequisites."* |
| 2 | Humain | *"Why are those prerequisites needed?"* ← question ambiguë |

**Processus de condensation :**

```
Entrée condensation :
  Historique : [("Is probability a class topic?",
                 "Yes, probability is a prerequisite.")]
  Question suivi : "Why are those prerequisites needed?"

Instruction LLM : "Given the following conversation and a follow-up
                   question, rephrase the follow-up question to be a
                   standalone question, in its original language."

Sortie (question autonome) :
  "Why is knowledge of probability and statistics required
   as a prerequisite for this course?"
```

Cette question autonome est ensuite transmise au retriever, qui retourne les chunks pertinents sur la probabilité — et non plus sur les prérequis informatiques.

---

## 7. Démonstration multi-tours — code complet

### 7.1 Setup du pipeline complet

```python
import os
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()  # charge OPENAI_API_KEY depuis .env

# --- 1. Charger le vector store déjà persisté ---
persist_directory = "docs/chroma/"
embedding = OpenAIEmbeddings()

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)
print(f"Base vectorielle chargée : {vectordb._collection.count()} chunks")

# --- 2. LLM ---
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# --- 3. Mémoire ---
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# --- 4. Chaîne conversationnelle ---
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectordb.as_retriever(),
    memory=memory
)
```

### 7.2 Session de chat — démonstration du cours

```python
# ─── Tour 1 ──────────────────────────────────────────────
result1 = qa({"question": "Is probability a class topic?"})
print("Tour 1 :", result1["answer"])
# → "Yes, probability is assumed to be known.
#    The course uses it extensively in the ML models."

# ─── Tour 2 : question de suivi ambiguë ──────────────────
result2 = qa({"question": "why are those prerequisites needed?"})
print("Tour 2 :", result2["answer"])
# → "Probability is needed because many ML algorithms rely on
#    probabilistic models, Bayes' theorem, distributions, etc."
#
# ✅ La condensation a produit la question autonome :
#    "Why is probability required as a prerequisite for this ML course?"
#    Le retriever a donc retourné les bons chunks.

# ─── Tour 3 : autre question de suivi ────────────────────
result3 = qa({"question": "Who are the TAs of the course?"})
print("Tour 3 :", result3["answer"])
# → "The TAs are Alice Smith, Bob Jones and Carol Lee."

# ─── Tour 4 ──────────────────────────────────────────────
result4 = qa({"question": "What are their specialties?"})
print("Tour 4 :", result4["answer"])
# → "Alice specializes in NLP, Bob in computer vision,
#    and Carol in reinforcement learning."
#
# ✅ Question autonome générée :
#    "What are the specialties of the TAs Alice Smith, Bob Jones
#     and Carol Lee?"
```

### 7.3 Inspection de la mémoire

```python
# Visualiser l'état de la mémoire après la conversation
for msg in memory.chat_memory.messages:
    role = "Humain" if msg.type == "human" else "IA"
    print(f"[{role}] {msg.content[:80]}...")
```

---

## 8. Gestion manuelle de la mémoire (interface UI)

### 8.1 Pourquoi gérer la mémoire manuellement ?

Dans une interface utilisateur (Panel, Gradio, Streamlit), il est souvent préférable de **gérer `chat_history` manuellement** plutôt que de passer `memory=` à la chaîne. Cette approche offre :

- Un **contrôle total** sur l'affichage de l'historique dans l'UI.
- La possibilité de **modifier ou effacer** certains tours.
- Une **flexibilité** accrue pour les interfaces multi-onglets.

### 8.2 Implémentation manuelle

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Chaîne SANS memory= (gestion manuelle)
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectordb.as_retriever()
    # ← pas de memory= ici
)

# Historique géré côté application
chat_history = []

def chat(question: str) -> str:
    """Appel conversationnel avec gestion manuelle du chat_history."""
    result = qa({
        "question": question,
        "chat_history": chat_history   # ← transmis explicitement
    })
    answer = result["answer"]
    # Mise à jour manuelle de l'historique
    chat_history.append((question, answer))
    return answer

# Usage
print(chat("Who are the TAs?"))
# → "The TAs are Alice, Bob and Carol."

print(chat("What are their specialties?"))
# → "Alice works on NLP, Bob on vision, Carol on RL."

print(f"Historique : {len(chat_history)} tours conservés")
```

### 8.3 Différence clé : `memory=` vs gestion manuelle

| Aspect | `memory=memory` | Gestion manuelle |
|---|---|---|
| **Mise à jour** | Automatique après chaque appel | Manuelle : `chat_history.append(...)` |
| **Affichage UI** | Nécessite d'accéder à `memory.chat_memory` | Direct : `chat_history` est la liste |
| **Flexibilité** | Limitée | Totale (modifier, effacer, sérialiser) |
| **Simplicité** | Plus simple à mettre en place | Quelques lignes de plus |
| **Usage recommandé** | Script ou notebook | Interface utilisateur complète |

---

## 9. Architecture complète de l'interface Panel/Gradio

### 9.1 Vue d'ensemble du pipeline final du cours

Le cours assemble l'ensemble des leçons dans une interface fonctionnelle à 4 onglets :

```
┌─────────────────────────────────────────────────────────────┐
│                    Interface Panel / Gradio                  │
├─────────────┬──────────────┬─────────────────┬──────────────┤
│ Conversation│   Database   │  Chat History   │   Configure  │
│  (chat UI)  │ (chunks vu)  │  (logs bruts)   │ (upload PDF) │
└──────┬──────┴──────────────┴─────────────────┴──────────────┘
       │
       │  question + chat_history
       ▼
ConversationalRetrievalChain
       │
       ├─ 1. LLM (condensation) → question autonome
       ├─ 2. Retriever (Chroma, similarity, k=4)
       └─ 3. LLM (génération) → réponse + source_documents
```

### 9.2 Code de l'interface complète (simplifié pédagogiquement)

```python
import panel as pn
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

pn.extension()

# ── Variables globales ─────────────────────────────────────────
chat_history  = []
source_chunks = []   # chunks récupérés au dernier tour

# ── Widgets ────────────────────────────────────────────────────
inp           = pn.widgets.TextInput(placeholder="Posez votre question...")
file_input    = pn.widgets.FileInput(accept=".pdf")
k_slider      = pn.widgets.IntSlider(name="k (chunks)", start=1, end=10, value=4)
conversation  = pn.Column()
db_display    = pn.Column()
history_disp  = pn.Column()

# ── LLM ────────────────────────────────────────────────────────
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def load_db(file_content: bytes, k: int):
    """Charge un PDF uploadé et retourne une ConversationalRetrievalChain."""
    # 1. Écriture temporaire
    with open("uploaded.pdf", "wb") as f:
        f.write(file_content)
    # 2. Chargement
    loader = PyPDFLoader("uploaded.pdf")
    docs   = loader.load()
    # 3. Splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(docs)
    # 4. Indexation
    vectordb = Chroma.from_documents(splits, OpenAIEmbeddings())
    # 5. Chaîne
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        ),
        return_source_documents=True
    )
    return qa

def respond(event):
    """Callback déclenché à chaque envoi de question."""
    global chat_history, source_chunks

    question = inp.value
    inp.value = ""  # effacer le champ

    # Charger la DB si un fichier est uploadé
    if file_input.value:
        qa = load_db(file_input.value, k_slider.value)
    else:
        conversation.append(pn.pane.Markdown("⚠️ Veuillez d'abord uploader un PDF."))
        return

    # Appel conversationnel
    result       = qa({"question": question, "chat_history": chat_history})
    answer       = result["answer"]
    source_chunks = result.get("source_documents", [])
    chat_history.append((question, answer))

    # Mise à jour de l'onglet Conversation
    conversation.append(pn.pane.Markdown(f"**Vous :** {question}"))
    conversation.append(pn.pane.Markdown(f"**IA :** {answer}"))

    # Mise à jour de l'onglet Database
    db_display.clear()
    for i, doc in enumerate(source_chunks, 1):
        db_display.append(pn.pane.Markdown(
            f"**Chunk {i}** — source: `{doc.metadata.get('source','?')}` "
            f"p.{doc.metadata.get('page','?')}\n\n{doc.page_content[:300]}..."
        ))

    # Mise à jour de l'onglet History
    history_disp.clear()
    for q, a in chat_history:
        history_disp.append(pn.pane.Markdown(f"- **Q:** {q}\n  **A:** {a[:100]}..."))

inp.param.watch(respond, "value")

# ── Interface 4 onglets ────────────────────────────────────────
tabs = pn.Tabs(
    ("💬 Conversation", conversation),
    ("🗄️ Database",     db_display),
    ("📜 Chat History", history_disp),
    ("⚙️ Configure",    pn.Column(file_input, k_slider)),
)

pn.Column(inp, tabs).servable()
```

---

## 10. Comparatif des types de mémoire LangChain

LangChain propose plusieurs types de mémoire, selon le compromis entre **fidélité**, **longueur du contexte** et **coût** :

| Type | Classe | Principe | Avantage | Limite |
|---|---|---|---|---|
| **Buffer** | `ConversationBufferMemory` | Conservation intégrale de tous les messages | Simple, fidèle | Contexte croît sans limite → dépasse la fenêtre LLM |
| **Buffer Window** | `ConversationBufferWindowMemory` | Conserve les *N* derniers échanges | Contexte borné | Perd l'historique ancien |
| **Summary** | `ConversationSummaryMemory` | Résumé compressé de l'historique | Contexte compact | Un appel LLM supplémentaire pour la synthèse |
| **Token Buffer** | `ConversationTokenBufferMemory` | Fenêtre glissante sur un budget de tokens | Contrôle précis | Complexe à calibrer |
| **Entity** | `ConversationEntityMemory` | Extraction et mise à jour d'entités clés | Précision sur les entités | Plus complexe à mettre en place |

### Recommandation pratique

```python
# Texte court, prototype → ConversationBufferMemory (ce cours)
memory = ConversationBufferMemory(memory_key="chat_history",
                                   return_messages=True)

# Long historique → ConversationSummaryMemory
from langchain.memory import ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=llm,
                                    memory_key="chat_history",
                                    return_messages=True)

# Budget tokens précis → ConversationTokenBufferMemory
from langchain.memory import ConversationTokenBufferMemory
memory = ConversationTokenBufferMemory(llm=llm,
                                        max_token_limit=2000,
                                        memory_key="chat_history",
                                        return_messages=True)
```

> **Pour approfondir les types de mémoire**, le cours renvoie vers *LangChain for LLM Application Development* (Chase & Ng, DeepLearning.AI), qui couvre la mémoire de façon exhaustive.

---

## 11. Pièges courants et bonnes pratiques

### Piège 1 : oublier `return_messages=True`

```python
# ❌ Mauvais — retourne une chaîne concaténée, incompatible avec les modèles de chat
memory = ConversationBufferMemory(memory_key="chat_history")

# ✅ Bon — retourne une liste de HumanMessage/AIMessage
memory = ConversationBufferMemory(memory_key="chat_history",
                                   return_messages=True)
```

### Piège 2 : mémoire non réinitialisée entre sessions

```python
# ❌ Mauvais — la mémoire persiste entre deux utilisateurs différents (bug sérieux !)
memory = ConversationBufferMemory(...)  # défini au niveau module

# ✅ Bon — une instance par session utilisateur
def create_chain():
    memory = ConversationBufferMemory(memory_key="chat_history",
                                       return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)
```

### Piège 3 : chat_history non initialisé

```python
# ❌ Mauvais — crash si chat_history est None
result = qa({"question": q, "chat_history": None})

# ✅ Bon — initialiser à une liste vide
chat_history = []
result = qa({"question": q, "chat_history": chat_history})
```

### Piège 4 : négliger la traçabilité des sources

```python
# ✅ Toujours activer return_source_documents pour auditer les réponses
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,   # ← important en production
    memory=memory
)

result = qa({"question": question})
print("Réponse :", result["answer"])
print("Sources utilisées :")
for doc in result["source_documents"]:
    print(f"  - {doc.metadata['source']}, page {doc.metadata.get('page','?')}")
```

### Bonne pratique : combiner retrieval avancé et mémoire

```python
# MMR + mémoire conversationnelle : combinaison recommandée en production
retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10}
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,      # ← MMR pour la diversité
    memory=memory,            # ← mémoire pour la cohérence
    return_source_documents=True   # ← traçabilité
)
```

---

## 12. Synthèse de la leçon

### 12.1 Ce que résout cette leçon

| Problème | Solution |
|---|---|
| `RetrievalQA` oublie chaque tour | `ConversationalRetrievalChain` + `memory` |
| Questions de suivi ambiguës | Mécanisme de condensation (question autonome) |
| Contexte conversationnel illimité | Choix du bon type de mémoire selon la contrainte |
| Interface utilisateur multi-onglets | Gestion manuelle de `chat_history` |

### 12.2 Pipeline RAG complet — vue finale

```
Documents
   ↓ Loading (PyPDFLoader, WebBaseLoader, YoutubeAudioLoader...)
Splits (chunks)
   ↓ Splitting (RecursiveCharacterTextSplitter, chunk_size, overlap)
Embeddings
   ↓ Embedding (OpenAIEmbeddings)
Vector Store (Chroma)
   ↓ Retriever (similarity / MMR / self-query / compression)
ConversationalRetrievalChain
   ↓ LLM appel 1 : condensation (question autonome)
   ↓ Retriever : top-k chunks
   ↓ LLM appel 2 : génération (Stuff / Map-Reduce / Refine)
Réponse + sources + chat_history mis à jour
```

### 12.3 Checklist avant déploiement

- [ ] `return_messages=True` dans la mémoire
- [ ] `return_source_documents=True` pour la traçabilité
- [ ] `temperature=0` pour des réponses factuelles
- [ ] Une instance de mémoire **par session** utilisateur
- [ ] Type de mémoire adapté à la longueur de conversation attendue
- [ ] Retriever avancé (MMR) si diversité requise
- [ ] Golden set de questions pour valider la qualité avant mise en production

---

## 13. Références bibliographiques

1. **Lewis, P., Perez, E., Piktus, A., et al.** (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*. DOI : [`10.48550/arXiv.2005.11401`](https://arxiv.org/abs/2005.11401)

2. **Chase, H.** (2022). *LangChain*. Open-source library. GitHub : [`github.com/langchain-ai/langchain`](https://github.com/langchain-ai/langchain)

3. **Brown, T., Mann, B., Ryder, N., et al.** (2020). "Language Models are Few-Shot Learners." *NeurIPS 2020*. DOI : [`10.48550/arXiv.2005.14165`](https://arxiv.org/abs/2005.14165)

4. **Karpukhin, V., Oğuz, B., Min, S., et al.** (2020). "Dense Passage Retrieval for Open-Domain Question Answering." *EMNLP 2020*. DOI : [`10.48550/arXiv.2004.04906`](https://arxiv.org/abs/2004.04906)

5. **Carbonell, J. & Goldstein, J.** (1998). "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries." *SIGIR 1998*. DOI : [`10.1145/290941.291025`](https://dl.acm.org/doi/10.1145/290941.291025)

6. **LangChain Documentation** (2024). *ConversationalRetrievalChain*. [`python.langchain.com/docs/use_cases/question_answering/chat_history`](https://python.langchain.com/docs/use_cases/question_answering/chat_history)

---

> **Auteur de la note** : Dr. Yvan GUIFO FODJO — EFREI Paris — 2025–2026  
> **Source primaire** : Cours *LangChain: Chat with Your Data*, Leçon 7 (Harrison Chase, DeepLearning.AI)  
> **Date de synthèse** : Juin 2026
