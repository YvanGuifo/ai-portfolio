# 📓 Journal d'apprentissage — Semaine 1 : LLM Foundations

> **Période** : 22–28 juin 2026 · **Sprint 1 — Fondations** · **Volume** : ~12h  
> **Objectif de la semaine** : Comprendre les fondamentaux des LLM, le prompt engineering et les premiers agents  
> **Stack** : Python 3.10+, OpenAI GPT-3.5-turbo / GPT-4, LangChain ≥ 0.1, Chroma  
> **Auteur** : Dr. Yvan GUIFO FODJO — EFREI Paris

---

## Table des matières

1. [Vue d'ensemble de la semaine](#1-vue-densemble-de-la-semaine)
2. [Lundi 22 — Comment fonctionnent les LLM Transformer](#2-lundi-22--comment-fonctionnent-les-llm-transformer)
3. [Lundi 22 — Prompt Engineering for Developers](#3-lundi-22--prompt-engineering-for-developers)
4. [Mardi 23 — Building Systems with the ChatGPT API](#4-mardi-23--building-systems-with-the-chatgpt-api)
5. [Mercredi–Jeudi 24-25 — LangChain: Chat with Your Data](#5-mercredijeudi-24-25--langchain-chat-with-your-data)
6. [Vendredi–Samedi 26-27 — Functions, Tools & Agents](#6-vendrediisamedi-26-27--functions-tools--agents)
7. [Samedi 27 — Building Agentic RAG with LlamaIndex](#7-samedi-27--building-agentic-rag-with-llamaindex)
8. [Fil rouge : connexions entre les cours](#8-fil-rouge--connexions-entre-les-cours)
9. [Concepts clés consolidés](#9-concepts-clés-consolidés)
10. [Bilan de la semaine](#10-bilan-de-la-semaine)
11. [Références de la semaine](#11-références-de-la-semaine)

---

## 1. Vue d'ensemble de la semaine

### Planning réalisé

| Jour | Cours | Instructeur(s) | Durée | Source |
|---|---|---|---|---|
| Lun 22 | *How Transformer LLMs Work* | Jay Alammar & Maarten Grootendorst | ~1h34 | DeepLearning.AI |
| Lun 22 | *ChatGPT Prompt Engineering for Developers* | Isa Fulford & Andrew Ng | ~1h30 | DeepLearning.AI × OpenAI |
| Mar 23 | *Building Systems with the ChatGPT API* | Andrew Ng & Isa Fulford | ~1h45 | DeepLearning.AI × OpenAI |
| Mer-Jeu | *LangChain: Chat with Your Data* | Harrison Chase | ~2h | DeepLearning.AI × LangChain |
| Ven-Sam | *Functions, Tools & Agents* | — | ~2h | DeepLearning.AI |
| Sam 27 | *Building Agentic RAG with LlamaIndex* | — | ~1h | DeepLearning.AI |

### Fil conducteur de la semaine

La semaine trace un **arc de progression cohérent** : on commence par comprendre *ce qu'est* un LLM (architecture interne), puis *comment le piloter* (prompt engineering), puis *comment le composer en systèmes* (multi-step, RAG, agents). Chaque cours pose les bases du suivant.

```
Comprendre le LLM          Piloter le LLM         Composer des systèmes
(Transformer, attention) → (prompts, température) → (RAG, agents, mémoire)
     Cours 1                   Cours 2               Cours 3-4-5-6
```

---

## 2. Lundi 22 — Comment fonctionnent les LLM Transformer

> **Cours** : *How Transformer LLMs Work* — Jay Alammar & Maarten Grootendorst  
> **Livre associé** : *Hands-On Large Language Models* (O'Reilly, 2024)  
> **Article fondateur** : Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS.

### Ce que j'ai appris

**Pourquoi ce cours en premier ?** Avant d'utiliser des LLM comme des boîtes noires, il faut en avoir une intuition mécanique. Ce cours m'a donné exactement ça.

#### 2.1 Le problème de représentation du langage

Le texte est une donnée non structurée. Trois générations de solutions :

| Approche | Principe | Limite |
|---|---|---|
| **Bag-of-Words** | Compter les mots, ignorer l'ordre | Perd tout le sens contextuel |
| **Word2Vec** (Mikolov et al., 2013) | Embedding statique appris par co-occurrence | Un mot = un vecteur unique, peu importe le contexte |
| **Transformer** (Vaswani et al., 2017) | Embedding *contextualisé* via l'attention | — (c'est la solution actuelle) |

La révolution Word2Vec : les mots sémantiquement proches ont des vecteurs proches. La limite : *"bank"* (rive vs. banque) a le même vecteur dans tous les contextes. C'est exactement ce que l'attention résout.

#### 2.2 L'attention : le mécanisme central

L'attention permet à chaque token de **pondérer l'importance de tous les autres tokens** pour enrichir sa propre représentation. Formule :

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

- **Q** (Query) : ce que je cherche
- **K** (Key) : ce que chaque token peut offrir
- **V** (Value) : le contenu réel de chaque token

Avant l'attention, les RNN avec encodeur-décodeur compressaient toute une phrase en *un seul vecteur de contexte* — un goulot d'étranglement informationnel. L'attention donne accès à **tous les états cachés** en même temps (Bahdanau et al., 2014).

#### 2.3 Architecture Transformer decoder-only (les LLM modernes)

```
Texte brut
    ↓ Tokenisation (BPE, SentencePiece)
Séquence de tokens
    ↓ Token Embedding + Positional Encoding (RoPE)
Vecteurs enrichis
    ↓ × N blocs Transformer [Masked Self-Attention → FFN + résidus]
Représentations finales
    ↓ Language Modeling Head (softmax sur le vocabulaire)
Token suivant (génération autorégressive)
```

**Point important** : GPT, Claude, LLaMA sont tous *decoder-only*. BERT est *encoder-only*. T5 est *encoder-decoder*.

#### 2.4 Améliorations architecturales modernes

| Innovation | Rôle | Modèles concernés |
|---|---|---|
| **RMSNorm** (Pre-Norm) | Normalisation plus stable | LLaMA, Mistral |
| **RoPE** | Encodage de position relatif | LLaMA, Mistral, GPT-4 |
| **GQA** (Grouped-Query Attention) | Réduire la mémoire KV | LLaMA 2/3 |
| **FlashAttention** (Dao et al., 2022) | Attention IO-aware, très rapide | Plupart des LLM récents |
| **SwiGLU** | Activation FFN améliorée | LLaMA |
| **MoE** (Mixture of Experts) | Capacité ↑ sans calcul ↑ proportionnel | Mixtral 8x7B, DeepSeek-V2 |

L'exemple Mixtral m'a marqué : 8 experts de 7B paramètres chacun, mais seulement 2 actifs par token. Résultat : 46.7B paramètres totaux, coût équivalent à ~12.9B actifs. C'est une application directe à la fois au Green AI (efficience) et au génie logiciel (modularité).

#### 2.5 Ce que ça change pour ma pratique

> **Insight pour l'enseignement** : le mécanisme FFN comme "mémoire clé-valeur" (Geva et al., 2021) explique pourquoi les LLM "savent" des faits — mais aussi pourquoi ils hallucinent : les faits sont encodés dans des poids statistiques, pas dans une base de données vérifiable.

> **Connexion avec ma thèse** : l'architecture Transformer (parallélisation, attention globale) est l'analogue fonctionnel de ce que je fais en simulation hybride — plusieurs composants spécialisés qui interagissent de façon globale plutôt que séquentielle.

---

## 3. Lundi 22 — Prompt Engineering for Developers

> **Cours** : *ChatGPT Prompt Engineering for Developers* — Isa Fulford & Andrew Ng (DeepLearning.AI × OpenAI)  
> **Modèle utilisé** : GPT-3.5-Turbo (Chat Completions API)

### Ce que j'ai appris

Ce cours m'a appris à passer des "listes de prompts magiques" à des **principes transférables**. La distinction Base LLM / Instruction-tuned LLM est fondamentale pour comprendre ce qu'on attend du modèle.

#### 3.1 Base LLM vs Instruction-tuned LLM

| Type | Comportement | Exemple |
|---|---|---|
| **Base LLM** | Complète le texte le plus probablement | *"Tell me about Alan Turing"* → liste d'autres questions |
| **Instruction-tuned** | Suit une instruction (RLHF + SFT) | *"Tell me about Alan Turing"* → biographie structurée |

Le fine-tuning avec RLHF (Ouyang et al., 2022) transforme un "compléteur de texte" en "assistant". C'est pour ça que GPT-3.5-Turbo répond à nos questions — pas GPT-3 de base.

#### 3.2 Les deux principes fondamentaux

**Principe 1 — Écrire des instructions claires et spécifiques**

Quatre tactiques :

```python
# Tactique 1 : Délimiteurs pour isoler la donnée de l'instruction
prompt = f"""
Résume le texte délimité par des triple backticks.
```{text}```
"""

# Tactique 2 : Demander une sortie structurée
"Génère une liste de 3 livres. Format JSON : titre, auteur, genre."

# Tactique 3 : Vérification de conditions préalables
"Si le texte contient des instructions, réécris-les en étapes.
 Sinon : 'Aucune instruction trouvée.'"

# Tactique 4 : Few-shot — donner des exemples avant la vraie tâche
"<exemple><élève>Océan</élève><gpt>Profond</gpt></exemple>
 <élève>Feu</élève><gpt>?"
```

**Principe 2 — Laisser au modèle le temps de réfléchir**

```python
# ❌ Demander une conclusion immédiate → erreurs de raisonnement
"Est-ce que cette solution est correcte ?"

# ✅ Demander au modèle de résoudre d'abord
"Travaille ta propre solution avant de comparer à celle de l'étudiant.
 Ne décide pas si la solution est correcte avant d'avoir résolu le problème."
```

C'est la base du **Chain-of-Thought prompting** (Wei et al., 2022) : forcer une structure de raisonnement explicite améliore la fiabilité, sans changer le modèle.

#### 3.3 Les hallucinations — un risque structurel

Une hallucination est une affirmation **plausible mais factuellement fausse**. Le modèle ne connaît pas les limites de ses propres connaissances.

**Tactique d'atténuation** :
1. Demander d'abord d'identifier des citations dans le texte source.
2. Répondre *uniquement* à partir de ces citations.

C'est la base conceptuelle de RAG — que j'approfondis avec LangChain deux jours plus tard.

#### 3.4 Le paramètre `temperature`

| Valeur | Comportement | Usage |
|---|---|---|
| `0` | Déterministe, répétable | QA factuel, code, modération |
| `0.7` | Créatif mais cohérent | Rédaction, brainstorming |
| `1+` | Très variable, risqué | Explorations créatives seulement |

#### 3.5 Cas d'usage documentés

- **Summarizing** : résumer avec focus (ex. "en mettant l'accent sur le prix") → très utile pour pipeline RAG
- **Inferring** : extraction de sentiment, d'entités, de sujets sans étiquetage préalable
- **Transforming** : traduction, correction grammaticale, changement de ton
- **Expanding** : génération de réponse email personnalisée (avec `temperature=0.7`)

> **Insight clé** : le *prompting* n'est pas un art — c'est un processus itératif systématique. Il n'existe pas de "prompt parfait", il existe un cycle : formuler → exécuter → évaluer l'écart → ajuster → recommencer.

---

## 4. Mardi 23 — Building Systems with the ChatGPT API

> **Cours** : *Building Systems with the ChatGPT API* — Andrew Ng & Isa Fulford  
> **Niveau** : Débutant → Intermédiaire · **Durée** : ~1h45

### Ce que j'ai appris

Ce cours représente le saut conceptuel le plus important de la semaine : on passe du *prompt unique* au *système logiciel complet*. Ce n'est plus du prompting — c'est de l'**ingénierie logicielle IA**.

#### 4.1 L'accélération du développement IA

| Paradigme | Temps pour un classificateur |
|---|---|
| ML classique (données + entraînement + déploiement) | 6 mois à 2 ans |
| LLM + prompting + API | Quelques heures à quelques jours |

> Citation d'Andrew Ng : *"Des applications qu'on mettait 6 mois à 1 an à construire peuvent maintenant être construites en heures ou jours."*

⚠️ Cette accélération ne vaut que pour les **données non structurées** (texte, images). Pour les données tabulaires, le ML classique reste supérieur.

#### 4.2 Architecture d'un système LLM robuste

Le cours construit un système de service client end-to-end pour un détaillant électronique. Le flux :

```
Entrée utilisateur
    ↓
[Modération] — API Moderation OpenAI → rejeter si contenu problématique
    ↓
[Classification] — GPT-3.5 → catégorie (plainte / info produit / autre)
    ↓
[Récupération] — fonctions Python → catalogue produit pertinent
    ↓
[Génération] — GPT-3.5 avec contexte → réponse utile
    ↓
[Validation] — GPT-4 → vérifier factualité, absence d'hallucination
    ↓
Sortie utilisateur
```

**Leçon structurelle** : chaque étape est **testable indépendamment**. C'est l'équivalent des tests unitaires en génie logiciel classique — mais pour des pipelines LLM.

#### 4.3 Quand chaîner les prompts ?

| Situation | Single prompt | Multi-step chaîné |
|---|---|---|
| Tâche simple et autonome | ✅ | |
| Décisions conditionnelles multiples | | ✅ |
| Contexte externe à charger dynamiquement | | ✅ |
| Validation / modération nécessaire | | ✅ |
| Limite de tokens critique | | ✅ |

#### 4.4 Chain-of-Thought dans un système

```python
system_message = """
Résous le problème pas à pas.
Vérifie le raisonnement à chaque étape.
Si tu trouves une erreur, corrige-la et explique pourquoi.
"""
```

Wei et al. (2022) démontrent que forcer cette structure améliore la fiabilité sans modifier le modèle — c'est du "prompt engineering de niveau système".

#### 4.5 Évaluation et boucle d'amélioration

Le cours introduit une philosophie d'évaluation itérative :
1. Développer un ensemble de tests (golden set).
2. Utiliser un LLM (GPT-4) pour évaluer les réponses.
3. Mesurer, ajuster le prompt, recommencer.

> **Connexion enseignement** : cette approche est directement transposable à la conception de TP étudiants : définir des critères de réussite avant de construire l'exercice, puis évaluer avec une grille explicite.

---

## 5. Mercredi–Jeudi 24-25 — LangChain: Chat with Your Data

> **Cours** : *LangChain: Chat with Your Data* — Harrison Chase (co-fondateur LangChain)  
> **Stack** : LangChain ≥ 0.1, OpenAI GPT-3.5-turbo, Chroma · **Durée** : ~2h  
> **Référence centrale** : Lewis et al. (2020). *RAG for Knowledge-Intensive NLP Tasks*. NeurIPS. DOI: 10.48550/arXiv.2005.11401

### Ce que j'ai appris

Ce cours est le plus dense et le plus directement applicable de la semaine. Il couvre l'intégralité du pipeline RAG en 6 étapes avec LangChain.

#### 5.1 Pourquoi RAG ? Les 4 limites des LLM seuls

| Limite | Description | Solution RAG |
|---|---|---|
| **Hallucinations** | Faits plausibles mais faux | Ancrer dans des documents réels |
| **Knowledge cutoff** | Connaissance figée à l'entraînement | Injecter des données récentes |
| **Données privées** | Corpus internes inaccessibles | Indexer les documents internes |
| **Traçabilité** | Impossible de citer la source | `return_source_documents=True` |

RAG réduit les hallucinations de 30–50% (Lewis et al., 2020).

#### 5.2 Le pipeline RAG en 6 étapes

```
Documents sources
    ↓ 1. Loading — >80 loaders (PyPDFLoader, WebBaseLoader, YoutubeAudioLoader...)
    ↓ 2. Splitting — RecursiveCharacterTextSplitter (chunk_size, chunk_overlap)
    ↓ 3. Embedding — OpenAIEmbeddings → vecteurs denses
    ↓ 4. Vectorstore — Chroma (persist_directory)
         [PHASE EN LIGNE]
    ↓ 5. Retrieval — similarity / MMR / self-query / compression
    ↓ 6. QA + Chat — RetrievalQA / ConversationalRetrievalChain
Réponse + sources
```

#### 5.3 Leçon 1 — Document Loading

Chaque loader produit un objet `Document` uniforme :

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("lecture01.pdf")
pages = loader.load()
# 22 pages → 22 Documents
# pages[0].metadata = {'source': 'lecture01.pdf', 'page': 0}
```

**Insight** : l'uniformité de l'objet `Document` (page_content + metadata) est ce qui rend LangChain modulaire. On peut changer le loader sans toucher au reste du pipeline.

#### 5.4 Leçon 2 — Document Splitting

Un découpage naïf casse le sens. Le `RecursiveCharacterTextSplitter` tente dans l'ordre : paragraphes → lignes → phrases → mots.

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]
)
```

**Règle pratique** : `chunk_overlap` = 10–20% de `chunk_size`. L'overlap préserve le contexte aux frontières.

#### 5.5 Leçon 3 — Vectorstore & Embeddings

```python
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

vectordb = Chroma.from_documents(splits, OpenAIEmbeddings(),
                                  persist_directory="docs/chroma/")
vectordb.persist()
```

**Piège documenté dans le cours** : dupliquer des documents dans la base → certains chunks récupérés 2×. Dédupliquer AVANT l'indexation.

#### 5.6 Leçon 4 — Retrieval avancé

La similarité cosinus seule ne suffit pas en production :

```python
# MMR — diversité des résultats (Carbonell & Goldstein, 1998)
retriever = vectordb.as_retriever(search_type="mmr",
                                   search_kwargs={"k":3, "fetch_k":10})

# Self-Query — filtres automatiques sur métadonnées
retriever = SelfQueryRetriever.from_llm(llm, vectordb, "Lectures ML", metadata_info)

# Compression contextuelle — extraire seulement l'info pertinente
retriever = ContextualCompressionRetriever(
    base_compressor=LLMChainExtractor.from_llm(llm),
    base_retriever=vectordb.as_retriever()
)
```

**Combo recommandé en production** : MMR (diversité) + Compression (concision).

#### 5.7 Leçon 5 — Question Answering

```python
from langchain.chains import RetrievalQA

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm, retriever=vectordb.as_retriever(),
    return_source_documents=True
)
```

Trois stratégies de chaînage :

| Stratégie | Nb chunks | Coût | Cas d'usage |
|---|---|---|---|
| **Stuff** | ≤ 4 | Faible | Prototype, petits volumes |
| **Map-Reduce** | Illimité | Moyen | Grands corpus |
| **Refine** | Illimité | Élevé | Qualité narrative |

#### 5.8 Leçon 6 — Chat et mémoire conversationnelle

Le problème démontré : `RetrievalQA` n'a **aucune notion d'état**. La question *"why are those prerequisites needed?"* après *"Is probability a class topic?"* retourne une réponse hors sujet — le modèle a oublié le tour précédent.

Solution : `ConversationalRetrievalChain` + condensation de la question de suivi.

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

memory = ConversationBufferMemory(memory_key="chat_history",
                                   return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm,
                                            retriever=vectordb.as_retriever(),
                                            memory=memory)
```

**Mécanisme de condensation** :

```
Chat history + Question de suivi
    ↓ LLM appel 1 (condensation)
"Pourquoi la proba est-elle requise comme prérequis du cours ?"  ← question autonome
    ↓ Retriever
chunks pertinents sur la probabilité (et non sur les prérequis informatiques)
    ↓ LLM appel 2 (génération)
Réponse cohérente
```

> **Insight critique** : la modularité de LangChain est sa vraie force. MMR, self-query, compression — tous compatibles avec `ConversationalRetrievalChain`. On assemble les composants comme des briques LEGO.

> **Connexion Green AI** : RAG évite de re-entraîner le modèle pour chaque nouvelle donnée — c'est à la fois une économie computationnelle et une réduction d'empreinte carbone directe.

---

## 6. Vendredi–Samedi 26-27 — Functions, Tools & Agents

> **Cours** : *Functions, Tools & Agents with LangChain* — DeepLearning.AI  
> **Notes** : cours en cours au moment de la rédaction de ce journal — points clés à compléter

### Ce que j'ai retenu jusqu'ici

Les functions/tools sont le mécanisme par lequel un LLM peut **appeler du code externe** de façon structurée. C'est la brique manquante entre le LLM pur et l'agent autonome.

```python
# Schéma général : le LLM décide QUAND appeler la fonction
# et avec QUELS arguments
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }
]
```

Le LLM retourne un `tool_call` au lieu d'une réponse textuelle → le code exécute la fonction → le résultat est réinjecté dans le contexte → le LLM génère la réponse finale.

### Questions ouvertes à explorer

- Comment gérer les erreurs de fonction proprement dans un agent ?
- Quelle est la différence entre un agent ReAct et un agent OpenAI Functions ?
- Comment limiter les boucles d'appels infinies (loop detection) ?

---

## 7. Samedi 27 — Building Agentic RAG with LlamaIndex

> **Cours** : *Building Agentic RAG with LlamaIndex* — DeepLearning.AI  
> **Notes** : cours en cours — à compléter

### Premières impressions

LlamaIndex est une alternative à LangChain, plus orientée **indexation et retrieval** avec une philosophie différente :
- LangChain : chaînes et agents, très général
- LlamaIndex : optimisé pour les pipelines RAG, avec des abstractions de plus haut niveau (`QueryEngine`, `Router`)

Le RAG "agentique" ajoute une couche de raisonnement : l'agent décide *quelle* base vectorielle interroger, avec *quelle stratégie*, selon le type de question. C'est du RAG avec routing dynamique.

### Questions ouvertes

- LangChain vs LlamaIndex : quand choisir l'un plutôt que l'autre en production ?
- Comment évaluer la qualité d'un QueryRouter ?

---

## 8. Fil rouge : connexions entre les cours

Cette semaine trace une **progression architecturale cohérente** que je peux représenter ainsi :

```
Cours 1 (Transformer)
→ Comprendre POURQUOI les LLM fonctionnent (attention, tokenisation)
→ Identifier leurs limites structurelles (fenêtre de contexte, hallucinations)

Cours 2 (Prompt Engineering)
→ Apprendre à PILOTER un LLM depuis l'extérieur
→ Comprendre les hallucinations comme limite à atténuer (RAG, citations)

Cours 3 (Building Systems)
→ Composer plusieurs appels LLM en SYSTÈME logiciel
→ Modération → Classification → Récupération → Génération → Validation

Cours 4 (LangChain Chat)
→ Industrialiser le RAG avec une bibliothèque (LangChain)
→ Pipeline complet : Loading → Splitting → Vectorstore → Retrieval → QA → Chat

Cours 5 (Functions & Agents)
→ Donner au LLM la capacité d'AGIR (appels d'outils, APIs externes)

Cours 6 (Agentic RAG)
→ Combiner agents et RAG : routing dynamique, décision de retrieval
```

**La grande ligne** : on passe d'un LLM qui *prédit* à un système qui *raisonne et agit*.

---

## 9. Concepts clés consolidés

### 9.1 Tableau de synthèse

| Concept | Définition | Cours source |
|---|---|---|
| **Transformer decoder-only** | Architecture générant du texte token par token par autoregression | Cours 1 |
| **Self-Attention** | Mécanisme permettant à chaque token de pondérer les autres | Cours 1 |
| **Instruction-tuned LLM** | Modèle ajusté via RLHF pour suivre des instructions | Cours 1 & 2 |
| **Hallucination** | Affirmation plausible mais factuellement fausse | Cours 2 |
| **Chain-of-Thought** | Forcer un raisonnement explicite étape par étape | Cours 2 & 3 |
| **Chunking** | Découpage d'un document en fragments sémantiques | Cours 4 |
| **Embedding** | Représentation vectorielle dense d'un texte | Cours 4 |
| **Vector store** | Base de données optimisée pour la recherche de vecteurs similaires | Cours 4 |
| **RAG** | Retrieval-Augmented Generation — enrichir le LLM avec des docs externes | Cours 4 |
| **MMR** | Maximal Marginal Relevance — diversifier les résultats de retrieval | Cours 4 |
| **Self-Query** | Filtres automatiques sur métadonnées via le LLM | Cours 4 |
| **Chat History** | Liste des échanges précédents pour la cohérence multi-tours | Cours 4 |
| **Condensation** | Reformuler une question de suivi en question autonome | Cours 4 |
| **Tool/Function calling** | Permettre au LLM d'appeler du code externe de façon structurée | Cours 5 |
| **Agentic RAG** | Routing dynamique du retrieval par un agent | Cours 6 |

### 9.2 Code que je retiens

```python
# Pattern RAG minimal fonctionnel (LangChain)
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# Phase offline
docs   = PyPDFLoader("document.pdf").load()
splits = RecursiveCharacterTextSplitter(
             chunk_size=1000, chunk_overlap=150).split_documents(docs)
vectordb = Chroma.from_documents(splits, OpenAIEmbeddings(),
                                  persist_directory="./chroma")

# Phase en ligne
memory = ConversationBufferMemory(memory_key="chat_history",
                                   return_messages=True)
qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    retriever=vectordb.as_retriever(search_type="mmr"),
    memory=memory,
    return_source_documents=True
)

# Chat
result = qa({"question": "Quels sont les prérequis du cours ?"})
print(result["answer"])
```

---

## 10. Bilan de la semaine

### Ce qui a bien fonctionné

- **La progression pédagogique** est excellente : les 6 cours s'enchaînent naturellement, chacun apportant la couche suivante.
- **LangChain** tient sa promesse de modularité : changer un retriever ne casse pas le reste du pipeline.
- **L'approche RAG** est immédiatement applicable à mes cours : indexer mes propres supports pédagogiques et construire un assistant pour les étudiants.

### Ce qui a été difficile

- La **limite de token** et son impact sur le choix de stratégie QA (Stuff vs Map-Reduce) — il faut développer l'intuition sur les volumes de données.
- La **gestion de la mémoire conversationnelle** dans une interface multi-utilisateurs (isolation par session) — un piège non évident.
- Les cours 5 et 6 (Functions & Agents, Agentic RAG) n'étaient pas terminés au moment de la rédaction.

### Ce que je ferai différemment

- Commencer par coder *avant* de prendre des notes — les concepts s'ancrent mieux avec la pratique.
- Construire un **golden set** de questions-réponses dès le début de tout nouveau pipeline RAG, pour pouvoir évaluer chaque changement.

### Connexions avec mon profil recherche et enseignement

| Mon domaine | Connexion avec la semaine 1 |
|---|---|
| **Génie logiciel** | Pipeline RAG = architecture microservices (chaque étape = composant testable, remplaçable) |
| **Simulation hybride** | Transformer decoder-only = simulation événementielle (génération token par token, état interne) ; MoE = architecture à agents spécialisés |
| **Green AI** | RAG évite le re-entraînement → économie carbone directe ; `temperature=0` → moins de tokens générés |
| **IA responsable** | Hallucinations = problème de traçabilité ; `return_source_documents=True` = mécanisme d'audit |
| **Enseignement** | Système multi-step (cours 3) = modèle pour concevoir des TP progressifs ; RAG = assistant étudiant sur mes propres supports |

### Livrable accompli

- [x] Notes de synthèse complètes (6 cours documentés)
- [ ] Compte Azure créé (200$ crédits) — *à faire*
- [ ] Repo GitHub ai-portfolio initialisé — *à faire*

### Prochain sprint (Semaine 2 — Agents & RAG avancé)

- Agentic Workflows with LangGraph
- Multi AI Agent Systems with crewAI
- Optimizing RAG + Red Teaming LLM
- Vertex AI RAG Engine (Google Cloud)
- **Livrable** : mini-pipeline RAG fonctionnel sur GitHub

---

## 11. Références de la semaine

> Toutes les références sont vérifiables. Les DOI permettent l'accès aux articles originaux.

1. **Vaswani, A., Shazeer, N., Parmar, N., et al.** (2017). *Attention Is All You Need*. NeurIPS. [`arXiv:1706.03762`](https://arxiv.org/abs/1706.03762)

2. **Mikolov, T., Chen, K., Corrado, G., & Dean, J.** (2013). *Efficient Estimation of Word Representations in Vector Space*. [`arXiv:1301.3781`](https://arxiv.org/abs/1301.3781)

3. **Bahdanau, D., Cho, K., & Bengio, Y.** (2014). *Neural Machine Translation by Jointly Learning to Align and Translate*. [`arXiv:1409.0473`](https://arxiv.org/abs/1409.0473)

4. **Ouyang, L., Wu, J., Jiang, X., et al.** (2022). *Training language models to follow instructions with human feedback*. NeurIPS. [`arXiv:2203.02155`](https://arxiv.org/abs/2203.02155)

5. **Wei, J., Wang, X., Schuurmans, D., et al.** (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS. [`arXiv:2201.11903`](https://arxiv.org/abs/2201.11903)

6. **Lewis, P., Perez, E., Piktus, A., et al.** (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS. DOI: [`10.48550/arXiv.2005.11401`](https://arxiv.org/abs/2005.11401)

7. **Karpukhin, V., Oğuz, B., Min, S., et al.** (2020). *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP. [`arXiv:2004.04906`](https://arxiv.org/abs/2004.04906)

8. **Carbonell, J. & Goldstein, J.** (1998). *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries*. SIGIR. DOI: [`10.1145/290941.291025`](https://dl.acm.org/doi/10.1145/290941.291025)

9. **Dao, T., Fu, D.Y., Ermon, S., et al.** (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS. [`arXiv:2205.14135`](https://arxiv.org/abs/2205.14135)

10. **Jiang, A.Q., et al.** (2024). *Mixtral of Experts*. [`arXiv:2401.04088`](https://arxiv.org/abs/2401.04088)

11. **Geva, M., Schuster, R., Berant, J., & Levy, O.** (2021). *Transformer Feed-Forward Layers Are Key-Value Memories*. EMNLP. [`arXiv:2012.14913`](https://arxiv.org/abs/2012.14913)

12. **Chase, H.** (2022). *LangChain*. [`github.com/langchain-ai/langchain`](https://github.com/langchain-ai/langchain)

13. **Alammar, J. & Grootendorst, M.** (2024). *Hands-On Large Language Models*. O'Reilly Media. ISBN: 978-1098150969

---

> **Rédigé par** : Dr. Yvan GUIFO FODJO · EFREI Paris · Sprint 1, Semaine 1  
> **Sources primaires** : notes de cours DeepLearning.AI (fichiers `01-how-llms-work.md`, `02-prompt-engineering.md`, `03-building-systems-chatgpt.md`, `04-langchain-chat-data.md`, `05-functions-tools-agents.md`, `06-agentic-rag-llamaindex.md`)  
> **Date de rédaction** : Juin 2026
