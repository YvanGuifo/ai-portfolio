# LangChain: Chat with Your Data — Synthèse complète

> **Cours** : *LangChain: Chat with Your Data* — DeepLearning.AI × LangChain  
> **Instructeur** : Harrison Chase (co-fondateur et CEO de LangChain)  
> **Contributeurs matériel** : Ankush Gola & Lance Martin (LangChain) ; Geoff Ladwig & Diala Ezzeddine (DeepLearning.AI)  
> **Prérequis recommandé** : *LangChain for LLM Application Development* (Chase & Ng) — prompts, modèles, chaînes, agents  
> **Durée officielle** : 1h08 (7 leçons + quiz)  
> **Niveau** : Débutant  
> **Langage / Stack** : Python, LangChain, OpenAI (GPT-3.5), Chroma  
> **Date de synthèse** : Juin 2026

---

## Table des matières

1. [Vue d'ensemble du cours](#1-vue-densemble-du-cours)
2. [Document Loading — Charger les données](#2-document-loading--charger-les-données)
3. [Document Splitting — Découper en chunks sémantiques](#3-document-splitting--découper-en-chunks-sémantiques)
4. [Vectorstores and Embedding — Indexer pour la recherche](#4-vectorstores-and-embedding--indexer-pour-la-recherche)
5. [Retrieval — Au-delà de la similarité simple](#5-retrieval--au-delà-de-la-similarité-simple)
6. [Question Answering — Générer la réponse](#6-question-answering--générer-la-réponse)
7. [Chat — Ajouter la mémoire conversationnelle](#7-chat--ajouter-la-mémoire-conversationnelle)
8. [Architecture complète du pipeline RAG](#8-architecture-complète-du-pipeline-rag)
9. [Notions essentielles à retenir](#9-notions-essentielles-à-retenir)
10. [Références bibliographiques](#10-références-bibliographiques)

---

## 1. Vue d'ensemble du cours

### 1.1 Positionnement et objectif

Ce cours est le deuxième volet de la collaboration LangChain × DeepLearning.AI, après *LangChain for LLM Application Development*. Il **zoome sur un cas d'usage spécifique** : comment construire une application qui répond à des questions à partir de **données privées** — documents d'entreprise non disponibles publiquement sur Internet, ou données plus récentes que la date d'entraînement du LLM.

Le cours couvre deux grands thèmes :

1. **Retrieval Augmented Generation (RAG)** — récupérer des documents contextuels pertinents dans une base externe avant de générer une réponse.
2. **Construction d'un chatbot** complet qui répond aux questions à partir du contenu de documents, plutôt qu'à partir des seules connaissances apprises pendant l'entraînement.

### 1.2 Pourquoi RAG ?

Un LLM pris isolément ne connaît que ce qu'il a vu pendant son entraînement. Cela exclut :
- les **données privées** d'une entreprise (documents internes, bases propriétaires) ;
- les **données postérieures** à la date de coupure des connaissances du modèle.

RAG répond à ce problème en injectant, à chaque requête, le contenu pertinent récupéré dans une base externe, plutôt qu'en s'appuyant uniquement sur les paramètres internes du modèle (Lewis et al., 2020).

### 1.3 Le pipeline en six étapes

Le cours suit linéairement les six étapes du pipeline RAG, chacune faisant l'objet d'une leçon dédiée :

```
Document Loading → Document Splitting → Vectorstores & Embedding
        → Retrieval → Question Answering → Chat (mémoire)
```

| Étape | Question à laquelle elle répond |
|---|---|
| **Loading** | Comment importer des données depuis des sources hétérogènes ? |
| **Splitting** | Comment découper en fragments exploitables sans casser le sens ? |
| **Vectorstores & Embedding** | Comment indexer ces fragments pour les retrouver rapidement ? |
| **Retrieval** | Comment retrouver les fragments les plus pertinents et diversifiés ? |
| **Question Answering** | Comment utiliser ces fragments pour générer une réponse fiable ? |
| **Chat** | Comment gérer les questions de suivi et la conversation ? |

---

## 2. Document Loading — Charger les données

### 2.1 Le rôle des Document Loaders

Les **document loaders** gèrent les spécificités d'accès et de conversion des données depuis des formats et sources variés (sites web, bases de données, YouTube, etc.) vers un **format standardisé**. LangChain propose plus de **80 loaders différents**.

Chaque document chargé devient un objet `Document` standard, composé de :
- **`page_content`** : le contenu textuel ;
- **`metadata`** : informations associées (source, numéro de page, etc.).

### 2.2 Catégorisation des sources

| Catégorie | Exemples de sources |
|---|---|
| **Données non structurées publiques** | YouTube, Twitter, Hacker News |
| **Données non structurées propriétaires** | Figma, Notion |
| **Données structurées** (tabulaires, avec texte exploitable) | Airbyte, Stripe, Airtable |

### 2.3 Loader PDF — `PyPDFLoader`

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()
```

- Chaque page du PDF devient un `Document` distinct (22 pages → 22 documents dans l'exemple du cours).
- `metadata` contient la **source** (nom du fichier) et le numéro de **page**.

### 2.4 Loader audio YouTube — Whisper

Pour transcrire l'audio d'une vidéo YouTube, deux composants se combinent :

- **`YoutubeAudioLoader`** : télécharge l'audio depuis l'URL YouTube.
- **`OpenAIWhisperParser`** : modèle de reconnaissance vocale (speech-to-text) d'OpenAI qui convertit l'audio en texte exploitable (Radford et al., 2022).

```python
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

url = "https://www.youtube.com/watch?v=..."
save_dir = "docs/youtube/"
loader = GenericLoader(
    YoutubeAudioLoader([url], save_dir),
    OpenAIWhisperParser()
)
docs = loader.load()
```

### 2.5 Loader web — `WebBaseLoader`

```python
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://github.com/.../some-file.md")
docs = loader.load()
```

⚠️ Le contenu brut récupéré contient souvent beaucoup d'espaces blancs et de bruit, ce qui illustre la nécessité d'un **post-traitement** avant exploitation.

### 2.6 Loader Notion

`NotionDirectoryLoader` charge des données exportées depuis Notion (au format Markdown), une source très utilisée pour les bases de connaissances personnelles ou d'entreprise.

> **Point pédagogique du cours** : c'est l'occasion d'identifier des sources de données pour lesquelles LangChain n'a pas encore de loader — une opportunité de contribution open source (« *maybe you can even make a PR to LangChain* »).

---

## 3. Document Splitting — Découper en chunks sémantiques

### 3.1 Pourquoi découper, et pourquoi c'est subtil

Un découpage naïf (par exemple, une coupe arbitraire tous les *N* caractères) peut **séparer une information de son contexte**. Exemple du cours : une phrase sur les spécifications d'une Toyota Camry coupée en deux chunks distincts — aucun des deux chunks ne contient assez d'information pour répondre correctement à une question sur ces spécifications.

L'enjeu est donc d'obtenir des **chunks sémantiquement cohérents**.

### 3.2 Paramètres communs à tous les text splitters

| Paramètre | Rôle |
|---|---|
| **`chunk_size`** | Taille maximale d'un chunk (mesurée par une fonction de longueur — caractères ou tokens) |
| **`chunk_overlap`** | Chevauchement entre deux chunks consécutifs (fenêtre glissante), pour préserver la continuité du contexte aux frontières |

Tous les text splitters de LangChain exposent deux méthodes : **`create_documents`** (prend une liste de textes) et **`split_documents`** (prend une liste de `Document`).

### 3.3 `CharacterTextSplitter` vs `RecursiveCharacterTextSplitter`

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

r_splitter = RecursiveCharacterTextSplitter(chunk_size=26, chunk_overlap=4)
c_splitter = CharacterTextSplitter(chunk_size=26, chunk_overlap=4)
```

- **`CharacterTextSplitter`** : découpe sur un **unique caractère séparateur** (par défaut, le saut de ligne `\n`). S'il n'y a pas de séparateur dans le texte, il ne découpe pas du tout, même si la taille dépasse `chunk_size`.
- **`RecursiveCharacterTextSplitter`** : tente une **liste ordonnée de séparateurs**, des plus grossiers aux plus fins :

```python
RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""]
)
```

Le splitter essaie d'abord de découper sur double saut de ligne (séparation de paragraphes). S'il faut encore réduire la taille, il passe au saut de ligne simple, puis à l'espace, puis enfin caractère par caractère.

**Conséquence pratique** : sur un texte avec des paragraphes, le `RecursiveCharacterTextSplitter` produit des chunks qui respectent les frontières de paragraphes, même si certains sont plus courts que `chunk_size` — un découpage **plus pertinent** qu'une coupure arbitraire au milieu d'une phrase.

### 3.4 Découpage par phrase avec une regex *lookbehind*

Pour découper précisément entre phrases (séparateur `". "`), une regex naïve place mal les points. La solution est un **lookbehind regex** : `"(?<=\. )"`, qui découpe juste après le point sans le consommer, plaçant correctement la ponctuation.

### 3.5 Découpage par tokens — `TokenTextSplitter`

Les LLM ont des fenêtres de contexte définies en **tokens**, pas en caractères. Il est donc pertinent de découper directement par tokens pour refléter fidèlement la perception du modèle :

```python
from langchain.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
```

Un même mot peut correspondre à un nombre de caractères très différent du nombre de tokens (ex. : *foo*, *bar*, *zzzzzz* peuvent être 1 à 3 tokens selon leur fréquence d'apparition dans le corpus d'entraînement du tokenizer).

### 3.6 Découpage sensible à la structure — `MarkdownHeaderTextSplitter`

Ce splitter découpe un document Markdown selon ses **en-têtes** (`#`, `##`, `###`...) et **propage l'information hiérarchique dans les métadonnées** de chaque chunk résultant.

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
splits = splitter.split_text(markdown_document)
```

Un chunk situé sous *Titre > Chapitre 1 > Sous-section* portera ces trois niveaux dans ses métadonnées — une information précieuse pour le filtrage ultérieur (voir §5.3, self-query).

### 3.7 Splitter spécifique au code

LangChain propose également un splitter avec des séparateurs adaptés à chaque langage de programmation (Python, Ruby, C, etc.), qui prend en compte la syntaxe propre à chaque langage lors du découpage.

---

## 4. Vectorstores and Embedding — Indexer pour la recherche

### 4.1 Rappel sur les embeddings

Un **embedding** transforme un texte en une représentation numérique (vecteur dense) telle que des textes au contenu similaire ont des vecteurs proches dans l'espace numérique. Cette notion, déjà couverte dans le cours précédent (*LangChain for LLM Application Development*), est ici approfondie avec un focus sur les **cas limites** où la méthode échoue.

### 4.2 Le pipeline complet de bout en bout

```
Documents → Splits (chunks) → Embeddings → Vector Store
```

À l'usage :

```
Question → Embedding de la question → Comparaison à tous les vecteurs du store
        → Sélection des n plus similaires → Passage à un LLM avec la question
        → Réponse
```

Un **vector store** est une base de données optimisée pour la recherche de vecteurs similaires.

### 4.3 Mesurer la similarité — produit scalaire

Sur des phrases-tests (deux phrases sur des animaux de compagnie, une phrase sur une voiture), le cours illustre avec un **produit scalaire** (*dot product*, plus le score est élevé, plus c'est similaire) :
- Phrase 1 vs Phrase 2 (deux phrases sur des animaux) : score ≈ 0.96
- Phrase 1 vs Phrase 3 (animal vs voiture) : score ≈ 0.77
- Phrase 2 vs Phrase 3 : score ≈ 0.76

### 4.4 Mise en pratique avec Chroma

```python
from langchain.vectorstores import Chroma

persist_directory = 'docs/chroma/'
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
vectordb.persist()
```

**Chroma** est choisi pour ce cours car il est **léger et en mémoire**, ce qui facilite la prise en main. D'autres vector stores (parmi plus de 30 intégrations disponibles dans LangChain) proposent des solutions hébergées, utiles pour persister de grandes quantités de données dans le cloud.

### 4.5 Recherche par similarité simple

```python
question = "is there an email I can ask for help"
docs = vectordb.similarity_search(question, k=3)
```

Le paramètre **`k`** définit le nombre de documents retournés.

### 4.6 Trois cas limites de la similarité sémantique pure

Le cours introduit volontairement des **données "sales"** (un document dupliqué) pour révéler des défaillances typiques :

#### Cas limite 1 — Documents en double

Si la même information existe dans deux chunks distincts (ex. : deux versions du même PDF), une recherche par similarité peut retourner **deux fois le même contenu**, gaspillant une place dans le contexte transmis au LLM sans apporter d'information nouvelle.

#### Cas limite 2 — Absence de prise en compte des métadonnées structurées

Question : *« Que disent-ils sur la régression dans la troisième conférence ? »*. La recherche par similarité sémantique pure ignore la contrainte structurée *« troisième conférence »* — elle ne fait qu'une recherche sémantique sur l'ensemble du contenu, retournant des documents de plusieurs conférences différentes (1, 2 et 3) dès qu'ils évoquent la régression.

**Cause profonde** : l'information « troisième conférence » est une métadonnée structurée, non capturée par l'embedding sémantique de la phrase entière.

#### Cas limite 3 — Compromis sur le paramètre `k`

Augmenter `k` retourne plus de documents, mais les derniers documents de la liste sont souvent moins pertinents — un compromis entre couverture et précision.

> Ces trois limites motivent directement les techniques avancées de la leçon suivante (Retrieval).

---

## 5. Retrieval — Au-delà de la similarité simple

### 5.1 Maximum Marginal Relevance (MMR)

**Problème** : une recherche par similarité pure retourne souvent des documents très similaires entre eux, au risque de **manquer de la diversité** d'information.

**Exemple du cours** : un chef cuisinier demande des informations sur un champignon entièrement blanc. Les deux documents les plus similaires décrivent l'aspect physique du champignon (corps fructifère, couleur blanche), mais aucun ne mentionne qu'il s'agit d'un champignon **toxique** — pourtant une information cruciale présente dans un document moins similaire en surface.

**Principe de MMR** (Carbonell & Goldstein, 1998) :

1. Récupérer un ensemble initial de `fetch_k` documents par similarité sémantique pure.
2. Parmi cet ensemble, sélectionner itérativement les documents qui maximisent à la fois la **pertinence** et la **diversité** par rapport aux documents déjà sélectionnés.
3. Retourner un sous-ensemble final de taille `k`.

```python
docs_mmr = vectordb.max_marginal_relevance_search(question, k=2, fetch_k=3)
```

Sur l'exemple des champignons : avec une similarité simple (`k=2`), aucun document ne mentionne la toxicité. Avec MMR (`k=2, fetch_k=3`), l'information sur la toxicité apparaît parmi les résultats — un gain direct de diversité informationnelle.

### 5.2 Self-Query Retriever — séparer le sémantique du structuré

**Problème** : certaines questions combinent une composante **sémantique** et une composante de **filtre sur les métadonnées**. Exemple : *« Quels films sur les aliens sont sortis en 1980 ? »* contient :
- une partie sémantique (*aliens*) → recherche dans le contenu ;
- une partie structurée (*année = 1980*) → filtre sur les métadonnées.

**Principe** : utiliser un LLM pour **scinder automatiquement** la question initiale en (1) une requête sémantique et (2) un filtre de métadonnées, exploitable nativement par la plupart des vector stores.

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(name="source", description="The lecture the chunk is from", type="string"),
    AttributeInfo(name="page", description="The page number in the lecture", type="integer"),
]

retriever = SelfQueryRetriever.from_llm(
    llm, vectordb, document_content_description, metadata_field_info, verbose=True
)
```

Sur la question *« Que disent-ils sur la régression dans la troisième conférence ? »* (cas limite 2 du chapitre 4), le retriever infère automatiquement :
- une requête sémantique : `regression` ;
- un filtre : `source == docs/cs229_lectures/MachineLearning-Lecture03.pdf`.

Tous les documents retournés appartiennent alors bien à la troisième conférence — le cas limite est résolu.

> **Remarque de l'instructeur** : le self-query retriever est présenté comme « ma technique préférée » du cours, avec une invitation explicite à expérimenter des métadonnées imbriquées plus complexes.

### 5.3 Compression contextuelle

**Principe** : au lieu de transmettre l'intégralité de chaque document récupéré (souvent long, avec peu de phrases réellement pertinentes), on passe chaque document récupéré dans un **LLM extracteur** qui n'en retient que les segments pertinents pour la question.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever()
)
```

**Compromis** : plus d'appels au LLM (un appel d'extraction par document récupéré), mais une réponse finale plus **focalisée** sur l'essentiel.

**Combinaison avec MMR** : la compression seule reproduit le problème de duplication (cas limite 1) si le retriever sous-jacent reste en similarité simple. En configurant le retriever de base en mode MMR (`search_type="mmr"`), on obtient simultanément la concision de la compression **et** la diversité de MMR.

### 5.4 Méthodes de retrieval sans base vectorielle

Le cours mentionne, à titre d'ouverture, des techniques de NLP traditionnelles qui n'utilisent **aucune base vectorielle** :

- **SVM Retriever** : s'appuie sur un module d'embedding et un classifieur à vecteurs de support.
- **TF-IDF Retriever** : repose sur la fréquence des termes, sans notion d'embedding sémantique.

Dans la comparaison du cours, le SVM retriever produit des résultats globalement de meilleure qualité que le TF-IDF retriever sur la question test, illustrant que les approches purement lexicales restent moins performantes que des méthodes intégrant une notion sémantique.

---

## 6. Question Answering — Générer la réponse

### 6.1 Flux général

```
Question → Récupération des documents pertinents
        → [Documents + Prompt système + Question] → LLM → Réponse
```

Par défaut, **tous les chunks récupérés sont placés dans le même appel** au LLM (technique « *stuff* »). C'est la méthode la plus simple : un seul appel au modèle, mais elle se heurte aux **limites de fenêtre de contexte** si trop de documents sont récupérés.

### 6.2 `RetrievalQA` — chaîne de base

```python
from langchain.chains import RetrievalQA

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())
result = qa_chain({"query": question})
```

Le paramètre **`temperature=0`** est recommandé pour ce type de tâche : une faible variabilité produit des réponses **factuelles et fiables**, plutôt que créatives.

### 6.3 Personnaliser le prompt

```python
from langchain.prompts import PromptTemplate

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm, retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
```

- **`return_source_documents=True`** : permet d'inspecter les documents source utilisés pour générer la réponse — un mécanisme de **traçabilité** important pour vérifier que la réponse est bien ancrée dans les données.

### 6.4 Trois stratégies pour gérer plusieurs documents

Lorsque le nombre de documents récupérés dépasse la fenêtre de contexte, trois alternatives à la technique « *stuff* » existent :

| Méthode | Principe | Avantage | Inconvénient |
|---|---|---|---|
| **Stuff** (défaut) | Tous les documents dans un seul appel | Un seul appel LLM | Limité par la fenêtre de contexte |
| **MapReduce** | Chaque document traité individuellement, puis les réponses combinées en un appel final | Scalable à un nombre arbitraire de documents | Plus lent, plus d'appels ; performance dégradée si l'info est répartie entre plusieurs documents |
| **Refine** | Chaque document traité **séquentiellement**, en améliorant la réponse précédente | Meilleure agrégation séquentielle de l'information | Séquentiel donc non parallélisable ; sensible à l'ordre des documents |

### 6.5 Observation empirique du cours

Sur la question *« Est-ce que la probabilité est un sujet du cours ? »* :
- **MapReduce** échoue à donner une réponse claire (*« There is no clear answer on this question based on the given portion of the document »*), car chaque document est traité en isolation et l'information pertinente est répartie entre plusieurs documents.
- **Refine** réussit à produire une réponse cohérente, car chaque appel successif **dispose du contexte accumulé** des appels précédents.

> Le cours encourage l'inspection des traces d'exécution (UI de monitoring LangChain) pour visualiser concrètement chaque appel intermédiaire d'une chaîne MapReduce ou Refine.

### 6.6 Limite fondamentale : pas de mémoire conversationnelle

```python
question = "Is probability a class topic?"
result = qa_chain({"query": question})
# → réponse correcte sur les prérequis (probabilité, statistiques)

question2 = "why are those prerequisites needed?"
result2 = qa_chain({"query": question2})
# → réponse incohérente, parle des prérequis informatiques, pas de probabilité
```

**Diagnostic du cours** : la chaîne `RetrievalQA` **n'a aucune notion d'état** — elle ne se souvient pas des questions ou réponses précédentes. Chaque appel est traité de manière totalement indépendante. C'est cette limite que la leçon suivante résout.

---

## 7. Chat — Ajouter la mémoire conversationnelle

### 7.1 Le concept de Chat History

Pour gérer les questions de suivi, on introduit l'**historique de conversation** (*chat history*) : l'ensemble des échanges précédents entre l'utilisateur et la chaîne, qui est pris en compte à chaque nouvelle requête.

> Tous les types de retrieval avancés vus précédemment (self-query, compression, MMR) restent utilisables ici — l'architecture de LangChain est **modulaire** et les composants s'assemblent librement.

### 7.2 `ConversationBufferMemory`

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
```

- Conserve une **liste brute** (*buffer*) de tous les messages échangés.
- **`return_messages=True`** : renvoie l'historique comme une liste structurée de messages, plutôt qu'une simple chaîne de caractères concatenée.
- Type de mémoire le plus simple disponible dans LangChain (pour une étude approfondie des autres types de mémoire — résumé, fenêtre glissante —, le cours renvoie vers *LangChain for LLM Application Development*).

### 7.3 `ConversationalRetrievalChain`

```python
from langchain.chains import ConversationalRetrievalChain

qa = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory
)
```

Cette chaîne ajoute une étape déterminante par rapport à `RetrievalQA` + mémoire simple : elle **condense la question de suivi et l'historique en une question autonome** (*standalone question*) avant d'interroger le vector store.

### 7.4 Le mécanisme de condensation, étape par étape

Sur l'enchaînement *« Est-ce que la probabilité est un sujet du cours ? »* → *« Pourquoi ces prérequis sont-ils nécessaires ? »* :

1. **Premier appel LLM** (condensation) : reçoit l'historique + la question de suivi, avec l'instruction *« étant donné la conversation suivante et une question de suivi, reformule la question de suivi en question autonome »*. Résultat : *« Quelle est la raison d'exiger des bases en probabilité et statistiques comme prérequis du cours ? »*
2. Cette **question autonome** est transmise au retriever, qui récupère les documents pertinents.
3. **Second appel LLM** (génération) : reçoit les documents récupérés + la question autonome, et produit la réponse finale.

Grâce à cette condensation, la réponse finale reste bien **ancrée sur le sujet des probabilités**, sans confusion avec d'autres prérequis (informatique, par exemple) mentionnés ailleurs dans le corpus.

### 7.5 Architecture complète d'une interface de chat

Le cours assemble l'ensemble du pipeline dans une interface utilisateur fonctionnelle (`panel`), démontrant le **flux complet** :

```
Upload fichier PDF → PyPDFLoader → Splitting → Embeddings → Vector Store
        → as_retriever(search_type="similarity", search_kwargs={"k": k})
        → ConversationalRetrievalChain (mémoire gérée à l'extérieur de la chaîne)
```

**Point d'architecture important** : dans cette interface, la **mémoire est gérée manuellement à l'extérieur de la chaîne** (et non passée en paramètre `memory=`), pour plus de flexibilité dans l'affichage de l'historique côté interface. Le code étend manuellement `chat_history` après chaque tour de conversation.

L'interface présentée propose plusieurs onglets : conversation, inspection de la base vectorielle (derniers chunks récupérés), historique brut des échanges, configuration (upload de documents).

**Exemple de session multi-tours** :
- *« Qui sont les assistants d'enseignement (TAs) ? »* → réponse avec noms.
- *« Quelles sont leurs spécialités ? »* (question de suivi, sans répéter « TAs ») → réponse cohérente grâce à la mémoire conversationnelle.

---

## 8. Architecture complète du pipeline RAG

### 8.1 Vue d'ensemble synthétique

```
┌─────────────┐     ┌──────────┐     ┌────────────┐     ┌─────────────┐
│  Documents  │ --> │  Splits  │ --> │ Embeddings │ --> │ Vector Store│
│ (loaders)   │     │ (chunks) │     │            │     │  (Chroma)   │
└─────────────┘     └──────────┘     └────────────┘     └─────────────┘
                                                                │
                          ┌─────────────────────────────────────┘
                          ▼
                  ┌───────────────┐
                  │   Retrieval   │  ← similarité / MMR / self-query / compression
                  └───────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │  Question Answering    │  ← stuff / map-reduce / refine
              │  (+ Chat History)      │  ← condensation en question autonome
              └────────────────────────┘
                          │
                          ▼
                     Réponse finale
                (+ source documents)
```

### 8.2 Tableau récapitulatif des choix techniques par étape

| Étape | Décision clé | Options disponibles |
|---|---|---|
| Loading | Quel(s) loader(s) pour mes sources ? | PDF, web, YouTube/Whisper, Notion, +80 autres |
| Splitting | Chunk size / overlap / séparateurs | Character, Recursive, Token, Markdown-aware, code-aware |
| Embedding & Store | Quel vector store ? | Chroma (léger, in-memory) ou solutions hébergées (cloud) |
| Retrieval | Similarité simple suffit-elle ? | MMR (diversité), self-query (filtres), compression (concision) |
| QA | Combien de documents, quelle stratégie ? | Stuff (peu de docs), MapReduce / Refine (beaucoup de docs) |
| Chat | Faut-il gérer des questions de suivi ? | ConversationBufferMemory + condensation en question autonome |

---

## 9. Notions essentielles à retenir

1. **RAG (Retrieval Augmented Generation)** : technique consistant à enrichir la génération d'un LLM par des documents récupérés dynamiquement dans une base externe, plutôt que de s'appuyer uniquement sur les connaissances internes du modèle (Lewis et al., 2020).

2. **Document standardisé** : tout loader LangChain produit des objets `Document` uniformes (`page_content` + `metadata`), quelle que soit la source d'origine.

3. **Chunk size / chunk overlap** : les deux paramètres fondamentaux de tout découpage de texte ; le chevauchement préserve la continuité sémantique aux frontières des chunks.

4. **Le découpage récursif (`RecursiveCharacterTextSplitter`) est le choix par défaut recommandé** : il respecte la structure naturelle du texte (paragraphes > lignes > mots > caractères) plutôt qu'une coupure arbitraire.

5. **L'embedding sémantique seul a des angles morts** : duplication non détectée, ignorance des métadonnées structurées, compromis sur le nombre de résultats `k`.

6. **MMR (Maximum Marginal Relevance)** : équilibre pertinence et diversité dans les résultats de recherche (Carbonell & Goldstein, 1998) — pertinent dès que la similarité simple risque de retourner des résultats redondants.

7. **Self-Query Retriever** : utilise un LLM pour scinder automatiquement une requête en composante sémantique + filtre sur métadonnées structurées — résout le cas où une question combine recherche de contenu et contrainte factuelle (date, source, catégorie...).

8. **Compression contextuelle** : réduit chaque document récupéré à ses passages les plus pertinents avant de les transmettre au LLM final, au prix d'appels supplémentaires.

9. **Stuff / MapReduce / Refine** : trois stratégies pour gérer la combinaison de plusieurs documents récupérés face aux limites de fenêtre de contexte ; Refine privilégie la cohérence séquentielle, MapReduce privilégie la scalabilité parallèle.

10. **Une chaîne `RetrievalQA` simple n'a pas de mémoire** : chaque appel est indépendant ; les questions de suivi nécessitent l'ajout explicite d'un mécanisme de mémoire conversationnelle.

11. **`ConversationalRetrievalChain`** : ajoute une étape de **condensation** de la question de suivi et de l'historique en une question autonome, avant la phase de recherche documentaire — c'est cette étape, et non la mémoire seule, qui résout la cohérence des questions de suivi.

12. **Modularité de l'architecture** : chaque composant (loader, splitter, embedding, vector store, retriever, chain) est interchangeable indépendamment des autres — un principe de conception central de LangChain.

---

## 10. Références bibliographiques

1. **Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D.** (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. Advances in Neural Information Processing Systems (NeurIPS), 33, 9459–9474.

2. **Carbonell, J., & Goldstein, J.** (1998). *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries*. Proceedings of the 21st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '98), 335–336.

3. **Radford, A., Kim, J.W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I.** (2022). *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper). arXiv:2212.04356. Version publiée : Proceedings of the 40th International Conference on Machine Learning (ICML 2023), PMLR 202, 28492–28518.

4. **Mikolov, T., Chen, K., Corrado, G., & Dean, J.** (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv:1301.3781. *(fondation conceptuelle des embeddings, abordée dans le cours prérequis)*

5. **Chase, H.** — LangChain (framework open source). Documentation officielle : https://python.langchain.com

---

> **Note méthodologique** : Cette synthèse est rédigée à partir des transcriptions intégrales des 7 leçons vidéo du cours, accessibles publiquement sur la plateforme DeepLearning.AI, complétées par les références scientifiques originales identifiées pour chaque technique présentée (RAG, MMR, Whisper). Le cours datant de 2023, certains noms de classes LangChain mentionnés (`RetrievalQA`, `ConversationalRetrievalChain`) appartiennent à l'API LangChain de l'époque ; des équivalents existent dans les versions plus récentes du framework (notamment via LangGraph et LCEL), non couverts par ce cours.
