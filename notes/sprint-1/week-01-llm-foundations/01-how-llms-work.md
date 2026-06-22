# How Transformer LLMs Work — Synthèse complète

> **Cours** : *How Transformer LLMs Work* — DeepLearning.AI  
> **Instructeurs** : Jay Alammar & Maarten Grootendorst  
> **Livre associé** : *Hands-On Large Language Models* (Alammar & Grootendorst, O'Reilly, 2024)  
> **Article fondateur** : Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS 2017.  
> **Durée** : ~1h34  
> **Synthèse rédigée** : Juin 2026  

---

## Table des matières

1. [Vue d'ensemble du cours](#1-vue-densemble-du-cours)  
2. [Représenter le langage : du Bag-of-Words aux embeddings](#2-représenter-le-langage--du-bag-of-words-aux-embeddings)  
3. [Word2Vec et les embeddings de mots](#3-word2vec-et-les-embeddings-de-mots)  
4. [Encoder et décoder le contexte avec l'attention](#4-encoder-et-décoder-le-contexte-avec-lattention)  
5. [L'architecture Transformer](#5-larchitecture-transformer)  
6. [La tokenisation](#6-la-tokenisation)  
7. [Vue architecturale d'un LLM decoder-only](#7-vue-architecturale-dun-llm-decoder-only)  
8. [Le bloc Transformer en détail](#8-le-bloc-transformer-en-détail)  
9. [Self-Attention : le mécanisme central](#9-self-attention--le-mécanisme-central)  
10. [Améliorations récentes de l'architecture](#10-améliorations-récentes-de-larchitecture)  
11. [Mixture of Experts (MoE)](#11-mixture-of-experts-moe)  
12. [Notions essentielles à retenir](#12-notions-essentielles-à-retenir)  
13. [Références bibliographiques](#13-références-bibliographiques)  

---

## 1. Vue d'ensemble du cours

Le cours retrace l'évolution des représentations numériques du langage, depuis les approches les plus simples (sac de mots) jusqu'à l'architecture Transformer qui sous-tend les grands modèles de langage (LLM) actuels. L'objectif pédagogique principal est de fournir une **intuition solide** sur le fonctionnement interne des Transformers, afin de mieux comprendre leurs comportements et de les utiliser plus efficacement.

Le Transformer, introduit par Vaswani et al. (2017) pour la traduction automatique, repose sur un mécanisme d'**attention** qui permet au modèle de traiter l'ensemble de la séquence d'entrée en parallèle, contrairement aux architectures récurrentes (RNN) qui traitent les tokens de manière séquentielle.

L'architecture originale comporte deux parties : un **encodeur** (qui produit des représentations contextualisées de l'entrée) et un **décodeur** (qui génère du texte token par token). La plupart des LLM modernes (GPT, Claude, LLaMA, etc.) utilisent une architecture **decoder-only**, tandis que les modèles d'embeddings et de classification (BERT, etc.) utilisent une architecture **encoder-only**.

---

## 2. Représenter le langage : du Bag-of-Words aux embeddings

### Le problème fondamental

Le texte est une donnée **non structurée** qui perd son sens lorsqu'il est réduit à des zéros et des uns ou à des caractères isolés. Toute l'histoire de l'IA du langage est marquée par la recherche de **représentations numériques** qui préservent au mieux le sens du texte.

### Le Bag-of-Words (BoW)

Le sac de mots est la méthode la plus simple pour représenter numériquement un texte :

1. **Tokenisation** : découpage du texte en mots (tokens) par séparation sur les espaces.
2. **Construction du vocabulaire** : ensemble de tous les mots uniques trouvés dans le corpus.
3. **Vectorisation** : pour chaque document, on compte combien de fois chaque mot du vocabulaire apparaît.

**Exemple** :  
- Document 1 : *"That is a cute dog"*  
- Document 2 : *"My cat is cute"*  
- Vocabulaire : {That, is, a, cute, dog, My, cat}  
- Vecteur du document 2 : [0, 1, 0, 1, 0, 1, 1]

La représentation est un **vecteur creux** (*sparse vector*) de taille égale au vocabulaire. L'ordre des valeurs dans le vecteur est fixe et permet de comparer des phrases entre elles.

### Limites du BoW

Le BoW traite le langage comme un simple « sac » de mots sans aucune notion de :

- **Sémantique** : les synonymes ou les mots proches ne sont pas regroupés.
- **Ordre** : *"le chat mange la souris"* et *"la souris mange le chat"* ont le même vecteur.
- **Contexte** : un mot polysémique (ex. *"bank"*) reçoit toujours la même représentation.

---

## 3. Word2Vec et les embeddings de mots

### Principe (Mikolov et al., 2013)

Word2Vec est l'un des premiers modèles à capturer le **sens** des mots dans des vecteurs denses (*dense embeddings*) de dimension fixe (typiquement 100 à 300 valeurs). Il a été entraîné sur de très grands corpus (ex. : l'intégralité de Wikipedia).

### Mécanisme d'apprentissage

1. Chaque mot du vocabulaire est initialisé avec un vecteur aléatoire.
2. Le modèle est entraîné sur des **paires de mots** issues du corpus : il tente de prédire si deux mots sont susceptibles d'être voisins dans une phrase.
3. Au fil de l'entraînement, les mots qui partagent des voisins similaires voient leurs embeddings se rapprocher dans l'espace vectoriel.

### Propriétés des embeddings

Chaque dimension d'un embedding capture implicitement une **propriété sémantique** du mot. Par exemple, le mot *"cats"* pourrait avoir :
- Un score élevé sur les propriétés « animal » et « pluriel ».
- Un score faible sur « humain » et « nouveau-né ».

En pratique, les dimensions individuelles n'ont pas d'interprétation directe — elles sont le résultat de calculs matriciels complexes. Mais elles permettent de mesurer la **similarité** entre mots : les mots proches sémantiquement sont regroupés dans l'espace vectoriel.

### Niveaux d'embeddings

| Niveau | Description |
|---|---|
| **Token embedding** | Vecteur pour un sous-mot ou token individuel |
| **Word embedding** | Moyenne des token embeddings d'un mot |
| **Sentence embedding** | Agrégation des embeddings de tous les tokens d'une phrase |
| **Document embedding** | Agrégation au niveau du document entier |

### Limitation fondamentale : embeddings statiques

Word2Vec produit un embedding **unique** par mot, indépendamment du contexte. Le mot *"bank"* aura le même vecteur qu'il désigne une banque financière ou la rive d'un fleuve. C'est cette limitation que les Transformers résolvent.

---

## 4. Encoder et décoder le contexte avec l'attention

### Les RNN et l'architecture encodeur-décodeur

Avant les Transformers, les **réseaux de neurones récurrents** (RNN) ont permis de modéliser des séquences en tenant compte de l'ordre des mots. L'architecture encodeur-décodeur fonctionne ainsi :

1. **L'encodeur** traite la séquence d'entrée token par token et produit un **vecteur de contexte** unique résumant toute l'entrée.
2. **Le décodeur** utilise ce vecteur de contexte pour générer la séquence de sortie, token par token, de manière **autorégressive** (chaque token généré est ajouté à l'entrée du pas suivant).

### Le problème du vecteur de contexte unique

Un seul vecteur peine à capturer l'intégralité du contexte d'une longue séquence. L'information des premiers mots est « écrasée » au fur et à mesure que la séquence s'allonge — on parle de **goulot d'étranglement informationnel** (*information bottleneck*).

### L'attention (Bahdanau et al., 2014)

Le mécanisme d'attention résout ce problème en permettant au décodeur d'**accéder directement à tous les états cachés de l'encodeur**, pas seulement au vecteur de contexte final :

- Chaque token de sortie peut **pondérer** différemment les tokens d'entrée en fonction de leur pertinence.
- Les mots sémantiquement liés (ex. *"I"* et le néerlandais *"Ik"* dans une traduction) reçoivent des **poids d'attention élevés**.
- Les mots peu liés reçoivent des poids faibles.

### Génération autorégressive

Le processus de génération reste séquentiel :

1. Entrée : *"I love llamas"* → le modèle génère le premier token : *"Ik"*
2. Entrée mise à jour : *"I love llamas Ik"* → génère *"hou"*
3. Et ainsi de suite jusqu'à ce que la séquence complète soit produite.

### Limite des RNN avec attention

Malgré l'attention, les RNN traitent les tokens **séquentiellement**, ce qui empêche la **parallélisation** de l'entraînement et limite l'échelle des modèles.

---

## 5. L'architecture Transformer

### L'innovation fondamentale (Vaswani et al., 2017)

Le Transformer élimine entièrement la récurrence (RNN) et repose **uniquement sur l'attention**. Cela permet un entraînement massivement parallèle sur GPU, rendant possibles des modèles beaucoup plus grands.

### Structure de l'architecture originale

L'architecture originale est composée de **blocs encodeur et décodeur empilés** :

**Côté encodeur :**
1. Les tokens d'entrée sont convertis en embeddings (initialisés aléatoirement, puis appris).
2. La **self-attention** (attention sur la séquence d'entrée elle-même) met à jour ces embeddings en intégrant l'information contextuelle de tous les autres tokens.
3. Un **réseau feedforward** (FFN) traite chaque position indépendamment pour produire les embeddings contextualisés finaux.

**Côté décodeur :**
1. Les tokens déjà générés passent par une **masked self-attention** (attention masquée : chaque position ne peut voir que les tokens précédents, pas les futurs).
2. Une couche d'**attention croisée** (*cross-attention*) combine les représentations du décodeur avec celles de l'encodeur.
3. Un réseau feedforward produit les représentations finales.
4. Le **language modeling head** (tête de modélisation du langage) génère le prochain token.

### Trois familles de modèles

| Famille | Principe | Exemples | Usages |
|---|---|---|---|
| **Encoder-only** | Représentation contextualisée du texte complet | BERT, RoBERTa | Embeddings, classification, NER, RAG |
| **Decoder-only** | Génération autorégressive de texte | GPT, Claude, LLaMA | Génération, Q&A, code, raisonnement |
| **Encoder-decoder** | Encodage puis génération | T5, BART, Flan-T5 | Traduction, résumé, reformulation |

Les LLM les plus populaires (GPT-4, Claude, LLaMA, Mistral, etc.) sont des modèles **decoder-only**.

---

## 6. La tokenisation

### Rôle

La tokenisation est l'étape de conversion du texte brut en une séquence de **tokens** — les unités élémentaires que le modèle manipule. Un token n'est pas nécessairement un mot entier ; il peut être un sous-mot, un caractère, ou un fragment.

### Pourquoi des sous-mots ?

Les tokeniseurs ont un **vocabulaire de taille fixe** (typiquement 30 000 à 100 000 tokens). Ils ne peuvent pas représenter tous les mots existants. La solution est de décomposer les mots rares en fragments connus :

- *"vocalization"* → *"vocal"* + *"ization"*
- *"unhappiness"* → *"un"* + *"happiness"*

### Algorithmes principaux

| Algorithme | Principe | Utilisé par |
|---|---|---|
| **BPE** (Byte-Pair Encoding) | Fusionne itérativement les paires de caractères/sous-mots les plus fréquentes | GPT, LLaMA |
| **WordPiece** | Similaire au BPE, mais optimise la vraisemblance du corpus | BERT |
| **SentencePiece** | Tokenisation agnostique de la langue, opère directement sur les octets | T5, LLaMA |

### Tokens spéciaux

Les tokeniseurs ajoutent des tokens spéciaux : `[BOS]` (début de séquence), `[EOS]` (fin de séquence), `[PAD]` (remplissage), `[UNK]` (inconnu).

### Impact sur les performances

Le choix du tokeniseur et la taille du vocabulaire influencent directement la capacité du modèle à représenter le langage, en particulier pour les langues autres que l'anglais et pour les domaines spécialisés (code, mathématiques, biologie).

---

## 7. Vue architecturale d'un LLM decoder-only

Le pipeline complet d'un LLM génératif decoder-only se décompose en quatre étapes :

### Étape 1 — Token Embedding

Chaque token de la séquence d'entrée est converti en un **vecteur d'embedding** de dimension fixe (ex. : 768, 4096, 12288 selon le modèle). Ces vecteurs sont appris durant l'entraînement.

### Étape 2 — Positional Encoding

Puisque le Transformer traite tous les tokens en parallèle (contrairement aux RNN), il n'a aucune notion d'ordre intrinsèque. Le **positional encoding** ajoute une information de position à chaque embedding, permettant au modèle de distinguer le premier mot du dernier.

Deux approches principales :
- **Encodage sinusoïdal** (Vaswani et al., 2017) : fonctions sinus et cosinus de fréquences variées.
- **RoPE** (Rotary Position Embedding, Su et al., 2021) : rotation des vecteurs query/key — utilisé dans LLaMA, Mistral et la plupart des LLM modernes.

### Étape 3 — Pile de blocs Transformer

Les embeddings enrichis passent à travers une pile de N blocs Transformer identiques (ex. : 32 blocs pour LLaMA 7B, 96 pour GPT-4). Chaque bloc contient une couche d'attention et un réseau feedforward (détaillés en section 8).

### Étape 4 — Language Modeling Head

Le dernier bloc produit un vecteur de sortie pour chaque position. Le language modeling head projette ce vecteur dans l'espace du vocabulaire (via une matrice de projection) et applique un **softmax** pour obtenir une distribution de probabilité sur tous les tokens possibles. Le token suivant est sélectionné selon une **stratégie de décodage** (greedy, top-k, top-p/nucleus, temperature).

---

## 8. Le bloc Transformer en détail

Chaque bloc Transformer (dans un decoder-only) contient deux sous-couches principales, chacune entourée d'une **connexion résiduelle** (*residual connection*) et d'une **normalisation de couche** (*layer normalization*) :

### 8.1 Masked Self-Attention (Multi-Head)

- Permet à chaque token de « regarder » tous les tokens **précédents** (mais pas les suivants — d'où le masque).
- Fonctionne avec **plusieurs têtes d'attention** (*multi-head attention*) en parallèle, chacune capturant des relations différentes (syntaxiques, sémantiques, positionnelles, etc.).

### 8.2 Feedforward Network (FFN)

- Un réseau de neurones à deux couches linéaires avec une activation non-linéaire (ReLU, GELU ou SwiGLU selon le modèle).
- Traite chaque position **indépendamment**.
- C'est dans le FFN que sont stockées la majorité des **connaissances factuelles** du modèle (Dai et al., 2022 ; Geva et al., 2021).

### 8.3 Connexions résiduelles et normalisation

- **Connexion résiduelle** : la sortie de chaque sous-couche est additionnée à son entrée (`output = sublayer(x) + x`). Cela facilite l'entraînement de réseaux profonds en atténuant le problème de disparition du gradient.
- **Layer Normalization** : normalise les activations pour stabiliser l'entraînement. Dans les architectures modernes (LLaMA, etc.), on utilise **RMSNorm** (Root Mean Square Normalization) placé *avant* chaque sous-couche (Pre-Norm), plutôt qu'après (Post-Norm du Transformer original).

---

## 9. Self-Attention : le mécanisme central

### Intuition

La self-attention permet à chaque token de « consulter » les autres tokens de la séquence pour enrichir sa propre représentation. Elle se décompose en deux étapes :

1. **Relevance scoring** : calculer à quel point chaque autre token est pertinent pour le token courant.
2. **Information combining** : combiner les représentations des tokens pertinents pour produire une représentation enrichie.

### Calcul détaillé (Scaled Dot-Product Attention)

Trois matrices de poids sont apprises : **W_Q** (Query), **W_K** (Key), **W_V** (Value).

Pour chaque token :
1. On calcule un vecteur **Query** (Q), un vecteur **Key** (K) et un vecteur **Value** (V) en multipliant l'embedding par les matrices respectives.
2. Le **score de pertinence** entre le token courant (query) et chaque autre token (key) est obtenu par produit scalaire : `score = Q · K^T`.
3. Les scores sont divisés par `√d_k` (dimension des clés) pour éviter des valeurs trop grandes, puis passés dans un **softmax** pour obtenir des poids normalisés sommant à 1.
4. La sortie est la somme pondérée des vecteurs **Value** : `Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V`.

### Multi-Head Attention

Plutôt qu'un seul jeu de matrices Q/K/V, le modèle utilise **h têtes** en parallèle (ex. : h=12 pour BERT-base, h=32 pour LLaMA 7B). Chaque tête opère sur un sous-espace de dimension d_model/h :

- Chaque tête capture un type de relation différent (sujet-verbe, coréférence, proximité positionnelle, etc.).
- Les sorties de toutes les têtes sont concaténées et projetées par une matrice de sortie.

### Masquage causal

Dans un decoder, un **masque triangulaire** empêche chaque position de voir les tokens futurs. Cela garantit que la génération est autorégressive : le token à la position *t* ne peut dépendre que des tokens aux positions 1 à t-1.

---

## 10. Améliorations récentes de l'architecture

Depuis le Transformer original de 2017, de nombreuses optimisations ont été apportées :

### 10.1 Positional Encoding

| Méthode | Description | Modèles |
|---|---|---|
| Sinusoïdal (2017) | Fonctions sin/cos fixes | Transformer original |
| Learned positional embeddings | Embeddings de position appris | GPT-2 |
| **RoPE** (Su et al., 2021) | Rotation appliquée aux vecteurs Q/K | LLaMA, Mistral, Qwen |
| **ALiBi** (Press et al., 2022) | Biais linéaire ajouté aux scores d'attention | BLOOM, MPT |

RoPE est la méthode dominante dans les LLM modernes car elle permet une meilleure extrapolation aux séquences plus longues que celles vues à l'entraînement.

### 10.2 Normalisation

- **Post-Norm** (original) : normalisation après la sous-couche.
- **Pre-Norm** (moderne) : normalisation avant la sous-couche — plus stable à l'entraînement.
- **RMSNorm** (Zhang & Sennrich, 2019) : remplace LayerNorm dans la plupart des LLM modernes (LLaMA, Mistral). Plus efficace en calcul.

### 10.3 Fonction d'activation du FFN

- **ReLU** (original) : `max(0, x)`.
- **GELU** (Hendrycks & Gimpel, 2016) : activation gaussienne, utilisée par GPT-2, BERT.
- **SwiGLU** (Shazeer, 2020) : combinaison de Swish et Gated Linear Unit — utilisée par LLaMA, PaLM, Mistral. Offre de meilleures performances empiriques.

### 10.4 Optimisations de l'attention

| Technique | Principe | Bénéfice |
|---|---|---|
| **Multi-Query Attention** (Shazeer, 2019) | Toutes les têtes partagent les mêmes K et V | Réduction mémoire, inférence plus rapide |
| **Grouped-Query Attention** (GQA, Ainslie et al., 2023) | Les têtes sont groupées, chaque groupe partage K/V | Compromis performance/efficacité. Utilisé par LLaMA 2+ |
| **Flash Attention** (Dao et al., 2022) | Algorithme optimisé pour le matériel GPU (tiling) | Accélère l'entraînement et l'inférence, réduit l'empreinte mémoire |
| **KV-Cache** | Cache les vecteurs K et V déjà calculés lors de la génération | Évite de recalculer l'attention sur les tokens passés |

### 10.5 Context length

Les fenêtres de contexte se sont considérablement allongées : de 512 tokens (BERT) à 2048 (GPT-3), puis 8K–32K (GPT-4, Claude), jusqu'à 128K–1M+ tokens pour les modèles les plus récents. RoPE et les techniques d'extension de contexte (YaRN, NTK-aware scaling) jouent un rôle clé.

---

## 11. Mixture of Experts (MoE)

### Principe

L'architecture **Mixture of Experts** (MoE) permet d'augmenter massivement le nombre de paramètres d'un modèle tout en maîtrisant le coût de calcul à l'inférence.

Au lieu d'un seul réseau FFN par bloc Transformer, un modèle MoE contient **plusieurs experts** (chacun étant un FFN complet). Un mécanisme de **routage** (*gating network* / *router*) sélectionne dynamiquement un petit nombre d'experts pour chaque token :

1. Le routeur reçoit l'embedding du token et produit des scores pour chaque expert.
2. Seuls les **top-k experts** (typiquement k=1 ou k=2) sont activés pour ce token.
3. Les sorties des experts sélectionnés sont pondérées par les scores du routeur.

### Avantages

- **Capacité accrue** : le modèle dispose de beaucoup plus de paramètres totaux, lui permettant de stocker plus de connaissances.
- **Efficacité computationnelle** : seuls les k experts actifs sont calculés pour chaque token, donc le coût de calcul par token est similaire à un modèle dense beaucoup plus petit.
- **Spécialisation** : chaque expert peut se spécialiser dans certains types de tokens ou de connaissances.

### Exemple concret

Mixtral 8x7B (Mistral AI, 2023) utilise 8 experts de 7B de paramètres chacun, avec k=2 experts actifs par token. Le modèle a un total de ~46.7B de paramètres mais un coût de calcul équivalent à ~12.9B de paramètres actifs par token.

### Défis

- **Load balancing** : il faut s'assurer que tous les experts sont utilisés de manière équilibrée (un expert inutilisé est un gaspillage de paramètres). Des pertes auxiliaires (*auxiliary loss*) encouragent une distribution uniforme.
- **Communication inter-GPU** : dans un entraînement distribué, le routage des tokens vers les experts situés sur différents GPU nécessite des transferts de données.
- **Instabilité d'entraînement** : les modèles MoE peuvent être plus difficiles à entraîner de manière stable.

### Modèles MoE notables

| Modèle | Experts | Paramètres totaux | Actifs par token |
|---|---|---|---|
| Switch Transformer (Google, 2022) | 128 | 1.6T | ~12B |
| Mixtral 8x7B (Mistral, 2023) | 8 | 46.7B | 12.9B |
| Mixtral 8x22B (Mistral, 2024) | 8 | 141B | 39B |
| DeepSeek-V2 (2024) | 160 | 236B | 21B |

---

## 12. Notions essentielles à retenir

### 12.1 Concepts fondamentaux

1. **Embedding** : représentation vectorielle dense d'un token, capturant son sens. Les embeddings contextualisés (Transformer) varient selon le contexte, contrairement aux embeddings statiques (Word2Vec).

2. **Tokenisation** : découpage du texte en sous-unités (sous-mots) de vocabulaire fixe. Les algorithmes BPE et SentencePiece sont les plus répandus.

3. **Attention** : mécanisme permettant à chaque token de pondérer l'importance de tous les autres tokens pour enrichir sa propre représentation. Formule : `Attention(Q,K,V) = softmax(QK^T / √d_k) · V`.

4. **Self-Attention** : cas particulier où la séquence source et la séquence cible sont identiques (le modèle « s'écoute lui-même »).

5. **Multi-Head Attention** : exécution parallèle de h mécanismes d'attention sur des sous-espaces différents, permettant de capturer des relations variées.

6. **Masquage causal** : dans un decoder, chaque token ne peut « voir » que les tokens qui le précèdent.

7. **Génération autorégressive** : le texte est produit un token à la fois, chaque nouveau token étant conditionné par tous les précédents.

### 12.2 Architecture Transformer (decoder-only)

Le pipeline complet :

```
Texte → Tokenisation → Token Embeddings + Positional Encoding
  → [Bloc Transformer × N] → Language Modeling Head → Token suivant
```

Chaque bloc Transformer contient :
```
→ RMSNorm → Masked Multi-Head Self-Attention → + (résidu)
→ RMSNorm → Feed-Forward Network (SwiGLU) → + (résidu)
```

### 12.3 Points clés de design

| Concept | Rôle |
|---|---|
| **Connexion résiduelle** | Facilite l'apprentissage profond en préservant le gradient |
| **Layer Normalization** | Stabilise l'entraînement en normalisant les activations |
| **Positional Encoding (RoPE)** | Injecte l'information d'ordre dans les représentations |
| **KV-Cache** | Accélère la génération en évitant le recalcul des tokens passés |
| **GQA** | Réduit la mémoire en partageant K/V entre groupes de têtes |
| **MoE** | Augmente la capacité sans augmenter proportionnellement le calcul |

### 12.4 Trois familles de modèles Transformer

| Architecture | Ce qu'elle fait | Ce qu'elle ne fait pas | Exemples |
|---|---|---|---|
| **Encoder-only** | Représenter/classifier du texte | Générer du texte | BERT, RoBERTa |
| **Decoder-only** | Générer du texte token par token | Encoder de manière bidirectionnelle | GPT-4, Claude, LLaMA |
| **Encoder-Decoder** | Encoder une entrée et générer une sortie | — | T5, BART |

### 12.5 Intuition sur la « magie » des LLM

Comme le souligne Andrew Ng dans l'introduction du cours, la puissance des LLM provient de **deux facteurs** :

1. **L'architecture Transformer** elle-même — qui est élégante mais relativement simple (attention + FFN + résidus).
2. **Les données d'entraînement** — des quantités massives de texte (des centaines de milliards à des trillions de tokens) qui permettent au modèle d'apprendre la structure du langage, les connaissances factuelles et les capacités de raisonnement.

---

## 13. Références bibliographiques

1. **Vaswani, A., Shazeer, N., Parmar, N., et al.** (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems (NeurIPS).

2. **Mikolov, T., Chen, K., Corrado, G., & Dean, J.** (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv:1301.3781.

3. **Bahdanau, D., Cho, K., & Bengio, Y.** (2014). *Neural Machine Translation by Jointly Learning to Align and Translate*. arXiv:1409.0473.

4. **Devlin, J., Chang, M.W., Lee, K., & Toutanova, K.** (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT.

5. **Radford, A., et al.** (2019). *Language Models are Unsupervised Multitask Learners* (GPT-2). OpenAI Technical Report.

6. **Su, J., Lu, Y., Pan, S., et al.** (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864.

7. **Shazeer, N.** (2020). *GLU Variants Improve Transformer*. arXiv:2002.05202.

8. **Shazeer, N.** (2019). *Fast Transformer Decoding: One Write-Head is All You Need* (Multi-Query Attention). arXiv:1911.02150.

9. **Ainslie, J., et al.** (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. arXiv:2305.13245.

10. **Dao, T., Fu, D.Y., Ermon, S., et al.** (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS.

11. **Fedus, W., Zoph, B., & Shazeer, N.** (2022). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. JMLR.

12. **Jiang, A.Q., et al.** (2024). *Mixtral of Experts*. Mistral AI Technical Report. arXiv:2401.04088.

13. **Alammar, J. & Grootendorst, M.** (2024). *Hands-On Large Language Models*. O'Reilly Media.

14. **Zhang, B. & Sennrich, R.** (2019). *Root Mean Square Layer Normalization*. NeurIPS.

15. **Geva, M., Schuster, R., Berant, J., & Levy, O.** (2021). *Transformer Feed-Forward Layers Are Key-Value Memories*. EMNLP.

---

> **Note** : Cette synthèse est basée sur le contenu du cours en ligne de DeepLearning.AI (transcriptions accessibles publiquement), complété par les références scientifiques originales citées dans le cours et son livre associé. Certains détails techniques (GQA, Flash Attention, MoE) sont enrichis à partir des publications originales pour offrir un contenu plus complet.
