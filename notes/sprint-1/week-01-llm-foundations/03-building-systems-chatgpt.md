# Building Systems with the ChatGPT API — Synthèse complète & pédagogique

> **Cours** : *Building Systems with the ChatGPT API*  
> **Plateforme** : DeepLearning.AI × OpenAI  
> **Instructeurs** : Andrew Ng (DeepLearning.AI) & Isa Fulford (OpenAI)  
> **Collaborateurs** : Andrew Kondrich, Joe Palermo, Boris Power, Ted Sanders (OpenAI)  
> **Durée officielle** : 1h45 (~11 leçons)  
> **Niveau** : Débutant → Intermédiaire  
> **Langage de programmation** : Python (API OpenAI)  
> **Modèle utilisé** : GPT-3.5-Turbo, GPT-4  
> **Date de synthèse** : Juin 2026

---

## Table des matières

1. [Vue d'ensemble et positionnement](#1-vue-densemble-et-positionnement)
2. [Contexte : de la prompting unique aux systèmes complexes](#2-contexte--de-la-prompting-unique-aux-systèmes-complexes)
3. [Fondamentaux : modèles de langage, format chat, et tokenisation](#3-fondamentaux--modèles-de-langage-format-chat-et-tokenisation)
4. [Leçon 3 : Classification et catégorisation](#4-leçon-3--classification-et-catégorisation)
5. [Leçon 4 : Modération et sécurité des entrées](#5-leçon-4--modération-et-sécurité-des-entrées)
6. [Leçon 5 : Chain-of-Thought Reasoning](#6-leçon-5--chain-of-thought-reasoning)
7. [Leçon 6 : Chaining Prompts — Architecture multi-étapes](#7-leçon-6--chaining-prompts--architecture-multi-étapes)
8. [Leçon 7 : Validation et vérification des sorties](#8-leçon-7--validation-et-vérification-des-sorties)
9. [Leçon 8–10 : Évaluation et amélioration itérative](#9-leçon-8-10--évaluation-et-amélioration-itérative)
10. [Philosophie générale : de la prompting traditionnelle au génie logiciel IA](#10-philosophie-générale--de-la-prompting-traditionnelle-au-génie-logiciel-ia)
11. [Principes clés et takeaways](#11-principes-clés-et-takeaways)
12. [Références scientifiques et académiques](#12-références-scientifiques-et-académiques)

---

## 1. Vue d'ensemble et positionnement

### 1.1 Objectif central du cours

Ce cours **ne porte pas sur le prompting seul** — il enseigne comment **construire une application logicielle complète et maintenable** à partir d'une ou plusieurs invocations d'API ChatGPT.

**Évolution conceptuelle** :
- **Avant** : comment écrire un bon prompt pour ChatGPT (course précédente, "ChatGPT Prompt Engineering for Developers")
- **Maintenant** : comment assembler plusieurs appels LLM, avec gestion d'état, validations, boucles de feedback, et évaluation

### 1.2 Exemple fil rouge du cours

Tout le cours s'articule autour d'un **cas d'étude unifié** : **un système de service client end-to-end pour un détaillant en ligne** (électronique).

**Flux simplifié** :
```
Entrée utilisateur
    ↓
[Modération] → Vérifier pas de contenu problématique
    ↓
[Classification] → Identifier le type de requête (plainte, info produit, etc.)
    ↓
[Récupération] → Charger les données externes (catalogue produit)
    ↓
[Génération] → Écrire une réponse utile
    ↓
[Validation] → Vérifier pas d'hallucination, pas d'erreur factuelle
    ↓
Sortie utilisateur
```

Ce design reflète une vérité fondamentale : **une application réelle requiert plusieurs étapes invisibles à l'utilisateur**.

### 1.3 Thème transversal : visibilité interne vs. expérience utilisateur

> Une application souvent a besoin de **plusieurs étapes internes invisibles à l'utilisateur final**, traitant l'entrée de manière séquentielle pour produire une sortie finale présentée à l'utilisateur.

Exemples :
- Modération → évite les sorties offensantes avant qu'elles ne sortent
- Classification → permet au système de brancher logiquement sur différents chemins
- Récupération → augmente la qualité en injectant le contexte exact nécessaire
- Validation → réduit les hallucinations

---

## 2. Contexte : de la prompting unique aux systèmes complexes

### 2.1 Avantage des LLM pour le développement d'applications

Avant les LLM instruction-tunés, construire un classificateur pour un cas d'usage spécifique prenait :

| Phase | Temps estimé | Effort |
|-------|--------------|--------|
| Collecte de données labellisées | Semaines à mois | Humain intensif (contractors) |
| Entraînement du modèle | Jours à mois | Ressources compute massives |
| Déploiement en cloud | Jours | Infrastructure, DevOps |
| **Total classique** | **Mois à 1+ ans** | **Équipe large** |

**Avec prompting et API ChatGPT** :

| Phase | Temps estimé |
|-------|--------------|
| Écrire le prompt | Minutes à heures |
| Itérer sur le prompt | Heures |
| Déployer l'appel API | Heures |
| **Total** | **Heures à jours** |

**Citation d'Andrew Ng** (du cours) :
> *"La puissance des LLM comme outils pour les développeurs — l'utilisation d'appels API pour construire rapidement des logiciels — reste largement sous-estimée."*

### 2.2 Quand la prompting simple ne suffit pas

Une **invocation unique d'LLM** convient quand :
- La tâche est relativement simple et auto-contenue
- L'intention de l'utilisateur est claire
- Aucune validation/filtrage préalable n'est nécessaire

Une **architecture multi-étapes** devient indispensable quand :
- Plusieurs **décisions conditionnelles** branch selon la nature de l'entrée
- L'entrée doit être **validée ou nettoyée** avant traitement
- Le contexte nécessaire pour répondre doit être **récupéré dynamiquement**
- La sortie doit être **validée** avant remise à l'utilisateur
- L'application nécessite **mesure et amélioration itérative**

---

## 3. Fondamentaux : modèles de langage, format chat, et tokenisation

### 3.1 Base LLM vs. Instruction-Tuned LLM

**Base LLM** : entraîné à prédire le mot suivant sur d'énormes corpus textuels.

Exemple :
```
Prompt  : "Quelle est la capitale de la France ?"
Output  : "Quelle est la plus grande ville de France ? Quelle est la population..."
(Le modèle complète avec une suite plausible de questions sur le web)
```

**Instruction-Tuned LLM** (ex. ChatGPT, Claude) : partant d'une base LLM, fine-tuné sur des paires instruction → réponse correcte, puis amélioré via **RLHF** (Reinforcement Learning from Human Feedback).

Exemple :
```
Prompt  : "Quelle est la capitale de la France ?"
Output  : "La capitale de la France est Paris."
(Le modèle suit l'instruction et répond directement)
```

**Recommandation** : toujours utiliser des modèles instruction-tunés pour les applications. Les base LLM sont plus difficiles à contrôler et peuvent produire des sorties inattendues.

### 3.2 L'architecture des systèmes instruction-tunés

```
Base LLM
(100+ milliards de paramètres)
    ↓
[Fine-tuning supervisé]
(paires instruction → réponse correcte)
    ↓
[RLHF — Reinforcement Learning from Human Feedback]
(humains évaluent la qualité selon : utile, honnête, inoffensif)
    ↓
Instruction-Tuned LLM
(ex. GPT-3.5-Turbo, GPT-4)
```

**Durée de ce pipeline** :
- Base LLM : mois de calcul intensif
- Fine-tuning + RLHF : jours à semaines sur des ressources bien plus modestes

### 3.3 Le format "Chat" — system, user, assistant

Au lieu de simples prompts textuels, l'API Chat Completions de OpenAI accepte une **liste de messages structurée**, chacun avec un **rôle**.

**Structure générale** :

```python
messages = [
    {"role": "system", "content": "Vous êtes un assistant utile et courtois."},
    {"role": "user", "content": "Quelle est la capitale de la France ?"},
    {"role": "assistant", "content": "La capitale de la France est Paris."},
    {"role": "user", "content": "Et celle de l'Allemagne ?"}
]
```

**Rôles et fonctions** :

| Rôle | Fonction | Visibilité |
|------|----------|-----------|
| `system` | Définit le comportement global, la personnalité, les contraintes de l'assistant | Invisible pour l'utilisateur final |
| `user` | Messages envoyés par l'utilisateur humain | Visible |
| `assistant` | Réponses précédentes du modèle (pour contexte conversationnel) | Visible / Historique |

**Exemple avec instructions composées** :

```python
messages = [
    {
        "role": "system", 
        "content": """Vous êtes un assistant qui répond dans le style du Dr. Seuss.
                     Toutes vos réponses doivent être exactement une phrase."""
    },
    {
        "role": "user", 
        "content": "Écrivez-moi un court poème sur une carotte heureuse."
    }
]

# Output: "Oh, comme est joyeuse cette carotte que je vois, qui sourit 
#          toujours et n'a jamais peur de rien!"
```

**Métaphore du message système** : c'est comme "chuchoter à l'oreille" de l'assistant pour orienter son comportement sans que l'utilisateur le sache.

### 3.4 Tokenisation et ses implications

Les LLM ne traitent pas les caractères individuels ni les mots complets. Ils les **découpent en tokens**.

**Exemple concret** :
```
Phrase: "Learning new things is fun!"
Tokens: ["Learning", " new", " things", " is", " fun", "!"]
(Chaque mot ou mot+espace = 1 token)

Phrase: "Prompting as powerful developer tool"
Tokens: ["Prom", "pt", "ing", " as", " powerful", ...]
(Le mot "prompting", moins courant, est découpé en 3 tokens)

Phrase: "lollipop"
Tokens: ["l", "oll", "ipop"]
(D'où la difficulté pour ChatGPT à inverser les lettres)
```

**Règle empirique** : 1 token ≈ 4 caractères en anglais ≈ 0,75 mot.

**Implications pratiques** :

1. **Limites de contexte** : GPT-3.5-Turbo accepte ~4,000 tokens (entrée + sortie). GPT-4 en accepte jusqu'à 128,000.
2. **Coûts API** : vous payez par token. Les prompts plus longs coûtent plus cher.
3. **Comportements inattendus** : certaines tâches (ex. inversion de lettres) sont difficiles pour les modèles en raison de la tokenisation.

**Astuce pratique** pour les jeux de mots avec ChatGPT :
```
Au lieu de:  "Reverse the letters in 'lollipop'"
Faites :     "Reverse the letters in 'l-o-l-l-i-p-o-p'"
```

En séparant les lettres par des tirets, chacune devient un token distinct, facilitant le traitement.

### 3.5 Gestion sécurisée des clés API

❌ **À ÉVITER** :
```python
import os
# ❌ MAUVAIS
api_key = "sk-xxxxxx"
```

✅ **À FAIRE** :
```python
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
api_key = os.getenv('OPENAI_API_KEY')
```

Créer un fichier `.env` (non versionné dans git) :
```
OPENAI_API_KEY=sk-xxxxxx
```

**Bénéfice** : votre clé API ne s'expose jamais en texte clair dans les notebooks partagés.

---

## 4. Leçon 3 : Classification et catégorisation

### 4.1 Principe

Après avoir validé une entrée (voir section 5), le système doit souvent **identifier le type de requête** pour brancher sur différentes logiques métier.

**Classification typique** :
```
Entrée: "Mon téléphone ne charge plus"
         ↓
Classification: Type = "Problème technique" 
               Catégorie = "Support"
               Produit = "Téléphone"
```

### 4.2 Prompting pour la classification

**Approche simple** :

```python
def classify_query(user_message):
    system_prompt = """Vous êtes un assistant de classification.
    Analysez l'entrée utilisateur et répondez avec un objet JSON contenant :
    {
        "category": une de ces valeurs : ["Plainte", "Info produit", "Livraison", "Retour"],
        "product": si un produit est mentionné, son nom ; sinon null,
        "sentiment": "positif" | "neutre" | "négatif"
    }
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    
    # Parser la réponse JSON
    import json
    return json.loads(response.choices[0].message.content)

# Utilisation
result = classify_query("Je veux savoir si vous avez le téléphone SmartX en stock.")
print(result)
# {'category': 'Info produit', 'product': 'SmartX', 'sentiment': 'neutre'}
```

### 4.3 Extraction structurée et validation

Pour les applications réelles, on peut demander au modèle de retourner une **liste d'objets structurés** :

```python
system_prompt = """Vous êtes un assistant de catégorisation.
Pour chaque requête utilisateur, produisez une liste Python d'objets JSON.
Chaque objet doit contenir:
- "category": une de ["Informatique", "Électronique", "TV", "Autre"]
- "products": liste de produits mentionnés (doit être dans la liste autorisée)

Liste de produits autorisés:
- SmartX ProPhone (catégorie: Informatique)
- Fotosnap DSLR Camera (catégorie: Électronique)
- FotoSnap SkyView 200 (catégorie: TV)

IMPORTANT: Only output the list of objects. Nothing else.
"""

user_input = """Tell me about the SmartX ProPhone and the Fotosnap camera.
Also, what TVs do you have in stock?"""

# Réponse attendue:
# [
#   {"category": "Informatique", "products": ["SmartX ProPhone"]},
#   {"category": "Électronique", "products": ["Fotosnap DSLR Camera"]},
#   {"category": "TV", "products": []}  # ou tous les TV par défaut
# ]
```

**Avantage** : sortie structurée → facilement parsable en Python → branching logique automatisé.

---

## 5. Leçon 4 : Modération et sécurité des entrées

### 5.1 Objectif

Avant de traiter une requête utilisateur, **filtrer le contenu problématique** :
- Discours haineux, discrimination
- Contenu violent, explicite
- Spam, arnaque
- Langage offensant

### 5.2 Approche manuelle via prompting

```python
def moderate_input(user_message):
    system_prompt = """Vous êtes un modérateur de contenu.
    Analysez le message utilisateur et déterminez s'il viole une de ces règles:
    1. Discours haineux ou discrimination
    2. Violence ou menaces
    3. Contenu sexuel explicite
    4. Arnaque ou tromperie
    
    Répondez avec un objet JSON:
    {
        "flagged": true/false,
        "violations": [liste des catégories violées ou []]
        "severity": "low" | "medium" | "high" (si flagged=true)
    }
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    
    return json.loads(response.choices[0].message.content)
```

### 5.3 API de modération OpenAI

OpenAI fournit également une **API de modération dédiée** (plus efficace et robuste) :

```python
from openai import OpenAI

client = OpenAI()

def check_moderation(text):
    response = client.moderations.create(input=text)
    result = response.results[0]
    
    return {
        "flagged": result.flagged,
        "categories": {
            "hate": result.category_scores.hate,
            "violence": result.category_scores.violence,
            "sexual": result.category_scores.sexual,
            # ... autres catégories
        }
    }

# Utilisation
result = check_moderation("Je déteste les [groupe]. Qu'est-ce qu'on peut faire ?")
print(result)
# {
#   "flagged": True,
#   "categories": {
#     "hate": 0.92,  # 92% de confiance que c'est du discours haineux
#     ...
#   }
# }
```

**Décision d'action** :
- Si `flagged=True` → rejeter la requête ou logguer comme avertissement
- Si `flagged=False` → continuer au traitement suivant

---

## 6. Leçon 5 : Chain-of-Thought Reasoning

### 6.1 Principe fondamental

Les LLM produisent **de meilleures réponses** quand on leur demande d'** "penser étape par étape"** plutôt que de donner une réponse directe.

**Exemple simple** :

```
Prompt sans CoT:
"Un distributeur a 5 boîtes de livres. 
 Chaque boîte contient 30 livres.
 Combien de livres au total ?"

Output: "150"
(Peut être correct par chance, mais pas de trace de raisonnement)

---

Prompt avec CoT:
"Un distributeur a 5 boîtes de livres.
 Chaque boîte contient 30 livres.
 Combien de livres au total ?
 
 Pensez étape par étape et montrez votre travail."

Output: "
 Étape 1: Il y a 5 boîtes
 Étape 2: Chaque boîte contient 30 livres
 Étape 3: Nombre total = 5 × 30 = 150 livres
 Réponse: 150 livres
"
```

### 6.2 Implémentation dans une application

**Cas d'usage** : résoudre une plainte client (requiert plusieurs étapes logiques).

```python
def resolve_complaint(complaint_text):
    system_prompt = """Vous êtes un agent de service client expert.
    Pour chaque plainte, vous devez:
    1. Identifier le problème exact
    2. Lister les causes possibles
    3. Proposer 2-3 solutions
    4. Recommander la meilleure action
    
    Montrez toutes les étapes de votre raisonnement AVANT la recommendation finale.
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": complaint_text}
        ]
    )
    
    return response.choices[0].message.content

# Exemple
complaint = """J'ai commandé un téléphone SmartX ProPhone il y a 3 semaines.
J'ai le numéro de suivi, mais le statut n'a pas changé depuis 10 jours."""

output = resolve_complaint(complaint)
print(output)
# Output:
# Étape 1: Problème identifié → Livraison retardée/bloquée
# Étape 2: Causes possibles:
#   - Entrée en entrepôt retardée
#   - Problème avec le transporteur
#   - Adresse invalide
# Étape 3: Solutions:
#   a) Contactez le transporteur
#   b) Renvoyez une commande...
#   c) Offrez un remboursement partiel
# Étape 4: Meilleure action → Contactez le transporteur ET offrez 10% de remise...
```

### 6.3 Amélioration itérative du raisonnement

**Technique avancée** : forcer le modèle à **vérifier son propre travail** (self-verification).

```python
def solve_with_verification(problem):
    system_prompt = """Résolvez ce problème étape par étape.
    PUIS, relisez votre solution et vérifiez-la:
    - Votre logique est-elle correcte ?
    - Avez-vous oublié quelque chose ?
    - La réponse finale a-t-elle du sens ?
    
    Si vous trouvez une erreur, corrigez-la et expliquez la correction.
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem}
        ],
        temperature=0  # Déterministe
    )
    
    return response.choices[0].message.content
```

**Citation du cours** :
> *Chain-of-Thought Reasoning améliore la fiabilité non en changeant le modèle, mais en forçant une *structure* meilleure du processus de raisonnement.*

**Référence scientifique** : Wei, J., et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS.

---

## 7. Leçon 6 : Chaining Prompts — Architecture multi-étapes

### 7.1 Quand et pourquoi chaîner ?

Une **architecture multi-étapes** n'est pas une complication inutile. Elle offre des bénéfices concrets :

| Aspect | Single Prompt Complexe | Multi-step Chainé |
|--------|------------------------|-------------------|
| **Clarté** | Difficile (spaghetti code) | Modulaire, testable |
| **Coûts** | Élève (contexte complet) | Réduit (contexte sélectif) |
| **Limites de token** | Risque de dépassement | Mieux contrôlé |
| **Testabilité** | Chaque étape invisible | Chaque étape testable |
| **Outils externes** | Non | Oui (appels API, DB) |

### 7.2 Architecture du système client du cours

```
┌─ Entrée utilisateur ─────────────────────────────────┐
│ "Tell me about the SmartX ProPhone and TVs"         │
└──────────────────────────────────────────────────────┘
                        ↓
┌─ Étape 1: Classification ────────────────────────────┐
│ Système: Extract category & product names           │
│ → Output: [                                         │
│     {"category": "Informatique",                    │
│      "products": ["SmartX ProPhone"]},              │
│     {"category": "TV", "products": []}             │
│   ]                                                 │
└──────────────────────────────────────────────────────┘
                        ↓
┌─ Étape 2: Récupération (helper functions) ──────────┐
│ get_product_by_name("SmartX ProPhone")             │
│ get_products_by_category("TV")                     │
│ → Output: {"SmartX ProPhone": {...spec...},        │
│           "Samsung 55 TV": {...spec...}, ...}      │
└──────────────────────────────────────────────────────┘
                        ↓
┌─ Étape 3: Génération ────────────────────────────────┐
│ Système: You are a helpful electronics assistant   │
│ Context: [Produit info retrieved above]            │
│ User: Initial query                                │
│ → Output: "The SmartX ProPhone features... The TVs │
│           we have available are..."                │
└──────────────────────────────────────────────────────┘
                        ↓
┌─ Étape 4: Validation ────────────────────────────────┐
│ Vérifier pas d'hallucination, info correcte       │
│ → Output: {"valid": True, "issues": []}           │
└──────────────────────────────────────────────────────┘
                        ↓
┌─ Sortie utilisateur ────────────────────────────────┐
│ Réponse structurée & vérifiée                     │
└──────────────────────────────────────────────────────┘
```

### 7.3 Implémentation détaillée

**Étape 1 : Classification**

```python
def extract_product_category_names(user_message):
    system_message = """You will be provided with customer service queries.
    Output a Python list of objects where each object has:
    {
        "category": one of ["Computers", "Electronics", "Televisions", "Appliances"],
        "products": list of product names mentioned
    }
    
    Allowed products:
    Computers: SmartX ProPhone, TechPro Ultrabook
    Electronics: Fotosnap DSLR Camera, PowerGlow LED Lamp
    Televisions: Samsung 55", LG OLED 77"
    
    Only output the list. Nothing else.
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    )
    
    import json
    return json.loads(response.choices[0].message.content)

# Test
result = extract_product_category_names("Tell me about the SmartX ProPhone and TVs.")
print(result)
# [
#   {"category": "Computers", "products": ["SmartX ProPhone"]},
#   {"category": "Televisions", "products": []}
# ]
```

**Étape 2 : Récupération de produits**

```python
# Product database (en réalité, ce serait une vraie DB)
products = {
    "SmartX ProPhone": {
        "category": "Computers",
        "price": "$899",
        "specs": "6.1\" AMOLED, 5G, 256GB",
        "warranty": "2 years"
    },
    "Samsung 55\"": {
        "category": "Televisions",
        "price": "$699",
        "specs": "4K, QLED, 120Hz",
        "warranty": "3 years"
    }
    # ... plus de produits
}

def get_product_by_name(name):
    return products.get(name)

def get_products_by_category(category):
    return [p for p in products.values() if p.get("category") == category]

def generate_product_info_string(categories_and_products):
    """Convert parsed categories/products into human-readable product info."""
    output = ""
    
    for item in categories_and_products:
        if item.get("products"):
            # Specific products mentioned
            for product_name in item["products"]:
                product_info = get_product_by_name(product_name)
                if product_info:
                    output += f"\n{product_name}:\n{json.dumps(product_info, indent=2)}\n"
        else:
            # Category mentioned, load all products in category
            category = item.get("category")
            products_in_cat = get_products_by_category(category)
            for p in products_in_cat:
                output += f"\n{p.get('name', 'Unknown')}:\n{json.dumps(p, indent=2)}\n"
    
    return output

# Test
info = generate_product_info_string(result)
print(info)
# SmartX ProPhone:
# {
#   "category": "Computers",
#   "price": "$899",
#   ...
# }
# 
# Samsung 55":
# {...}
```

**Étape 3 : Génération de réponse**

```python
def generate_customer_response(user_message, product_info):
    system_message = """You are a helpful and friendly electronics customer service assistant.
    Respond concisely with relevant product information.
    Ask follow-up questions to clarify customer needs.
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": f"Relevant product information:\n{product_info}"}
        ]
    )
    
    return response.choices[0].message.content

# Test
response = generate_customer_response(
    "Tell me about the SmartX ProPhone and TVs.",
    info
)
print(response)
# Output:
# "The SmartX ProPhone is our flagship device with a 6.1\" AMOLED display, 
#  5G connectivity, and 256GB storage for $899. We also carry Samsung and LG 
#  TVs in 4K and OLED formats. What size TV interests you?"
```

### 7.4 Gestion des helper functions et appels externes

**Concept clé** : à chaque étape du chaînage, vous pouvez appeler des **fonctions Python ordinaires** pour récupérer ou transformer des données.

Exemples :
- Requêtes dans une base de données
- Appels API à des services tiers (météo, prix en temps réel)
- Calculs mathématiques complexes
- Recherche dans un index (embeddings)

**Avantage** : le modèle n'a jamais besoin d'halluciner des données — elles sont injectées réellement.

### 7.5 Technique avancée : text embeddings pour la récupération

**Problème** : avec l'approche précédente (match exact sur nom), on rate des variations :
```
Utilisateur: "Show me mobile phones"
Notre système: Ne trouve rien (cherche "mobile" exactement)
```

**Solution** : utiliser les **embeddings** pour recherche sémantique.

```python
from openai import OpenAI

client = OpenAI()

def search_products_by_query(query):
    """Recherche sémantique de produits via embeddings."""
    
    # 1. Créer embedding de la requête utilisateur
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding
    
    # 2. Créer embeddings de tous les produits (une seule fois en cache)
    product_embeddings = {}
    for name, spec in products.items():
        product_text = f"{name}. {spec['specs']}"
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=product_text
        ).data[0].embedding
        product_embeddings[name] = emb
    
    # 3. Calculer similarité cosinus
    import numpy as np
    
    query_vec = np.array(query_embedding)
    scores = {}
    for name, emb_vec in product_embeddings.items():
        product_vec = np.array(emb_vec)
        # Cosine similarity
        similarity = np.dot(query_vec, product_vec) / (
            np.linalg.norm(query_vec) * np.linalg.norm(product_vec)
        )
        scores[name] = similarity
    
    # 4. Retourner les top-3 produits
    top_products = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    return [name for name, score in top_products]

# Test
results = search_products_by_query("Show me a mobile phone for photography")
# → Peut retourner ["SmartX ProPhone", "Fotosnap DSLR Camera", ...]
# (Même sans match exact sur "mobile")
```

**Note** : une synthèse complète sur les embeddings sort du cadre de ce cours. Consultez le cours DeepLearning.AI dédié aux embeddings pour un approche approfondie.

---

## 8. Leçon 7 : Validation et vérification des sorties

### 8.1 Problème central : hallucinations

Une **hallucination LLM** est une affirmation qui semble plausible, bien formée, mais factuellement **fausse ou inventée**.

**Exemple** :
```
Utilisateur: "Tell me about the TechPro Ultrabook Max"
Modèle: "The TechPro Ultrabook Max features a 17-inch display, 
         RTX 4090, and costs $999. Available in black and silver."
Réalité: Ce produit n'existe pas dans notre catalogue.
```

**Pourquoi c'est dangereux** :
- Crédibilité perdue si un client découvre l'invention
- Responsabilité légale (fausses affirmations produit)
- Expérience client dégradée

### 8.2 Tactique 1 : Validation par prompting

```python
def validate_response_factual(response, product_info):
    """Demander au modèle de vérifier que sa réponse utilise seulement les infos fournies."""
    
    system_message = """You are a fact-checker.
    Given a response and a product information database,
    check if the response ONLY uses information from the database.
    
    Return JSON:
    {
        "valid": true/false,
        "issues": [list of hallucinations or errors],
        "explanation": "Brief explanation"
    }
    """
    
    verification_prompt = f"""Response to check:
{response}

Product information provided:
{product_info}

Does the response only reference products and specs from the information above?"""
    
    result = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": verification_prompt}
        ]
    )
    
    return json.loads(result.choices[0].message.content)

# Test
check = validate_response_factual(
    "The TechPro Ultrabook Max costs $999.",
    product_info
)
print(check)
# {"valid": false, "issues": ["Product 'TechPro Ultrabook Max' not in database"], ...}
```

### 8.3 Tactique 2 : Cite your sources

```python
def generate_with_citations(user_message, product_info):
    """Demander au modèle de citer ses sources."""
    
    system_message = """You are a helpful assistant.
    IMPORTANT: You must cite which specific product information you are using
    to answer each part of the response.
    
    Format:
    - For each claim, include [Source: Product Name - specific field]
    - If information is not in the provided database, say "I don't have that info"
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": f"Product database:\n{product_info}"}
        ]
    )
    
    return response.choices[0].message.content

# Output exemple:
# "The SmartX ProPhone costs $899 [Source: SmartX ProPhone - price field]
#  with 256GB storage [Source: SmartX ProPhone - specs field]."
```

### 8.4 Tactique 3 : RAG (Retrieval-Augmented Generation)

L'approche du cours (Étapes 1-3) est déjà une forme de **RAG** :

```
Question utilisateur
        ↓
[Récupération] → Charger SEUL le contexte pertinent
        ↓
[Génération] → Répondre en utilisant ce contexte
```

Comparé à :
```
[Sans RAG] Question → [Génération sur tout le corpus] → Hallucination probable
```

**RAG réduit les hallucinations** parce que :
- Le modèle n'invente pas les données — elles sont explicitement fournies
- Le modèle peut refuser ("Je ne vois pas cette info fournie")

---

## 9. Leçon 8–10 : Évaluation et amélioration itérative

### 9.1 Framework d'évaluation

Une bonne application LLM nécessite un **processus continu d'évaluation et d'amélioration**.

```
Application en production
        ↓
[Collecter sorties] → Logguer toutes les réponses du modèle
        ↓
[Évaluer] → Vérifier si chaque sortie était bonne/mauvaise
        ↓
[Analyser] → Identifier tendances d'erreur
        ↓
[Améliorer] → Affiner prompts, logique, ou données
        ↓
[Tester] → Vérifier amélioration sur cas problématiques
        ↓
[Déployer] → Redéployer la version améliorée
```

### 9.2 Métriques d'évaluation

**Pour un système de service client** :

| Métrique | Définition | Évaluation |
|----------|-----------|-----------|
| **Correctness** | La réponse est-elle factuellement correcte ? | Human review ou LLM-based |
| **Relevance** | La réponse adresse-t-elle la question ? | Human review |
| **Completeness** | Tous les aspects sont-ils couverts ? | Human review ou checklist |
| **Tone** | Le ton est-il approprié (amical, professionnel) ? | Human review |
| **Hallucination rate** | % de réponses contenant des inventions | LLM-based verification |

### 9.3 Evaluation avec l'IA

```python
def evaluate_response_quality(response, original_question, reference_answer=None):
    """Utiliser un LLM pour évaluer la qualité d'une réponse."""
    
    system_message = """You are an expert evaluator of customer service responses.
    Evaluate the given response on these criteria (1-5 scale):
    
    1. Correctness: Is it factually accurate?
    2. Relevance: Does it address the question?
    3. Completeness: Are all aspects covered?
    4. Tone: Is it friendly and professional?
    5. Hallucination: Does it invent information?
    
    Return JSON with scores and brief justifications.
    """
    
    prompt = f"""Question: {original_question}
    
Response: {response}

{f"Reference answer (if available): {reference_answer}" if reference_answer else ""}

Evaluate on all criteria."""
    
    result = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )
    
    return json.loads(result.choices[0].message.content)

# Test
evaluation = evaluate_response_quality(
    "The TechPro Ultrabook costs $999 and has 16GB RAM.",
    "What's the price of the TechPro Ultrabook?",
    reference_answer="$999"
)
print(evaluation)
# {
#   "correctness": 5,
#   "relevance": 5,
#   "completeness": 4,
#   "tone": 4,
#   "hallucination": 1
# }
```

### 9.4 Processus d'amélioration

**Situation courante** : on observe que le modèle fait des erreurs dans un cas spécifique.

**Processus itératif** :

```
1. Identifier le problème
   "Le modèle invente souvent les stocks disponibles"

2. Formuler une hypothèse
   "Peut-être qu'un 'stock database' explicite aiderait"

3. Tester l'hypothèse
   - Ajouter une étape de récupération du stock
   - Évaluer sur 50 exemples précédents
   - Comparer métrique "hallucination rate"

4. Analyser résultats
   - Hallucination rate: 15% → 3% ✓
   - Temps de réponse: 0.5s → 0.8s (acceptable)

5. Déployer si amélioration confirmée
   - Mettre à jour production
   - Monitorer en continu
```

### 9.5 Gestion des cas limites

Certains cas sont **inévitablement difficiles**. Plutôt que de forcer une réponse fallacieuse, privilégier l'honnêteté :

```python
def generate_with_fallback(user_message, product_info):
    """Permettre au modèle de refuser si l'info n'est pas suffisante."""
    
    system_message = """You are a customer service assistant.
    
    CRITICAL: If the user asks something and you don't have the information in the database,
    you MUST say "I don't have that information" or "I'll need to check with my manager".
    
    NEVER invent product specs, prices, or availability.
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": f"Available info:\n{product_info}"}
        ]
    )
    
    return response.choices[0].message.content

# Exemple:
# User: "Do you have the foobar 9000?"
# Model: "I don't have the foobar 9000 in our current inventory. 
#         Let me check with my manager or you can contact support directly."
```

---

## 10. Philosophie générale : de la prompting traditionnelle au génie logiciel IA

### 10.1 Révolution des temps de développement

**Avant les LLM instruction-tunés** (machine learning classique) :

```
Défaut de spécification
        ↓
[Collecte de données] → Mois/années
        ↓
[Entraînement] → Semaines/mois
        ↓
[Déploiement] → Jours/semaines
        ↓
[Itération] → Semaines/mois pour chaque boucle

Total: 6 mois → 2 ans pour un simple classificateur
```

**Avec LLM et prompting** :

```
Défaut de spécification
        ↓
[Écrire prompt] → Minutes/heures
        ↓
[Itérer] → Heures
        ↓
[Déployer via API] → Heures
        ↓
[Itération] → Heures/jours pour améliorements

Total: Heures à jours pour une application complète
```

**Citation du cours** (Andrew Ng) :
> *"Des applications qu'on mettait 6 mois à 1 an à construire peuvent maintenant être construites en heures ou jours."*

### 10.2 Implications architecturales

**Paradigme classique ML** :
```
Données → Modèle → Prédictions
(Coûteux, long)
```

**Paradigme LLM + Prompting** :
```
Données + Prompts + Raison code externe → Réponse
(Rapide, itératif, moins coûteux)
```

### 10.3 Limite importante : données structurées

⚠️ **Le cours insiste** : cette approche ne fonctionne **bien que pour les données non-structurées** (texte, images).

Pour les **données structurées/tabulaires** (feuilles de calcul Excel, BD), le machine learning classique reste supérieur.

---

## 11. Principes clés et takeaways

### 11.1 Les 5 piliers d'un système LLM robuste

1. **Modération en entrée** → Éviter contenu problématique avant traitement
2. **Classification/Routage** → Diriger les requêtes correctement
3. **Récupération contextuelle** → Injecter seulement l'info pertinente
4. **Génération avec raison** → Chaîner plusieurs prompts intelligemment
5. **Validation en sortie** → Vérifier factualité et absence d'hallucination

### 11.2 Quand chaîner vs. single prompt ?

**Chaîner est préférable si** :
- Décisions conditionnelles multiples (routage)
- Contexte externe à charger dynamiquement
- Étapes testables/debuggables nécessaires
- Limites de token critiques (coûts)

**Single prompt suffit si** :
- Tâche simple et autonome
- Pas de récupération de contexte
- Pas d'appels à outils externes
- Peu de risque d'hallucination

### 11.3 API Key Security

✅ **Toujours** utiliser `.env` + `load_dotenv()` — jamais de clés hardcodées.

### 11.4 Tokenisation et ses implications

- 1 token ≈ 4 caractères / 0.75 mot
- Les LLM traitent tokens, pas caractères
- Implication : certaines tâches (inversion de lettres) sont difficiles
- Astuce : ajouter délimiteurs pour forcer tokenisation favorable

### 11.5 Le format Chat vs. autres formats

**Format Chat** (système + user + assistant) est la norme moderne :
- Permet de définir rôles clairs
- Facilite conversations multi-tour
- Flexible pour prompts complexes

### 11.6 Chain-of-Thought est universel

Demander au modèle de "penser étape par étape" améliore presque toujours la qualité, même quand il n'y a pas d'algorithme à montrer. C'est une technique transférable.

### 11.7 L'itération continue est essentielle

Il n'existe pas de "prompt parfait" universel. Le succès vient d'un **processus itératif systématique** : 
- Formuler
- Exécuter  
- Évaluer l'écart
- Ajuster
- Recommencer

### 11.8 RAG est la clé pour réduire hallucinations

Plutôt que de demander au modèle de "mémoriser" tous les produits, **récupérez dynamiquement** le contexte pertinent à chaque requête.

---

## 12. Références scientifiques et académiques

### 12.1 Articles fondateurs cités ou impliqués

1. **Wei, J., Wang, X., Schuurmans, D., et al.** (2022).  
   *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.*  
   Advances in Neural Information Processing Systems (NeurIPS).  
   **Impact** : Fondation théorique du Chain-of-Thought Reasoning.

2. **Ouyang, L., Wu, J., Jiang, X., et al.** (2022).  
   *Training language models to follow instructions with human feedback.*  
   OpenAI / NeurIPS.  
   **Impact** : Fondation de RLHF et instruction-tuning (GPT-3.5, ChatGPT).

3. **Christiano, P., Leike, J., Brown, T., et al.** (2017).  
   *Deep Reinforcement Learning from Human Preferences.*  
   NeurIPS.  
   **Impact** : Technique RLHF originale.

4. **Brown, T., Mann, B., Ryder, N., et al.** (2020).  
   *Language Models are Few-Shot Learners.*  
   NeurIPS (GPT-3).  
   **Impact** : Démonstration du few-shot et zero-shot learning.

5. **Lewis, P., Perez, E., Rinott, R., et al.** (2020).  
   *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*  
   NeurIPS.  
   **Impact** : Base théorique de RAG (mentionné implicitement dans le cours).

6. **Vaswani, A., Shazeer, N., Parmar, N., et al.** (2017).  
   *Attention Is All You Need.*  
   NeurIPS.  
   **Impact** : Architecture Transformer sous-jacente à tous les LLM modernes.

### 12.2 Documentation OpenAI officielle

- [OpenAI Chat Completions API](https://platform.openai.com/docs/guides/gpt)
- [Moderation API](https://platform.openai.com/docs/guides/moderation)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook) — Best practices & exemples

### 12.3 Ressources complémentaires

- **DeepLearning.AI — Embeddings course** (pour RAG avancé & semantic search)
- **Stanford CS224N** (NLP et LLMs) — https://web.stanford.edu/class/cs224n/
- **OpenAI & DeepLearning.AI Research** — https://deeplearning.ai/research/

---

## Conclusion : Les trois révolutions du cours

### Révolution 1 : Temps de développement

Construire une application IA n'est plus un projet de 6-12 mois. C'est un projet d'**heures à jours**.

### Révolution 2 : Architecture

Les applications LLM ne sont pas un simple appel d'API. Ce sont des **systèmes multi-étapes** avec modération, classification, récupération, génération, et validation.

### Révolution 3 : Maintenabilité

Le paradigme "écrire une fois, déployer partout" change en **"itérer continuellement"**. Les meilleurs systèmes LLM sont ceux avec un processus d'évaluation et amélioration robuste.

---

**Créée par** : Synthèse du cours DeepLearning.AI / OpenAI "Building Systems with the ChatGPT API"  
**Synthesized** : Juin 2026  
**Level** : PhD-Ready (Technical, Comprehensive, Implementation-Focused)
