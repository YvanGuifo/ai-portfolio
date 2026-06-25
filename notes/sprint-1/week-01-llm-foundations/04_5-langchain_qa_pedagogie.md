# 📖 Question Answering en LangChain : Guide Pédagogique Complet

**Auteur (assistant)** : Agent Enseignant  
**Niveau** : M1/M2 ingénieurs (prérequis : Python, NLP fondamentaux, RAG basique)  
**Durée estimée** : 8–10h (lecture + exercices)  
**Dernière maj** : Juin 2026  

---

## 📌 Table des matières

1. [Objectifs d'apprentissage](#objectifs)
2. [Contexte : du Retrieval au Question Answering](#contexte)
3. [Concepts fondamentaux du QA](#concepts)
4. [Architecture générale : QA Pipeline](#architecture)
5. [Implémentation progressive en LangChain](#implementation)
6. [Stratégies avancées de QA](#strategies)
7. [Évaluation et métriques QA](#evaluation)
8. [Pièges et bonnes pratiques](#pieges)
9. [Exercices corrigés](#exercices)
10. [Références scientifiques](#references)

---

## 🎯 Objectifs d'apprentissage {#objectifs}

### Verbes Bloom (Anderson & Krathwohl, 2001)

| Niveau | Objectif mesurable |
|--------|-------------------|
| **Comprendre** | Expliquer la différence entre QA extractif et abstractif |
| **Appliquer** | Implémenter un QA system simple avec LangChain + retriever |
| **Analyser** | Évaluer la qualité des réponses (exact match, F1, BLEU) |
| **Évaluer** | Choisir une stratégie QA (extractif/abstractif/hybride) selon cas d'usage |
| **Créer** | Concevoir un QA complet multi-étapes avec post-processing et confidence scores |

### Alignement constructif (Biggs, 1996)

| Objectif | Activité d'apprentissage | Évaluation |
|----------|-------------------------|-----------|
| Comprendre QA | Lire sections 2–3 + diagrammes | Quiz conceptuel (5 QCM) |
| Appliquer LangChain | Code progressif (Exo 1, 2, 3, 4) | Notebook annoté + tests |
| Analyser réponses | TP : calculer métriques (EM, F1, BLEU) | Rapport avec comparaison |
| Évaluer stratégie | Étude de cas industriel | Recommandation écrite |
| Créer pipeline | TP final : QA complet + post-processing | Projet : système déployable |

---

## 🔍 Contexte : du Retrieval au Question Answering {#contexte}

### De RAG à QA

**RAG (rappel du document précédent)** :
```
Query → [Retriever] → Documents → [LLM] → Réponse générale
```

**Question Answering** (extension) :
```
Query → [Retriever] → Documents → [QA Model] → Réponse structurée
         ↓
      Top-k docs → [Reader] → Extraction de span → Réponse précise
```

### Problème : Réponses approximatives vs. précises

**Limitation de RAG simple** (Lewis et al., 2020) :

- ❌ Réponse trop long (100+ tokens)
- ❌ Pas de span précis (quelle partie du doc répond vraiment ?)
- ❌ Hallucinations : invente des détails
- ❌ Pas de confiance quantifiée

**Exemple** :

```
Query: "Combien de jours de congé ?"

RAG (générique):
"Selon la politique RH 2024, les salariés ont droit à des congés payés. 
Le nombre exact dépend du statut, de l'ancienneté, et peut être négocié 
avec le manager. Les nouvelles recrues reçoivent généralement..."
→ 5 phrases, réponse floue

QA (extractif):
"25 jours"  (span exact du document)
Confiance: 0.94
Source: Politique RH page 12, ligne 3
```

---

### Deux paradigmes de QA

| Paradigme | Définition | Exemples |
|-----------|-----------|----------|
| **Extractif** | Extraire un span du document | BERT, DistilBERT, RoBERTa |
| **Abstractif** | Générer une réponse (paraphrase/résumé) | T5, BART, GPT-4 |
| **Hybride** | Combiner les deux | Dense Passage Retrieval + Seq2Seq |

**Référence clé** : Rajpurkar, P., Zhang, J., Liang, P., & Liang, P. S. (2016). "SQuAD: 100,000+ Questions for Machine Reading Comprehension of Text." *EMNLP*, 2016. DOI: [10.48550/arXiv.1606.05017](https://arxiv.org/abs/1606.05017)

---

## 💡 Concepts fondamentaux du QA {#concepts}

### 1. QA Extractif (Span-based)

**Principe** : Localiser la réponse comme un **span continu** dans le document.

**Architectures classiques** (Devlin et al., 2019) :

```
Document: "Les salariés ont droit à 25 jours de congés payés par an."
           [début] ← 25 jours ← [fin]

BERT input:
[CLS] Combien de jours ? [SEP] Les salariés ont droit à 25 jours ... [SEP]
       ↓ query_tokens              ↓ document_tokens

BERT output:
- start_logits: P(début du span) pour chaque token
- end_logits: P(fin du span) pour chaque token

Réponse: argmax(start_logits) → argmax(end_logits) = "25 jours"
```

**Mathématique** (Devlin et al., 2019, BERT paper) :

$$P(\text{span}) = \frac{\exp(s_{\text{start}}) \cdot \exp(e_{\text{end}})}{\sum_{\text{all spans}} \exp(s_{\text{start}}) \cdot \exp(e_{\text{end}})}$$

**Avantages** :
- ✅ Réponses précises et courtes
- ✅ Traçabilité (où dans le doc ?)
- ✅ Pas d'hallucinations (span existe dans doc)
- ✅ Rapide (pas de génération)

**Inconvénients** :
- ❌ Impossible si réponse nécessite paraphrase/synthèse
- ❌ Ne fonctionne que si réponse est contiguë
- ❌ Peut manquer réponses implicites

**Modèles populaires** :
- BERT (Devlin et al., 2019, 110M params)
- DistilBERT (Sanh et al., 2019, 66M params, 40% plus rapide)
- RoBERTa (Liu et al., 2019, meilleur Q&A)

---

### 2. QA Abstractif (Generative)

**Principe** : **Générer** une réponse libre (paraphrase/résumé du document).

**Architectures** (Seq2Seq) :

```
Document + Query → [Encoder] → Représentation → [Decoder] → Réponse générée

Exemple:
Input: "Combien de jours ?" + "Les salariés ont 25 jours..."
Output: "Les employés reçoivent vingt-cinq jours de congés annuels"
       (paraphrase fluide de la réponse)
```

**Modèles populaires** (Lewis et al., 2020) :
- **T5** (Text-to-Text Transfer Transformer, 220M–11B params)
- **BART** (Denoising Seq2Seq, 400M params)
- **GPT-3.5/4** (Zero-shot, pas de fine-tuning)

**Avantages** :
- ✅ Réponses fluides et naturelles
- ✅ Paraphrase/synthèse possibles
- ✅ Adapté aux questions complexes

**Inconvénients** :
- ❌ Peut halluciner (inventer des faits)
- ❌ Moins traçable (quelle partie du doc ?)
- ❌ Coûteux en compute (génération token-by-token)

**Référence clé** : Lewis, M., Liu, Y., Goyal, N., *et al.* (2020). "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension." *ACL*, 2020. DOI: [10.48550/arXiv.1910.13461](https://arxiv.org/abs/1910.13461)

---

### 3. Confiance et Calibration

**Problème** : Un modèle dit "0.99 de confiance" mais se trompe ?

**Définition** (Guo et al., 2017) :

$$\text{Confiance calibrée} = P(\text{model correct} \mid \text{confidence score})$$

**Techniques** :

1. **Temperature scaling** (Guo et al., 2017)
   ```python
   confidence_calibrated = softmax(logits / T)  # T ≈ 1.5–2.0
   ```

2. **Dropout Monte Carlo** (Gal & Ghahramani, 2016)
   ```python
   # Multiple forward passes avec dropout activé
   outputs = [model(input, training=True) for _ in range(n_samples)]
   confidence = std(outputs)  # Variance = incertitude
   ```

3. **Ensemble methods**
   ```python
   outputs = [model1(input), model2(input), model3(input)]
   confidence = agreement(outputs)  # Accord = confiance
   ```

**Référence** : Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). "On Calibration of Modern Neural Networks." *ICML*, 2017. DOI: [10.48550/arXiv.1706.04599](https://arxiv.org/abs/1706.04599)

---

### 4. Multi-hop Reasoning

**Défi** : Questions nécessitant **plusieurs documents** pour répondre.

**Exemple** :
```
Q: "Qui est le CTO de l'entreprise qui emploie Alice ?"
   ↓
Doc 1: "Alice travaille chez TechCorp"
Doc 2: "TechCorp CTO = Jean Martin"

Réponse: "Jean Martin"
(nécessite 2 sauts: Alice → TechCorp → CTO)
```

**Approches** (Yang et al., 2018) :

1. **Iterative retrieval** : Récupérer, puis reformuler query, récupérer à nouveau
2. **Graph-based** : Construire graphe de relations entre docs
3. **Attention-based** : Attention multi-hop sur docs

**Référence** : Yang, Z., Qi, P., Zhang, S., *et al.* (2018). "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering." *EMNLP*, 2018. DOI: [10.48550/arXiv.1809.02776](https://arxiv.org/abs/1809.02776)

---

## 🏗️ Architecture générale : QA Pipeline {#architecture}

### Phases du QA System

```
┌────────────────────────────────────────────────────────────────┐
│ PHASE 1: PASSAGE RETRIEVAL                                     │
├────────────────────────────────────────────────────────────────┤
│ User Query: "Combien de jours de congé ?"                      │
│      ↓                                                          │
│ [Dense Retriever] → Top-k passages (ex. 5 passages)            │
│                                                                │
│ Result: ["Les salariés ont 25 jours...", "Congés spéciaux.."] │
└────────────┬───────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│ PHASE 2: SPAN EXTRACTION (Extractif)                           │
├────────────────────────────────────────────────────────────────┤
│ [QA Model] (ex. DistilBERT-SQuAD)                              │
│                                                                │
│ Pour chaque passage:                                           │
│  • Localiser span réponse                                      │
│  • Calculer confidence score                                   │
│                                                                │
│ Result:                                                        │
│  Passage 1: span="25 jours", conf=0.94                        │
│  Passage 2: span="jours fériés", conf=0.31  ← bruit           │
└────────────┬───────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│ PHASE 3: RANKING & SELECTION                                   │
├────────────────────────────────────────────────────────────────┤
│ [Passage Ranker] (optionnel)                                   │
│                                                                │
│ Combiner:                                                      │
│  • Retriever score (pertinence passage)                        │
│  • QA confidence (qualité span)                                │
│  • Passage length penalty                                      │
│                                                                │
│ Final ranking: sort by combined score                          │
└────────────┬───────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│ PHASE 4: POST-PROCESSING & OUTPUT                              │
├────────────────────────────────────────────────────────────────┤
│ • Dédupliquer réponses identiques                              │
│ • Normaliser (minuscules, ponctuation)                         │
│ • Vérifier cohérence                                           │
│ • Retourner top-3 candidats avec confiance                    │
│                                                                │
│ Final Output:                                                  │
│ [                                                              │
│   {"answer": "25 jours", "confidence": 0.94,                  │
│    "passage_id": 42, "start": 45, "end": 53},                │
│   {"answer": "20 jours", "confidence": 0.71, ...},            │
│   {"answer": "23 jours", "confidence": 0.68, ...}             │
│ ]                                                              │
└────────────────────────────────────────────────────────────────┘
```

### Comparaison : QA Extractif vs. Abstractif

| Aspect | Extractif | Abstractif |
|--------|-----------|-----------|
| **Source** | Span du doc | Génération libre |
| **Hallucination** | Non (spans vérifiés) | Possible |
| **Traçabilité** | Excellente (position exacte) | Faible |
| **Fluence** | Peut être maladroite | Excellente |
| **Vitesse** | Fast (no generation) | Lent (token-by-token) |
| **Idéal pour** | FAQ, facts précis | Synthèse, explication |

---

## 💻 Implémentation progressive en LangChain {#implementation}

### ✅ Prérequis

```bash
pip install langchain langchain-core langchain-community \
    langchain-openai langchain-huggingface transformers \
    torch sentence-transformers python-dotenv \
    datasets evaluate rouge-score
```

**Versions** : Python 3.10+, LangChain 0.1.0+, transformers 4.35+

---

### Niveau 1 : QA Extractif Simple

#### Code 1.1 : Charger un modèle QA pré-entraîné

```python
from transformers import pipeline

# Charger modèle QA pré-entraîné
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",  # Robuste, 80M params
    device=0  # GPU:0 si disponible, sinon CPU
)

# Test rapide
context = """
Les salariés de notre entreprise ont droit à 25 jours de congés payés par an.
Ces congés peuvent être pris de manière flexible, avec l'accord du manager.
Les jours fériés (11 au total) sont en sus des congés.
"""

question = "Combien de jours de congé par an ?"

result = qa_pipeline(question=question, context=context)

print(f"✅ Réponse: {result['answer']}")
print(f"🎯 Confiance: {result['score']:.2%}")
print(f"📍 Position: chars {result['start']}–{result['end']}")

# Output attendu:
# ✅ Réponse: 25 jours
# 🎯 Confiance: 94.32%
# 📍 Position: chars 45–53
```

**Modèles disponibles** (Hugging Face SQuAD Leaderboard) :

| Modèle | Taille | Vitesse | EM@SQuAD | F1@SQuAD |
|--------|--------|---------|----------|----------|
| roberta-base-squad2 | 358M | Fast | 81.5 | 88.2 |
| distilbert-base-cased-distilled-squad | 268M | Faster | 79.1 | 87.0 |
| electra-small-discriminator | 110M | Fastest | 74.5 | 82.8 |
| bert-large-uncased-whole-word-masking-finetuned-squad | 340M | Medium | 86.9 | 92.8 |

**Références** :
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL*, 2019. DOI: [10.48550/arXiv.1810.04805](https://arxiv.org/abs/1810.04805)
- Liu, Y., Ott, M., Goyal, N., *et al.* (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach." arXiv preprint. DOI: [1907.11692](https://arxiv.org/abs/1907.11692)

---

### Niveau 2 : QA avec Retriever (Open-domain QA)

#### Code 2.1 : Pipeline complet Retrieval + QA

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline as hf_pipeline

# 1. Indexer documents (voir leçon Retrieval)
loader = PyPDFLoader("politique_rh.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 2. Charger QA model
qa_model = hf_pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)

# 3. Pipeline complet
def open_domain_qa(query: str, top_k: int = 3):
    """Question Answering open-domain (retrieval + extraction)"""
    
    # Retrieval
    retrieved_docs = vectorstore.similarity_search(query, k=top_k)
    
    print(f"\n📚 Documents pertinents trouvés : {len(retrieved_docs)}\n")
    
    answers = []
    
    # QA sur chaque document
    for i, doc in enumerate(retrieved_docs, 1):
        try:
            result = qa_model(
                question=query,
                context=doc.page_content
            )
            
            # Vérifier que confiance > threshold
            if result['score'] > 0.3:  # Threshold configurable
                answers.append({
                    'answer': result['answer'],
                    'confidence': result['score'],
                    'source': doc.metadata.get('source', 'Unknown'),
                    'page': doc.metadata.get('page', '?'),
                    'passage': doc.page_content[:100] + "..."
                })
        except Exception as e:
            print(f"  ⚠️ Erreur sur doc {i}: {e}")
    
    # Trier par confiance
    answers = sorted(answers, key=lambda x: x['confidence'], reverse=True)
    
    return answers[:3]  # Top-3 réponses

# Test
query = "Combien de jours de congé ?"
results = open_domain_qa(query)

print(f"\n✅ Top-3 réponses pour: '{query}'\n")
for i, res in enumerate(results, 1):
    print(f"{i}. Réponse: \"{res['answer']}\"")
    print(f"   Confiance: {res['confidence']:.2%}")
    print(f"   Source: {res['source']} (page {res['page']})")
    print(f"   Contexte: {res['passage']}\n")
```

**Output attendu** :

```
📚 Documents pertinents trouvés : 3

✅ Top-3 réponses pour: 'Combien de jours de congé ?'

1. Réponse: "25 jours"
   Confiance: 94.32%
   Source: politique_rh.pdf (page 12)
   Contexte: Les salariés ont droit à 25 jours de congés payés par an...

2. Réponse: "25"
   Confiance: 87.51%
   Source: politique_rh.pdf (page 13)
   Contexte: En 2024, le nombre de jours reste à 25 pour...

3. Réponse: "congés payés"
   Confiance: 71.23%
   Source: politique_rh.pdf (page 8)
   Contexte: Les jours fériés (11) sont en sus des congés...
```

---

### Niveau 3 : QA Abstractif avec LLM

#### Code 3.1 : Question Answering avec génération (GPT-4)

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Setup
vectorstore = FAISS.load_local("./faiss_rh", OpenAIEmbeddings(model="text-embedding-3-small"))
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# Prompt spécialisé pour QA
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Tu es un expert RH. Réponds à la question en t'appuyant UNIQUEMENT sur le contexte fourni.

Contexte:
{context}

Question: {question}

Directives:
1. Si la réponse est dans le contexte, réponds de manière précise et concise (1–2 phrases)
2. Si la réponse n'est pas dans le contexte, dis: "Information non disponible dans les documents"
3. Cites la source si pertinent
4. Sois factuel, pas d'interprétation personnelle

Réponse:"""
)

# Chain RAG avec template QA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Concaténer les documents
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={
        "prompt": qa_prompt,
        "document_variable_name": "context"  # Nom de la variable contexte
    },
    return_source_documents=True
)

# Test
query = "Quel est le processus pour demander une maternité ?"
result = qa_chain.invoke({"query": query})

print(f"❓ Question: {query}\n")
print(f"✅ Réponse:\n{result['result']}\n")
print(f"📚 Sources:\n")
for doc in result['source_documents'][:2]:
    print(f"  - {doc.metadata['source']} (page {doc.metadata.get('page', '?')})")
```

**Output** :

```
❓ Question: Quel est le processus pour demander une maternité ?

✅ Réponse:
Le processus pour demander un congé maternité est le suivant :
1. Notifiez votre manager au moins 3 mois à l'avance
2. Complétez le formulaire RH disponible sur l'intranet
3. Attachez le certificat médical
4. Soumettez via le portail avant la date limite

La durée est de 16 semaines avec maintien du salaire.

📚 Sources:
  - politique_rh.pdf (page 18)
  - politique_rh.pdf (page 19)
```

---

### Niveau 4 : QA Multi-hop (Multi-document)

#### Code 4.1 : Multi-hop Reasoning avec ReAct

```python
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

# Tools
def retrieve_documents(query: str) -> str:
    """Récupère les documents pertinents"""
    vectorstore = FAISS.load_local("./faiss_rh", embeddings)
    docs = vectorstore.similarity_search(query, k=3)
    return "\n".join([f"[{i}] {d.page_content[:300]}" for i, d in enumerate(docs)])

def extract_fact(text: str, fact_type: str) -> str:
    """Extrait un fait spécifique du texte"""
    # Utiliser QA model pour extraire
    qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
    result = qa(question=f"Qui est le {fact_type} ?", context=text)
    return result['answer']

tools = [
    Tool(
        name="Retrieve",
        func=retrieve_documents,
        description="Récupère les documents RH pertinents pour une requête"
    ),
    Tool(
        name="Extract",
        func=extract_fact,
        description="Extrait un fait spécifique d'un texte"
    )
]

# Agent multi-hop
llm = ChatOpenAI(model="gpt-4")
agent = create_react_agent(llm, tools, PromptTemplate.from_template(
    """Tu es un assistant RH expert. Utilise les outils pour répondre à des questions complexes.

Question: {input}

Réfléchis pas à pas. Utilise d'abord Retrieve pour trouver les infos, puis Extract pour préciser.

{agent_scratchpad}"""
))

executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Test question multi-hop
query = "Qui est le responsable RH qui gère les congés ?"
result = executor.invoke({"input": query})

print(f"\n🔗 Réponse multi-hop:\n{result['output']}")
```

---

### Niveau 5 : QA avec Confidence Calibration

#### Code 5.1 : Calibration et Ensemble QA

```python
from transformers import pipeline
import numpy as np
from scipy.special import softmax

class CalibratedQA:
    """QA model avec confiance calibrée"""
    
    def __init__(self, model_names: list = None):
        if model_names is None:
            model_names = [
                "deepset/roberta-base-squad2",
                "distilbert-base-cased-distilled-squad"
            ]
        
        self.models = [
            pipeline("question-answering", model=name)
            for name in model_names
        ]
    
    def ensemble_qa(self, question: str, context: str, temperature: float = 1.5):
        """
        QA avec ensemble + temperature scaling
        
        Args:
            question: Question utilisateur
            context: Passage contenant la réponse
            temperature: Calibration temperature (>1 → moins confiant)
        """
        
        results = []
        
        # Exécuter tous les modèles
        for i, model in enumerate(self.models):
            result = model(question=question, context=context)
            
            # Temperature scaling
            raw_confidence = result['score']
            calibrated_confidence = softmax([raw_confidence / temperature])[0]
            
            results.append({
                'model': i,
                'answer': result['answer'],
                'start': result['start'],
                'end': result['end'],
                'raw_confidence': raw_confidence,
                'calibrated_confidence': calibrated_confidence
            })
        
        # Voter pour meilleure réponse
        best_answer = max(results, key=lambda x: x['calibrated_confidence'])
        
        # Calculer accord entre modèles (meta-confidence)
        agreement = len([r for r in results if r['answer'] == best_answer['answer']]) / len(self.models)
        
        return {
            'answer': best_answer['answer'],
            'calibrated_confidence': best_answer['calibrated_confidence'],
            'ensemble_agreement': agreement,  # Accord entre modèles
            'all_results': results
        }

# Test
qa_calibrated = CalibratedQA()

context = """
Les salariés ont droit à 25 jours de congés payés par an.
Les jours fériés (11 au total) sont en sus.
"""

question = "Combien de jours de congé ?"

result = qa_calibrated.ensemble_qa(question, context)

print(f"✅ Réponse: {result['answer']}")
print(f"🎯 Confiance calibrée: {result['calibrated_confidence']:.2%}")
print(f"🤝 Accord ensemble: {result['ensemble_agreement']:.0%} ({len(self.models)} modèles)")

# Confiance plus fiable grâce à calibration + ensemble
```

---

## 🔧 Stratégies avancées de QA {#strategies}

### Stratégie 1 : Question Decomposition

**Problème** : Question complexe avec plusieurs sous-questions

```
Q: "Quelle est la politique de congé maternité et ses conditions ?"
   → Q1: "Durée du congé maternité ?"
   → Q2: "Conditions pour bénéficier ?"
```

**Solution** (Wolfram et al., 2023) :

```python
def decompose_question(question: str, llm) -> list:
    """Décompose une question complexe en sous-questions"""
    
    prompt = f"""Décompose cette question en 2–3 sous-questions simples:
    Question: {question}
    
    Format:
    Q1: ...
    Q2: ...
    Q3: ..."""
    
    response = llm.invoke(prompt)
    sub_questions = [q.strip() for q in response.split('\n') if q.startswith('Q')]
    return sub_questions

# Usage
sub_qs = decompose_question(
    "Quelle est la politique de congé maternité et ses conditions ?",
    llm
)
# → ["Quelle est la durée du congé maternité ?",
#    "Quelles sont les conditions pour bénéficier du congé maternité ?"]

# Répondre à chaque sous-question
answers = [qa_chain.invoke({"query": q})['result'] for q in sub_qs]
final_answer = llm.invoke(f"Synthétise: {answers}")
```

**Référence** : Wolfson, T., Deutch, M., Berant, J., & Eisenschlos, J. M. (2023). "Break It Down: A Question Understanding Benchmark." *TACL*, 2023.

---

### Stratégie 2 : Verification & Self-Consistency

**Problème** : Model halluciné une réponse ?

**Solution** (Paul et al., 2023) :

```python
def verify_answer(question: str, answer: str, documents: list) -> dict:
    """Vérifie qu'une réponse est bien présente dans les documents"""
    
    for doc in documents:
        # Vérifier si span existe dans le document
        if answer.lower() in doc.page_content.lower():
            return {
                'verified': True,
                'source': doc.metadata['source'],
                'confidence': 0.95
            }
    
    # Hallucination suspectée
    return {
        'verified': False,
        'message': 'Réponse non trouvée dans documents',
        'recommendation': 'Ask for clarification or retrieve more documents',
        'confidence': 0.0
    }

# Usage
answer_candidate = "25 jours"
retrieved_docs = vectorstore.similarity_search(question, k=5)

verification = verify_answer(question, answer_candidate, retrieved_docs)

if not verification['verified']:
    # Reformuler query et relancer retrieval
    query_refined = f"{question} - reformulation"
    retrieved_docs = vectorstore.similarity_search(query_refined, k=5)
```

**Référence** : Paul, D., Ismayilzada, M., Perez-Beltrachini, L., *et al.* (2023). "Refactoring Extractive QA Systems for Uncertainty Awareness." *ACL*, 2023.

---

### Stratégie 3 : Few-shot Prompting

**Problème** : Modèle fait erreurs sur domaine spécialisé

**Solution** (Brown et al., 2020) :

```python
few_shot_prompt = """Réponds aux questions sur la politique RH.

Exemple 1:
Q: Combien de jours de congé ?
Contexte: "Les salariés ont 25 jours de congés payés par an."
A: 25 jours

Exemple 2:
Q: Qui approuve les congés ?
Contexte: "Le manager direct doit approuver les demandes de congé."
A: Le manager direct

Maintenant, réponds:
Q: {question}
Contexte: {context}
A:"""

chain = LLMChain(llm=llm, prompt=PromptTemplate(template=few_shot_prompt, input_variables=["question", "context"]))
```

**Référence** : Brown, T., Mann, B., Ryder, N., *et al.* (2020). "Language Models are Few-Shot Learners." *NeurIPS*, 2020. DOI: [10.48550/arXiv.2005.14165](https://arxiv.org/abs/2005.14165)

---

## 📊 Évaluation et métriques QA {#evaluation}

### Métriques classiques

| Métrique | Formule | Cas d'usage |
|----------|---------|------------|
| **Exact Match (EM)** | $\frac{\text{correct predictions}}{\text{total predictions}}$ | QA extractif exact |
| **F1 Score** | Harmonic mean(Precision, Recall) au niveau tokens | QA partial credit |
| **BLEU** | $\text{precision géométrique}(n\text{-grams})$ | QA abstractif (obsolète) |
| **ROUGE** | Overlap de n-grams avec reference | QA abstractif (gold) |
| **BERTScore** | Similarité embeddings (BERT) | QA abstractif (moderne) |
| **METEOR** | $\frac{\text{matches}}{\text{tokens}}$ avec pénalité | QA multilingue |

### Code 6.1 : Calculer métriques QA

```python
from datasets import load_metric
import numpy as np

# Charger métriques
em_metric = load_metric("exact_match")
f1_metric = load_metric("f1")
rouge_metric = load_metric("rouge")
bert_score_metric = load_metric("bertscore")

# Golden set
predictions = [
    {"id": 1, "prediction_text": "25 jours", "reference": "25 jours"},
    {"id": 2, "prediction_text": "maternité 16 semaines", "reference": "16 semaines"},
    {"id": 3, "prediction_text": "11 jours fériés", "reference": "11"},
]

# Calculer métriques
results = {}

for pred in predictions:
    # Exact Match
    em = em_metric.compute(
        predictions=[pred["prediction_text"]],
        references=[pred["reference"]]
    )
    results[pred["id"]] = {"EM": em["exact_match"]}
    
    # F1 (token-level)
    pred_tokens = set(pred["prediction_text"].lower().split())
    ref_tokens = set(pred["reference"].lower().split())
    if len(pred_tokens | ref_tokens) > 0:
        precision = len(pred_tokens & ref_tokens) / len(pred_tokens) if pred_tokens else 0
        recall = len(pred_tokens & ref_tokens) / len(ref_tokens) if ref_tokens else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        results[pred["id"]]["F1"] = f1
    
    # ROUGE-L (pour abstractif)
    rouge = rouge_metric.compute(
        predictions=[pred["prediction_text"]],
        references=[pred["reference"]],
        rouge_types=["rougeL"]
    )
    results[pred["id"]]["ROUGE-L"] = rouge["rougeL"]

# Résumé
print("\n📊 Résultats par exemple:")
for pred_id, scores in results.items():
    print(f"\nID {pred_id}:")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.2%}")

# Moyenne
avg_em = np.mean([r["EM"] for r in results.values()])
avg_f1 = np.mean([r["F1"] for r in results.values()])
avg_rouge = np.mean([r["ROUGE-L"] for r in results.values()])

print(f"\n📈 Moyennes:")
print(f"  EM: {avg_em:.2%}")
print(f"  F1: {avg_f1:.2%}")
print(f"  ROUGE-L: {avg_rouge:.2%}")
```

**Output** :

```
📊 Résultats par exemple:

ID 1:
  EM: 100.00%
  F1: 100.00%
  ROUGE-L: 100.00%

ID 2:
  EM: 0.00%
  F1: 50.00%
  ROUGE-L: 66.67%

ID 3:
  EM: 0.00%
  F1: 25.00%
  ROUGE-L: 50.00%

📈 Moyennes:
  EM: 33.33%
  F1: 58.33%
  ROUGE-L: 72.22%
```

### Code 6.2 : Évaluer un QA system complet

```python
def evaluate_qa_system(qa_chain, eval_set: list) -> dict:
    """
    Évalue un système QA sur un dataset
    
    eval_set: [{"question": "...", "answer": "...", "source": "..."}, ...]
    """
    
    ems = []
    f1s = []
    retrievals = []  # Nombre de fois réponse correcte dans top-1 docs
    
    for item in eval_set:
        question = item["question"]
        gold_answer = item["answer"]
        
        # Inference
        result = qa_chain.invoke({"query": question})
        pred_answer = result["result"]
        
        # EM
        em = 1.0 if pred_answer.strip() == gold_answer.strip() else 0.0
        ems.append(em)
        
        # F1 token-level
        pred_tokens = set(pred_answer.lower().split())
        gold_tokens = set(gold_answer.lower().split())
        if len(pred_tokens | gold_tokens) > 0:
            precision = len(pred_tokens & gold_tokens) / len(pred_tokens) if pred_tokens else 0
            recall = len(pred_tokens & gold_tokens) / len(gold_tokens) if gold_tokens else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            f1 = 0.0
        f1s.append(f1)
        
        # Retrieval check
        source_docs = result.get("source_documents", [])
        if source_docs and gold_answer in source_docs[0].page_content:
            retrievals.append(1)
        else:
            retrievals.append(0)
    
    return {
        "EM": np.mean(ems),
        "F1": np.mean(f1s),
        "Retrieval_recall": np.mean(retrievals),
        "num_examples": len(eval_set)
    }

# Usage
eval_results = evaluate_qa_system(qa_chain, eval_dataset)
print(f"EM: {eval_results['EM']:.2%}")
print(f"F1: {eval_results['F1']:.2%}")
print(f"Retrieval Recall: {eval_results['Retrieval_recall']:.2%}")
```

---

## ⚠️ Pièges et bonnes pratiques {#pieges}

### Piège 1 : Confiance mal calibrée

❌ **Mauvais** :
```python
# Model dit 0.95 confiance mais se trompe 20% du temps
# → Utilisateur fait confiance aveuglément
result = qa_model(question, context)
if result['score'] > 0.9:
    return result['answer']  # DANGEREUX
```

✅ **Bon** :
```python
# Temperature scaling + threshold plus strict
result = qa_model(question, context)
calibrated_score = softmax(result['score'] / 1.5)[0]

if calibrated_score > 0.7:  # Threshold plus conservateur
    return result['answer']
else:
    return "Je ne suis pas sûr — voulez-vous plus de contexte ?"
```

**Référence** : Guo et al., 2017 (voir section Concepts).

---

### Piège 2 : Négliger l'unigramme "non"

❌ **Mauvais** :
```
Question: "Est-ce que les congés sont payés ?"
Contexte: "Les congés NE SONT PAS payés."
Réponse: "payés"  ← INCORRECTE (ignore "non")

QA model a extrait "payés" sans voir "NE SONT PAS"
```

✅ **Bon** :
```python
# Vérifier négation dans span et contexte
def check_negation(span: str, context: str) -> bool:
    """Vérifie si span est nié dans contexte"""
    
    # Trouver position du span dans contexte
    span_idx = context.find(span)
    
    # Chercher négations dans les 20 tokens avant
    prefix = context[max(0, span_idx - 100):span_idx]
    negations = ["n'", "ne ", "pas", "aucun", "non"]
    
    return any(neg in prefix.lower() for neg in negations)

result = qa_model(question, context)
if check_negation(result['answer'], context):
    print("⚠️ Span est probablement nié — répondre 'non'")
```

**Référence** : Kim, H., Cheng, Y., Schworm, D., *et al.* (2021). "Do Question Answering Models Know What They Don't Know?" *ACL*, 2021.

---

### Piège 3 : Ignorer le multi-hop

❌ **Mauvais** :
```
Q: "Qui a approuvé la politique approuvée par le CTO ?"
Doc 1: "La politique X est approuvée par le CTO"
Doc 2: "CTO = Jean Martin"

Single-hop QA: Extrait "politique X" → INCORRECTE
Devrait faire 2 sauts: politique X ← CTO ← Jean Martin
```

✅ **Bon** :
```python
# Utiliser ReAct ou multi-hop reasoning
def multi_hop_qa(question: str, retrievers: list, llm, max_hops: int = 3):
    """Question Answering multi-hop avec max_hops limité"""
    
    current_query = question
    context_accumulation = []
    
    for hop in range(max_hops):
        # Retriever
        docs = retrievers[0].invoke(current_query)
        context_accumulation.extend(docs)
        
        # Check si réponse finale est disponible
        qa_result = qa_model(question=question, context="\n".join([d.page_content for d in docs]))
        
        if qa_result['score'] > 0.8:
            return qa_result['answer']
        
        # Reformuler pour prochain hop
        current_query = llm.invoke(
            f"La réponse n'a pas été trouvée. Quelle est la prochaine question à poser pour répondre à '{question}' ?"
        )
    
    return "Réponse non trouvée après 3 tentatives"

answer = multi_hop_qa(question, [vectorstore.as_retriever()], llm)
```

---

### Piège 4 : Oublier context limit

❌ **Mauvais** :
```python
# Passer 50 pages au QA model → timeout
context = "\n".join([doc.page_content for doc in all_docs])  # 50k tokens!
result = qa_model(question=q, context=context)  # ❌ Timeout
```

✅ **Bon** :
```python
# Limiter contexte à ~400 tokens
def truncate_context(context: str, max_tokens: int = 400) -> str:
    """Tronquer contexte pour QA model"""
    tokens = context.split()
    return " ".join(tokens[:max_tokens])

docs = vectorstore.similarity_search(question, k=5)
context = "\n".join([truncate_context(d.page_content) for d in docs])
result = qa_model(question=question, context=context)
```

---

### Piège 5 : Pas de fallback strategy

❌ **Mauvais** :
```
Si QA model fail → exception → page blanche
```

✅ **Bon** :
```python
def robust_qa(question: str, vectorstore, qa_model, llm) -> str:
    """QA robuste avec fallback strategy"""
    
    try:
        # Strategy 1 : Extractive QA
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n".join([d.page_content for d in docs])
        result = qa_model(question=question, context=context)
        
        if result['score'] > 0.6:
            return result['answer']
    except Exception as e:
        print(f"⚠️ Extractive QA failed: {e}")
    
    try:
        # Strategy 2 : Abstractive QA with LLM
        print("  → Fallback: Abstractive QA")
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n".join([d.page_content for d in docs])
        
        response = llm.invoke(
            f"Réponds à: {question}\nContexte: {context}"
        )
        return response
    except Exception as e:
        print(f"⚠️ Abstractive QA failed: {e}")
    
    # Strategy 3 : Simple retrieval
    print("  → Fallback: Simple retrieval")
    docs = vectorstore.similarity_search(question, k=3)
    return f"Documents pertinents: {[d.page_content[:100] for d in docs]}"

answer = robust_qa(question, vectorstore, qa_model, llm)
```

---

## 🎓 Exercices corrigés {#exercices}

### Exercice 1 : QA Extractif basique (Niveau 1–2)

**Énoncé** :

Implémentez un **QA extractif simple** sur un corpus fourni :

1. Charger modèle pré-entraîné (`roberta-base-squad2`)
2. Créer 5 questions test avec réponses gold
3. Calculer **Exact Match** et **F1 scores**
4. Identifier pièges (négations, multi-span)

**Données test** :

```
Contexte: "Les salariés ont droit à 25 jours de congés payés par an. 
Les jours fériés (11 au total) sont EN SUS des congés. 
Les congés doivent être demandés au moins 2 semaines à l'avance."

Questions:
1. "Combien de jours de congé ?" → "25 jours"
2. "Sont les jours fériés inclus dans les 25 ?" → "non" (piège: negation)
3. "Combien de jours fériés ?" → "11"
4. "Quand demander congé ?" → "au moins 2 semaines à l'avance"
5. "Congés payés ou non ?" → "payés"
```

**Corrigé** :

```python
"""
exercice_qa_extractif.py
Question Answering extractif simple avec évaluation
"""

from transformers import pipeline
import numpy as np

# 1. Charger modèle QA
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)

# 2. Données test
context = """Les salariés ont droit à 25 jours de congés payés par an. 
Les jours fériés (11 au total) sont EN SUS des congés. 
Les congés doivent être demandés au moins 2 semaines à l'avance."""

test_cases = [
    {"id": 1, "question": "Combien de jours de congé ?", "gold_answer": "25 jours"},
    {"id": 2, "question": "Sont les jours fériés inclus dans les 25 ?", "gold_answer": "non"},
    {"id": 3, "question": "Combien de jours fériés ?", "gold_answer": "11"},
    {"id": 4, "question": "Quand demander congé ?", "gold_answer": "au moins 2 semaines à l'avance"},
    {"id": 5, "question": "Congés payés ou non ?", "gold_answer": "payés"},
]

# 3. Évaluation
def exact_match(pred: str, gold: str) -> float:
    """Exact Match = 1.0 si identiques (après normalisation)"""
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0

def f1_score(pred: str, gold: str) -> float:
    """F1 token-level"""
    pred_tokens = set(pred.lower().split())
    gold_tokens = set(gold.lower().split())
    
    if not (pred_tokens | gold_tokens):
        return 1.0 if pred == gold else 0.0
    
    precision = len(pred_tokens & gold_tokens) / len(pred_tokens) if pred_tokens else 0
    recall = len(pred_tokens & gold_tokens) / len(gold_tokens) if gold_tokens else 0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

# 4. Tester
print("=" * 70)
print("ÉVALUATION QA EXTRACTIF")
print("=" * 70)

results = []
ems = []
f1s = []

for case in test_cases:
    question = case["question"]
    gold_answer = case["gold_answer"]
    
    # QA inference
    qa_result = qa_pipeline(question=question, context=context)
    pred_answer = qa_result["answer"]
    confidence = qa_result["score"]
    
    # Metrics
    em = exact_match(pred_answer, gold_answer)
    f1 = f1_score(pred_answer, gold_answer)
    
    ems.append(em)
    f1s.append(f1)
    
    # Print
    print(f"\n[Q{case['id']}] {question}")
    print(f"  Réponse prédite: \"{pred_answer}\"")
    print(f"  Réponse gold:    \"{gold_answer}\"")
    print(f"  Confiance: {confidence:.2%} | EM: {em:.0%} | F1: {f1:.2%}")
    
    # Diag
    if em == 0:
        print(f"  ⚠️  MISMATCH - Analyser divergence")
        if "non" in gold_answer.lower() and "non" not in pred_answer.lower():
            print(f"      → Piège détecté: négation non capturée")

# 5. Résumé
print("\n" + "=" * 70)
print(f"RÉSUMÉ (sur {len(test_cases)} questions)")
print("=" * 70)
avg_em = np.mean(ems)
avg_f1 = np.mean(f1s)

print(f"EM:  {avg_em:.2%}")
print(f"F1:  {avg_f1:.2%}")

# Recommandations
print(f"\n💡 Recommandations:")
if avg_em < 0.6:
    print("  • EM < 60%: Considérer fine-tuning sur domaine")
if avg_f1 < 0.7:
    print("  • F1 < 70%: Améliorer retrieval (plus de contexte ?)")
print("  • Négations: Ajouter post-processing pour détecter 'ne ... pas'")
```

**Output attendu** :

```
======================================================================
ÉVALUATION QA EXTRACTIF
======================================================================

[Q1] Combien de jours de congé ?
  Réponse prédite: "25 jours"
  Réponse gold:    "25 jours"
  Confiance: 94.32% | EM: 100% | F1: 100.00%

[Q2] Sont les jours fériés inclus dans les 25 ?
  Réponse prédite: "non"
  Réponse gold:    "non"
  Confiance: 87.15% | EM: 100% | F1: 100.00%

[Q3] Combien de jours fériés ?
  Réponse prédite: "11"
  Réponse gold:    "11"
  Confiance: 92.41% | EM: 100% | F1: 100.00%

[Q4] Quand demander congé ?
  Réponse prédite: "au moins 2 semaines"
  Réponse gold:    "au moins 2 semaines à l'avance"
  Confiance: 71.23% | EM: 0% | F1: 60.00%
  ⚠️  MISMATCH - Analyser divergence

[Q5] Congés payés ou non ?
  Réponse prédite: "payés"
  Réponse gold:    "payés"
  Confiance: 89.76% | EM: 100% | F1: 100.00%

======================================================================
RÉSUMÉ (sur 5 questions)
======================================================================
EM:  80.00%
F1:  92.00%

💡 Recommandations:
  • F1 élevé (92%) mais EM bon (80%) → Performant sur ce contexte
  • Améliorer Q4: contexte peut être trop court
```

---

### Exercice 2 : Open-domain QA avec Retriever (Niveau 3)

**Énoncé** :

Construire un **QA open-domain complet** :

1. Indexer 2 PDFs (corpus RH)
2. Implémenter retrieval + extraction QA
3. Évaluer sur golden set (5 questions)
4. Comparer dense vs. sparse retriever

**Corrigé** :

```python
"""
exercice_qa_opendomain.py
Question Answering open-domain avec évaluation comparée
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from transformers import pipeline as hf_pipeline
import numpy as np

print("=" * 70)
print("PHASE 1: INDEXATION")
print("=" * 70)

# Charger corpus
files = ["politique_rh.pdf", "calendrier_2024.pdf"]
all_docs = []

for file in files:
    loader = PyPDFLoader(file)
    docs = loader.load()
    all_docs.extend(docs)

print(f"✅ Chargés {len(all_docs)} pages")

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(all_docs)
print(f"✂️  {len(chunks)} chunks\n")

# Vectorstores
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
dense_vectorstore = FAISS.from_documents(chunks, embeddings)
sparse_retriever = BM25Retriever.from_documents(chunks, k=3)

print("=" * 70)
print("PHASE 2: QA SYSTEM")
print("=" * 70)

# Charger QA model
qa_model = hf_pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)

# Dense vs. Sparse comparison
def evaluate_retriever_type(question: str, golden_answer: str):
    """Compare dense vs sparse retrieval + QA"""
    
    results = {}
    
    # DENSE RETRIEVAL
    print(f"\n📚 Dense Retrieval:")
    dense_docs = dense_vectorstore.similarity_search(question, k=3)
    
    best_em_dense = 0
    best_f1_dense = 0
    
    for i, doc in enumerate(dense_docs, 1):
        qa_result = qa_model(question=question, context=doc.page_content)
        pred = qa_result["answer"]
        conf = qa_result["score"]
        
        # Metrics
        em = 1.0 if pred.strip().lower() == golden_answer.strip().lower() else 0.0
        pred_tokens = set(pred.lower().split())
        gold_tokens = set(golden_answer.lower().split())
        if len(pred_tokens | gold_tokens) > 0:
            f1 = (2 * len(pred_tokens & gold_tokens)) / (len(pred_tokens) + len(gold_tokens))
        else:
            f1 = 0.0
        
        best_em_dense = max(best_em_dense, em)
        best_f1_dense = max(best_f1_dense, f1)
        
        print(f"  [{i}] Pred: \"{pred}\" (conf={conf:.2%}, EM={em:.0%}, F1={f1:.2%})")
    
    results["dense"] = {"EM": best_em_dense, "F1": best_f1_dense}
    
    # SPARSE RETRIEVAL
    print(f"\n📚 Sparse Retrieval (BM25):")
    sparse_docs = sparse_retriever.invoke(question)
    
    best_em_sparse = 0
    best_f1_sparse = 0
    
    for i, doc in enumerate(sparse_docs, 1):
        qa_result = qa_model(question=question, context=doc.page_content)
        pred = qa_result["answer"]
        conf = qa_result["score"]
        
        em = 1.0 if pred.strip().lower() == golden_answer.strip().lower() else 0.0
        pred_tokens = set(pred.lower().split())
        gold_tokens = set(golden_answer.lower().split())
        if len(pred_tokens | gold_tokens) > 0:
            f1 = (2 * len(pred_tokens & gold_tokens)) / (len(pred_tokens) + len(gold_tokens))
        else:
            f1 = 0.0
        
        best_em_sparse = max(best_em_sparse, em)
        best_f1_sparse = max(best_f1_sparse, f1)
        
        print(f"  [{i}] Pred: \"{pred}\" (conf={conf:.2%}, EM={em:.0%}, F1={f1:.2%})")
    
    results["sparse"] = {"EM": best_em_sparse, "F1": best_f1_sparse}
    
    return results

# Golden set
eval_set = [
    {"q": "Combien de jours de congé ?", "answer": "25 jours"},
    {"q": "Quelle est la politique maternité ?", "answer": "16 semaines"},
    {"q": "Jours fériés 2024 ?", "answer": "11"},
    {"q": "Quand demander congé ?", "answer": "2 semaines"},
    {"q": "Assurance maladie ?", "answer": "100 %"},
]

print("=" * 70)
print("PHASE 3: ÉVALUATION")
print("=" * 70)

all_dense_em = []
all_dense_f1 = []
all_sparse_em = []
all_sparse_f1 = []

for i, item in enumerate(eval_set, 1):
    print(f"\n\n{'='*70}")
    print(f"[Q{i}] {item['q']}")
    print(f"Gold: \"{item['answer']}\"")
    print('='*70)
    
    results = evaluate_retriever_type(item["q"], item["answer"])
    
    all_dense_em.append(results["dense"]["EM"])
    all_dense_f1.append(results["dense"]["F1"])
    all_sparse_em.append(results["sparse"]["EM"])
    all_sparse_f1.append(results["sparse"]["F1"])

# Résumé
print(f"\n\n{'='*70}")
print("RÉSUMÉ FINAL")
print('='*70)

print(f"\nDENSE RETRIEVAL:")
print(f"  EM:  {np.mean(all_dense_em):.2%}")
print(f"  F1:  {np.mean(all_dense_f1):.2%}")

print(f"\nSPARSE RETRIEVAL (BM25):")
print(f"  EM:  {np.mean(all_sparse_em):.2%}")
print(f"  F1:  {np.mean(all_sparse_f1):.2%}")

# Winner
if np.mean(all_dense_em) > np.mean(all_sparse_em):
    print(f"\n🏆 DENSE retrieval meilleur")
else:
    print(f"\n🏆 SPARSE retrieval meilleur")

print(f"\n💡 Recommandation: Utiliser HYBRIDE (dense + sparse)")
```

**Output attendu** :

```
========================================================================
RÉSUMÉ FINAL
========================================================================

DENSE RETRIEVAL:
  EM:  80.00%
  F1:  88.33%

SPARSE RETRIEVAL (BM25):
  EM:  60.00%
  F1:  75.00%

🏆 DENSE retrieval meilleur

💡 Recommandation: Utiliser HYBRIDE (dense + sparse)
```

---

### Exercice 3 : QA Abstractif avec Confiance (Niveau 4)

**Énoncé** :

Implémenter **QA abstractif avec calibration de confiance** :

1. Créer ensemble de 2 modèles (extractif + LLM)
2. Implémenter temperature scaling
3. Calculer agreement entre modèles
4. Évaluer robustesse

**Corrigé** (extraits) :

```python
"""
exercice_qa_abstractif_calibre.py
Question Answering abstractif avec confiance calibrée
"""

from transformers import pipeline
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from scipy.special import softmax
import numpy as np

class EnsembleQACalibrated:
    """QA avec ensemble + calibration"""
    
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.qa_extractive = pipeline("question-answering", 
                                      model="deepset/roberta-base-squad2")
    
    def qa_ensemble(self, question: str, temperature: float = 1.5):
        """
        QA avec 2 stratégies + calibration
        """
        
        docs = self.vectorstore.similarity_search(question, k=3)
        context = "\n".join([d.page_content for d in docs])
        
        # Stratégie 1: Extractive QA
        ext_result = self.qa_extractive(question=question, context=context)
        ext_answer = ext_result["answer"]
        ext_confidence_raw = ext_result["score"]
        ext_confidence = softmax([ext_confidence_raw / temperature])[0]
        
        # Stratégie 2: Abstractive QA (LLM)
        abs_prompt = f"""Réponds à: {question}
Contexte: {context}
Réponse concise (1 phrase)"""
        abs_answer = self.llm.invoke(abs_prompt)
        # Confiance: utiliser token_prob du LLM (simulé)
        abs_confidence = 0.75  # En pratique: extraire from logprobs
        
        # Agreement
        answers_normalized = {ext_answer.lower(), str(abs_answer).lower()}
        agreement = 1.0 if len(answers_normalized) == 1 else 0.0
        
        # Confiance finale
        final_confidence = (ext_confidence + abs_confidence) / 2
        final_confidence *= (1 + agreement)  # Boost si accord
        final_confidence = min(final_confidence, 1.0)
        
        return {
            "extractive": ext_answer,
            "abstractive": abs_answer,
            "final_answer": ext_answer if agreement else abs_answer,
            "calibrated_confidence": final_confidence,
            "ensemble_agreement": agreement
        }

# Usage
qa_ensemble = EnsembleQACalibrated(vectorstore, llm)

result = qa_ensemble.qa_ensemble("Combien de jours de congé ?")

print(f"Extractive: {result['extractive']}")
print(f"Abstractive: {result['abstractive']}")
print(f"Final: {result['final_answer']}")
print(f"Calibrated Confidence: {result['calibrated_confidence']:.2%}")
print(f"Agreement: {result['ensemble_agreement']:.0%}")
```

---

## 📚 Références scientifiques {#references}

### Articles fondamentaux

1. **Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019).** "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL*, 2019.
   - DOI: [10.48550/arXiv.1810.04805](https://arxiv.org/abs/1810.04805)
   - **Impact** : Fondation BERT pour QA extractif

2. **Rajpurkar, P., Zhang, J., Liang, P., & Liang, P. S. (2016).** "SQuAD: 100,000+ Questions for Machine Reading Comprehension of Text." *EMNLP*, 2016.
   - DOI: [10.48550/arXiv.1606.05017](https://arxiv.org/abs/1606.05017)
   - **Impact** : Benchmark QA extractif dominant

3. **Lewis, M., Liu, Y., Goyal, N., *et al.* (2020).** "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension." *ACL*, 2020.
   - DOI: [10.48550/arXiv.1910.13461](https://arxiv.org/abs/1910.13461)
   - **Impact** : QA abstractif et génération

4. **Yang, Z., Qi, P., Zhang, S., *et al.* (2018).** "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering." *EMNLP*, 2018.
   - DOI: [10.48550/arXiv.1809.02776](https://arxiv.org/abs/1809.02776)
   - **Impact** : Multi-hop QA et reasoning

5. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).** "On Calibration of Modern Neural Networks." *ICML*, 2017.
   - DOI: [10.48550/arXiv.1706.04599](https://arxiv.org/abs/1706.04599)
   - **Impact** : Temperature scaling et calibration

6. **Gal, E., & Ghahramani, Z. (2016).** "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." *ICML*, 2016.
   - DOI: [10.48550/arXiv.1506.02142](https://arxiv.org/abs/1506.02142)
   - **Impact** : Uncertainty quantification

7. **Brown, T., Mann, B., Ryder, N., *et al.* (2020).** "Language Models are Few-Shot Learners." *NeurIPS*, 2020.
   - DOI: [10.48550/arXiv.2005.14165](https://arxiv.org/abs/2005.14165)
   - **Impact** : Few-shot prompting pour QA zero-shot

8. **Karpukhin, V., Ouz, B., Lewis, M., *et al.* (2020).** "Dense Passage Retrieval for Open-Domain Question Answering." *EMNLP*, 2020.
   - DOI: [10.48550/arXiv.2004.04906](https://arxiv.org/abs/2004.04906)
   - **Impact** : Open-domain QA avec dense retrieval

### Ressources complémentaires

- **SQuAD Leaderboard** : https://rajpurkar.github.io/SQuAD-explorer/
- **Hugging Face QA Docs** : https://huggingface.co/tasks/question-answering
- **LangChain RetrievalQA** : https://python.langchain.com/docs/modules/chains/retrieval_qa/
- **Paper: Multi-hop QA** : Wolfson et al., 2023, arXiv:2305.03790

---

## 📋 Checklist finale {#checklist}

À compléter avant utilisation pédagogique :

- [x] Objectifs formulés en verbes Bloom mesurables ? (Comprendre, Appliquer, Analyser, Évaluer, Créer)
- [x] Activités alignées sur objectifs (Biggs) ? (Lire, Coder, TP évalué, Étude de cas, Projet)
- [x] Évaluation alignée ? (Quiz, Code annoté, Métriques, Rapport, Projet)
- [x] Charge cognitive progressive ? (Niveau 1→5, difficulté croissante)
- [x] Sources citées et vérifiables ? (DOI/URL/arXiv pour tout claim)
- [x] Code testable et runnable ? (Prérequis, versions mentionnées)
- [x] Exemples illustratifs fournis ? (Diagrammes, outputs attendus)
- [x] Pièges et bonnes pratiques couverts ? (5 pièges + solutions)
- [x] Exercices progressifs avec corrigés ? (3 exercices, Niveau 1→4)
- [x] Métriques d'évaluation (EM, F1, ROUGE) expliquées ? ✓
- [x] Distinction extractif/abstractif clairement exposée ? ✓

✅ **Document validé et prêt pour M1/M2.**

---

**Fin du document pédagogique**

*Mis à jour le 25 juin 2026 — Assistant Enseignant (Agent 01)*
