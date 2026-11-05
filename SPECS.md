# SPECS — `ner-mcp-fr` : serveur MCP de NER français

> Statut : **spécification validée (v1.0)** — arbitrages tranchés en §13.2 le 2026-11-05.
> Auteur : OpenCode / DeepSeek V4.1 Flash — 2026-11-05.

---

## 1. Contexte et intention

On veut exposer un **reconnaissance d'entités nommées (NER) en français** aux agents LLM via
le **Model Context Protocol (MCP)**. L'objectif n'est pas de réentraîner un modèle : un modèle
open-source de l'état de l'art francophone existe déjà et fait le travail. L'objectif est de le
rendre **utilisable proprement** : contrat d'outil stable, offsets exploitables, textes longs
gérés, démarrage rapide, exécution locale.

Point de départ fourni par l'utilisateur :

```python
from transformers import pipeline

ner = pipeline(
    'token-classification',
    model='CATIE-AQ/NERmembert-large-4entities',
    tokenizer='CATIE-AQ/NERmembert-large-4entities',
    aggregation_strategy="simple",
)
results = ner("Le dévoilement du logo officiel des JO ...")
```

Ce snippet est notre **référence de comportement**, pas notre implémentation cible :
`pipeline` tronque silencieusement au-delà de 512 tokens, ne renvoie pas d'information de
troncature, et l'agrégation `simple` produit des offsets qu'il faut revérifier. Le serveur MCP
doit corriger ces trois points.

## 2. Faits vérifiés sur les modèles (Hugging Face + essais réels, 2026-11-05)

| Propriété | `CATIE-AQ/NERmembert-large-4entities` | `CATIE-AQ/NERmembert-base-4entities` | `CATIE-AQ/NERmemberta-4entities` |
|---|---|---|---|
| Architecture | CamemBERT-large (24 couches) | CamemBERT-base (12 couches) | CamemBERTa v2 base (DeBERTa-v2) |
| Paramètres | 336 M | 110 M | 111 M |
| Poids | 1.34 GB (fp32) | 420 MB (fp32) | 422 MB (fp32) |
| Contexte max | 512 tokens | 512 tokens | 1024 tokens (annoncé) |
| Tokenizer | SentencePiece (correct) | SentencePiece (correct) | byte-level **sans règles de fusion** |
| F1 PER / LOC / ORG / MISC | **0.982 / 0.964 / 0.919 / 0.834** | 0.978 / 0.958 / 0.903 / 0.814 | 0.979 / 0.961 / 0.915 / 0.812 |
| Verdict | utilisable | utilisable | **inutilisable en l'état** |

Trois constats qui ont changé la conception :

1. **La famille CamemBERTa est cassée dans ce dépôt.** `NERmemberta-*`, `NERmembert2-*` et
   `almanach/camembertav2-base` publient un vocabulaire byte-level sans fichier `merges.txt` :
   le tokenizer éclate chaque mot en caractères (`Ġ S y l v a i n`). Les prédictions deviennent
   du bruit (`Boyer` → `MISC`, aucun `PER`, `Art` → `MISC`). Vérifié à la main avec
   `AutoTokenizer` (fast et slow) : même résultat. **Le modèle marqué `new_version` par les
   auteurs est donc inutilisable, et le profil `fast` a été redéfini sur `NERmembert-base`.**
2. Le contexte est de **512 tokens** dans les deux profils retenus, pas 512 mots : ~350 mots de
   français. Un article de presse dépasse cela systématiquement. Le découpage en fenêtres n'est
   pas une option, et l'avantage « 1024 tokens » du profil `fast` initialement prévu a disparu
   avec le tokenizer cassé.
3. **La qualité dépend du texte, pas seulement du F1.** Sur le paragraphe JO de référence,
   `large` et `base` donnent presque le même résultat (10 entités, mêmes frontières), avec un
   score légèrement meilleur pour `base` sur `Paris`. Le F1 publié n'est pas un discriminant
   suffisant pour choisir : le profil par défaut privilégie la mémoire.

Licence MIT pour les trois : usage commercial et redistribution autorisés, attribution requise
dans le README et dans la réponse de l'outil `ner_info`.

## 3. Objectifs

- **O1** — Un serveur MCP local, transport **stdio**, installable en une commande, sans
  dépendance à un service tiers.
- **O2** — Extraire `PER`, `LOC`, `ORG`, `MISC` avec **offsets caractère vérifiables** dans le
  texte d'entrée tel qu'il a été fourni (`text[start:end] == entity.text`).
- **O3** — Gérer des textes **plus longs que le contexte du modèle** sans troncature silencieuse,
  avec un champ explicite signalant les limites atteintes.
- **O4** — Réponses **structurées** (JSON Schema déclaré) : un agent doit pouvoir les traiter
  sans parsing fragile.
- **O5** — Démarrage à froid maîtrisé : modèle chargé une fois, réutilisé, temps de chargement
  mesuré et exposé.
- **O6** — Code testé, typé, linté ; tout développeur peut comprendre et reprendre le dépôt.
- **O7** — Un **CLI de vérification** (`ner-mcp-fr extract`) qui analyse un fichier ou STDIN et
  rend l'annotation lisible (scores, offsets, texte coloré), pour auditer le modèle sans agent.

## 4. Non-objectifs (v1)

- ❌ Entraînement / fine-tuning / évaluation de modèles.
- ❌ Multilingue (le serveur est **mono-modèle français** ; comportement hors français documenté
  comme dégradé, pas comme supporté).
- ❌ Entity linking vers Wikidata, résolution de coréférence, normalisation de dates.
- ❌ Serveur hébergé / multi-tenant / authentification / quotas.
- ❌ Transport HTTP (Streamable HTTP) — tranché en §13.2.
- ❌ CLI interactive (REPL, TUI). Le CLI est non-interactif, scriptable : `extract` et `info`.
  Pas de prompt, pas de menu, pas de `curses`.
- ❌ Interface web, CLI interactive, dashboard.

## 5. Stack technique

| Choix | Justification |
|---|---|
| **Python ≥ 3.12** (3.12 recommandé, 3.13/3.14 fonctionnels) | `onnxruntime` publie des wheels 3.11 → 3.14 : plus de contrainte liée à torch. |
| **`mcp` SDK v2.x** (`from mcp.server import MCPServer`) | SDK officiel. ⚠️ Le nom `FastMCP` de la v1 a été remplacé par `MCPServer` en v2 ; toute doc/tutoriel v1 est à ignorer. |
| **`onnxruntime`** | Exécute le graphe. 75 MB installé contre 581 MB pour torch (macOS arm64). |
| **`tokenizers`** | Tokenisation seule, 9,5 MB : ids, offsets et `word_ids` vérifiés identiques à `AutoTokenizer`, donc `transformers` (56 MB) devient inutile au runtime. |
| **`huggingface-hub`** | Téléchargement et cache des poids, mode hors-ligne. |
| **`numpy`** | Batching, softmax, argmax. |
| **`torch` + `transformers`** | **groupe `export` uniquement**, pour produire le `.onnx`. Jamais installés par `uvx`. |
| **`uv`** | Gestion d'environnement et `uvx` pour l'exécution sans installation. |
| **`pydantic`** (via le SDK) | Schémas d'entrée/sortie des outils. |
| **`argparse`** (stdlib) | CLI sans dépendance supplémentaire : `typer`/`click` n'apportent rien pour 3 sous-commandes. |
| **`pytest` + `ruff`** | Tests et lint. |

**Pourquoi ONNX.** Avec torch, `uvx ner-mcp-fr` installait 778 MB, dont 581 MB de torch : un
outil qui analyse des phrases coûtait le prix d'un framework d'entraînement. ONNX ramène
l'installation à **149 MB** mesurés, sans changer le modèle ni une seule entité produite.

Les mesures qui ont décidé du reste (§18) : fp32 ONNX est identique au torch au bit près ; fp16
donne les mêmes entités pour moitié de téléchargement ; int8 détruit le modèle. D'où **fp16 par
défaut, int8 interdit**.

**La précision n'est pas un réglage utilisateur.** Elle est portée par l'artefact publié et
seulement *rapportée* par `ner_info`. Une variable d'environnement qui changerait les poids
reviendrait à laisser un typo modifier les prédictions.

## 6. Architecture

```
ner-mcp-fr/
├── SPECS.md
├── README.md                  # install, config hôte, exemples, attribution modèle
├── pyproject.toml             # project + extras + script ner-mcp-fr
├── src/ner_mcp/
│   ├── __init__.py            # __version__
│   ├── config.py              # Settings (env vars), constantes des artefacts
│   ├── schemas.py             # Pydantic: Entity, ExtractionResult, BatchResult, ModelInfo
│   ├── chunking.py            # découpage phrase-aware + remapping d'offsets
│   ├── artifacts.py           # téléchargement, cache, vérification SHA-256
│   ├── backend.py             # session onnxruntime + tokenizer, batching
│   ├── engine.py              # chunking, agrégation, filtrage (sans MCP)
│   ├── files.py               # lecture fichier/STDIN partagée CLI + outil MCP
│   ├── render.py              # formats de sortie (json, text, highlight, markdown)
│   ├── server.py              # MCPServer, définition des tools, main()
│   └── cli.py                 # argparse: mcp / extract / info
├── tests/
│   ├── conftest.py            # résultat figé, phrase de référence
│   ├── test_chunking.py
│   ├── test_aggregation.py    # agrégation pure, invariant d'offset
│   ├── test_files.py          # lecture fichier, encodage, allowlist
│   ├── test_render.py         # formats json/text/highlight/markdown sur résultat figé
│   └── test_cli.py            # parsing, codes de sortie, fichier vs STDIN
├── scripts/export_onnx.py     # export fp16 reproductible (groupe `export`, torch)
└── .gitignore
```

Séparation stricte : `backend.py` **ne connaît pas MCP**, `engine.py` **ne connaît ni MCP ni
onnxruntime**, `server.py` **ne connaît ni torch ni ONNX**.
Cela rend le moteur testable sans protocole et réutilisable (CLI, HTTP plus tard).

### Cycle de vie du modèle

- Chargement **paresseux** au premier appel d'outil, protégé par un verrou (le SDK peut
  recevoir des requêtes concurrentes).
- Session `onnxruntime` en `ORT_ENABLE_ALL`, sans état d'entraînement.
- Chargement **optionnellement anticipé** (`NER_PRELOAD=1`) pour ne pas payer le coût au premier
  appel.
- Un échec de téléchargement (réseau, cache, token) remonte une erreur MCP explicite avec la
  cause et la remédiation, jamais une trace Python brute.

## 7. Poids du modèle

Un seul modèle, servi par un **graphe ONNX fp16** hébergé à part des poids PyTorch.

| Élément | Valeur |
|---|---|
| Source (tokenizer + `id2label`) | `CATIE-AQ/NERmembert-base-4entities` |
| Poids servis | `athoune/ner-mcp-fr-onnx` → `model-fp16.onnx` (210 MB) |
| Contexte | 512 tokens |
| SHA-256 épinglé | oui, dans `config.py` (`DEFAULT_ARTIFACT_SHA256`) |
| Licence | MIT (export d'un modèle MIT) |

`onnxruntime` ne sait pas lire un checkpoint PyTorch, d'où ce second dépôt : un seul fichier,
produit par `scripts/export_onnx.py`. Le tokenizer et la table de labels restent chez CATIE-AQ —
inutile de recopier des fichiers déjà fiables.

Le profil `large` (336 M) **disparaît** : son artefact fp16 pèserait 670 MB, pour un gain de F1
non démontré sur du texte réel (`base` et `large` donnent les mêmes 10 entités sur le texte de
référence, avec des scores quasi identiques). Le script sait l'exporter ; il n'est simplement pas
publié.

⚠️ **Ne pas utiliser `CATIE-AQ/NERmemberta-*` ni `NERmembert2-*`** : leur tokenizer byte-level
est publié sans règles de fusion, ce qui détruit les prédictions (§2). Deux tests verrouillent
`SOURCE_MODEL`.

## 8. Contrat des outils MCP

Quatre outils, pas plus. La logique métier (déduplication, résumé, comptage) reste aux agents.

Le point coûteux est le **transport du texte** : un agent qui colle un article de 200 kB dans
`ner_extract` paie ce texte deux fois (une fois pour le lire, une fois dans l'historique). D'où
`ner_extract_file`, qui prend un **chemin** et lit le fichier côté serveur : le contenu ne
traverse jamais le contexte du modèle. C'est le mode à privilégier dès qu'un texte vient d'un
fichier.

### 8.1 `ner_extract` — extraction principale

**Entrée**

| Param | Type | Défaut | Contrainte |
|---|---|---|---|
| `text` | string | — | requis, non vide, ≤ `NER_MAX_INPUT_CHARS` (défaut 200 000) |
| `labels` | string[] | `["PER","LOC","ORG","MISC"]` | sous-ensemble de ces 4 valeurs |
| `min_score` | float | `0.0` | `0.0 ≤ s ≤ 1.0` |
| `merge_adjacent` | bool | `true` | fusionne deux entités contiguës de même label séparées par un seul espace |
| `trim_punctuation` | bool | `true` | retire la ponctuation de phrase collée au span (`Grand Rex.` → `Grand Rex`) |

**Sortie** (voir `ExtractionResult` §9)

### 8.2 `ner_extract_file` — fichier lu côté serveur

**Entrée**

| Param | Type | Défaut | Contrainte |
|---|---|---|---|
| `path` | string | — | requis ; relatif au cwd du serveur ou absolu |
| `labels`, `min_score`, `merge_adjacent` | | | idem `ner_extract` |
| `encoding` | string | `utf-8` (BOM détecté) | encodage du fichier |

**Sortie** : `ExtractionResult` avec `source` = chemin absolu résolu.

Règles :

- Le fichier est lu en **octets** puis décodé ; aucun remplacement silencieux de caractère.
- Taille limitée par `NER_MAX_INPUT_CHARS` **avant** lecture (borne en octets = 4 × la limite).
- `NER_ALLOWED_DIRS` (liste séparée par `os.pathsep`) restreint les chemins lisibles. Vide par
  défaut : le serveur tourne avec les droits de l'utilisateur, dans le même domaine de confiance
  que le shell de l'agent. À renseigner dès que le serveur ingère du contenu non fiable
  (protection contre l'injection de prompt qui ferait lire `~/.ssh` ou `/etc/passwd`).
- Une erreur de lecture remonte un message explicite, jamais une trace.

### 8.3 `ner_batch` — lot

**Entrée** : `texts: string[]` (1..64), mêmes filtres que `ner_extract`.
**Sortie** : `{ "results": [{"index": 0, "text_length": …, "entities": [...]}, …], "model": …, "elapsed_ms": … }`

Traitement en **un seul batch ONNX** (padding dynamique) pour exploiter le matériel. Un texte
invalide dans le lot ne fait pas échouer les autres : il renvoie une entrée avec `error`.

### 8.4 `ner_info` — introspection

**Entrée** : aucune.
**Sortie** : model ID, précision, provider, labels, `max_tokens`, stratégie de
chunking, statut de chargement, version du paquet, licence et attribution, et les compteurs de
la dernière requête (`elapsed_ms`, `chunks`, `truncated`).

Justification : sans cet outil, un agent qui obtient un résultat bizarre (modèle non chargé,
device CPU au lieu de MPS, texte tronqué) n'a aucun moyen de le diagnostiquer sans accès shell.

## 9. Schémas de réponse

```jsonc
{
  "model": "CATIE-AQ/NERmembert-base-4entities",
  "precision": "fp16",
  "device": "CPUExecutionProvider",
  "source": "/Users/moi/article.txt",
  "entities": [
    {
      "text": "Paris",
      "label": "LOC",
      "start": 334,
      "end": 339,
      "score": 0.9991
    }
  ],
  "text_length": 617,
  "chunks": 1,
  "truncated": false,
  "dropped_by_min_score": 0,
  "elapsed_ms": 214
}
```

Règles :

- **Invariant d'offset** : `text[start:end] == entity.text` **toujours**, vérifié par test sur
  chaque entité produite. C'est le contrat le plus important du projet.
- Les offsets sont des **index de code points Unicode** (sémantique Python), pas des unités
  UTF-16. Pour du français, les deux coïncident ; un client JavaScript qui manipule des offsets
  doit utiliser le champ `text` en cas de doute (emoji, caractères hors BMP). Documenté dans le
  README et dans la description de l'outil.
- `source` est le chemin absolu du fichier pour `ner_extract_file`, `null` pour du texte inline
  ou STDIN. Il permet de tracer l'origine d'une analyse sans renvoyer le texte.
- `score` est la probabilité softmax moyenne des sous-tokens de l'entité (comportement
  `aggregation_strategy="simple"` de HF, réimplémenté pour maîtriser les offsets). Quand deux
  mots adjacents de même label sont fusionnés, l'entité conserve le **meilleur** score des deux :
  moyenner deux scores de mots de longueurs différentes sous-estimerait la confiance.
- `truncated` ne passe à `true` que si une limite dure a été atteinte (ex. `text` dépassant
  `NER_MAX_INPUT_CHARS`, fenêtre finale irréductible). Un texte découpé proprement n'est
  **pas** tronqué.
- **Bords de span** : le modèle avale la ponctuation de phrase et le balisage Markdown
  (`Grand Rex.`, `Jeux paralympiques.`, `Ebenezer**`, `Saint-Asaph)**`, `| Paris |`). Par défaut,
  ces caractères de bord sont retirés *après* fusion, sans toucher au reste ni au score ;
  `trim_punctuation=false` rend les spans bruts. La règle est fondée sur la catégorie Unicode
  `P*` plus `|` et `~`, pas sur une liste de caractères écrite à la main : elle couvre `*`, `#`,
  `_`, `«»`, `()`, `…` sans nouveau bug au prochain document. Un span entièrement décoratif
  (`**`) disparaît au lieu d'être émis. Le retrait a lieu après la fusion des mots, sinon un
  point collé casserait la fusion des entités adjacentes.
- `chunks` = nombre de fenêtres d'inférence réellement exécutées (1 pour un texte court).
- Ordre des entités : par `start` croissant, puis `end` croissant. Déterministe.

## 10. Algorithme

### 10.1 Découpage (chunking)

1. Tokeniser avec le tokenizer **fast** pour connaître le nombre de sous-tokens réels
   (jamais une heuristique « caractères / 4 »).
2. Si `n_tokens ≤ max_tokens - 2` → une seule fenêtre, `chunks = 1`.
3. Sinon : découper aux frontières de phrases (`[.!?…]` suivi d'espace ou fin, doubles sauts de
   ligne), accumulation gloutonne jusqu'à `max_tokens - 32` (marge de sécurité), avec
   **chevauchement d'une phrase** entre fenêtres pour éviter de couper une entité à la jonction.
4. Si une phrase unique dépasse la fenêtre : découpage sur les espaces (jamais au milieu d'un
   mot), puis sur les caractères en dernier recours.
5. Chaque fenêtre conserve son **offset absolu** dans le texte d'origine ; les offsets renvoyés
   sont remappés vers le texte complet à la fin.
6. **Déduplication** des entités issues de fenêtres chevauchantes par clé `(label, start, end)`,
   en gardant le score maximal.

### 10.2 Inférence et agrégation

1. Batch de fenêtres avec padding dynamique.
2. Logits → softmax → argmax par sous-token.
3. Utiliser `word_ids` pour regrouper les sous-tokens d'un même mot ; label du mot = label du
   premier sous-token (comportement CamemBERT/SentencePiece), score du mot = moyenne des
   probabilités de ses sous-tokens.
4. Fusionner les mots contigus de même label en une entité ; `text` = `input_text[start:end]`,
   espaces internes inclus.
5. Les offsets proviennent de `offset_mapping` du tokenizer fast ; le préfixe `▁` de
   SentencePiece et les espaces de début sont **retirés en avançant `start`**, jamais en
   réécrivant le texte.
6. Filtrer par `labels`, puis par `min_score` (compter `dropped_by_min_score`).

### 10.3 Matériel

- `NER_DEVICE=auto` : `CUDAExecutionProvider` si `onnxruntime-gpu` est installé, sinon CPU.
- **CoreML est écarté après mesure** : 29 nœuds supportés sur 597, 171 ms contre 74 ms sur le
  texte de référence, à cause des frontières de partition. Le CPU gagne.
- `NER_DEVICE=mps` est **rejeté avec un message explicite** : il n'existe pas de provider Metal,
  et laisser croire le contraire ferait perdre une heure à quelqu'un.
- Pas de gradient, pas de mode entraînement : une session `InferenceSession` avec
  `ORT_ENABLE_ALL`. `NER_THREADS` règle `intra_op_num_threads`.
- Chargement mesuré : 1,5 s à chaud (session + tokenizer), 71 ms par paragraphe de 678
  caractères, 1,2 s pour 10 kB (5 fenêtres, 154 entités).

## 11. Configuration (variables d'environnement)

| Variable | Défaut | Effet |
|---|---|---|
| `NER_MODEL_ID` | `CATIE-AQ/NERmembert-base-4entities` | source du tokenizer et de `id2label` |
| `NER_ARTIFACT_REPO` | `athoune/ner-mcp-fr-onnx` | dépôt Hugging Face des poids ONNX |
| `NER_ARTIFACT_FILE` | `model-fp16.onnx` | fichier de poids |
| `NER_ARTIFACT_SHA256` | (épinglé) | vérification d'intégrité, échec explicite si divergence |
| `NER_ONNX_PATH` | — | fichier local, court-circuite le téléchargement |
| `NER_DEVICE` | `auto` | `auto`/`cpu`/`cuda` (provider onnxruntime) |
| `NER_MAX_INPUT_CHARS` | `200000` | garde-fou anti-DoS mémoire, aussi borne des fichiers |
| `NER_BATCH_SIZE` | `8` | fenêtres par passe (`ner_batch`) |
| `NER_MIN_SCORE` | `0.0` | filtre global par défaut |
| `NER_CACHE_DIR` | (HF défaut) | cache des poids |
| `NER_PRELOAD` | `0` | chargement au démarrage |
| `NER_OFFLINE` | `0` | `1` = pas de réseau, échec clair si modèle absent |
| `NER_LOG_LEVEL` | `INFO` | logs **sur stderr** (stdout = le protocole) |
| `NER_ALLOWED_DIRS` | (vide) | répertoires autorisés pour `ner_extract_file`, séparés par `os.pathsep` |

La précision des poids n'est **pas** configurable (voir §5 et §10.3) : elle vient du checkpoint.

Configuration invalide → échec au démarrage avec message actionnable (fail fast), pas de
retombée silencieuse sur un défaut.

## 12. Interface en ligne de commande (CLI)

**Motivation.** Un serveur MCP est invisible : quand l'analyse sort une entité bizarre, on n'a
aucun moyen de voir *pourquoi* sans instrumenter l'hôte. Le CLI est l'outil de vérification
directe : un fichier ou STDIN entre, les entités sortent, avec le texte annoté et les scores.
Même moteur que le serveur (`engine.py`), zéro duplication de logique — sinon les deux surfaces
divergent et on ne vérifie plus ce que voit l'agent.

### 12.1 Surface

Un seul binaire, `ner-mcp-fr`, trois sous-commandes :

```bash
ner-mcp-fr                          # aucun argument → serveur MCP stdio (compat hôte)
ner-mcp-fr mcp                      # idem, explicite
ner-mcp-fr extract article.txt      # analyse d'un fichier
cat article.txt | ner-mcp-fr extract        # analyse de STDIN
pbpaste | ner-mcp-fr extract --format highlight --stats
ner-mcp-fr info                     # modèle, device, limites, statut de chargement
```

La sous-commande s'appelle **`mcp`**, pas `serve` : « serve » suggère un serveur HTTP avec une
page web, ce que ce projet n'est pas. Aucun argument = `mcp`, pour que la config d'un hôte
(`command: ner-mcp-fr`) ne dépende pas d'un mot supplémentaire.

### 12.2 `extract`

| Option | Défaut | Effet |
|---|---|---|
| `PATH` (positionnel) | `-` (STDIN) | fichier à analyser ; `-` lit STDIN |
| `--format` | `text` si TTY, `json` sinon | `json`, `jsonl`, `text`, `highlight`, `markdown` |
| `--labels` | toutes | `PER,LOC,ORG,MISC` (liste séparée par virgules) |
| `--min-score` | config | filtre de confiance |
| `--no-merge` | fusion activée | désactive `merge_adjacent` |
| `--keep-punctuation` | nettoyage activé | conserve les spans bruts du modèle |
| `--model` | config | force le modèle source (tokenizer + labels) |
| `--onnx` | config | poids ONNX locaux, sans téléchargement |
| `--device` | config | `auto`/`cpu`/`cuda` |
| `--color` / `--no-color` | auto (TTY) | force ou coupe l'ANSI |
| `--encoding` | `utf-8` (BOM détecté) | encodage du fichier/STDIN |
| `--no-legend` | — | masque la légende des couleurs |
| `--stats` | — | écrit chunks / elapsed_ms / filtrés sur **stderr** |
| `--fail-on-empty` | — | sortie 3 si aucune entité (scripting/CI) |

**Formats de sortie**

- `json` — le schéma §9 complet, indenté, avec `source` quand l'entrée est un fichier.
  Format de référence, stable, testé.
- `jsonl` — une entité JSON par ligne (streaming, `jq`-friendly) ; les métadonnées vont sur
  stderr avec `--stats`.
- `text` — tableau aligné : `LABEL  0.9991  [334,339)  Paris`. Lisible, sans couleur.
- `highlight` — **le mode « vérifier »** : le texte original avec les entités colorées par label
  (PER jaune, LOC vert, ORG bleu, MISC magenta) et une légende. On juge les frontières à l'œil.
- `markdown` — tableau + texte annoté `**Paris** (LOC)`, pour coller un rapport dans un ticket.

Le défaut `text` sous TTY / `json` en pipe est le **seul comportement contextuel autorisé**, et
il est documenté : un humain voit un tableau, un script reçoit du JSON sans le demander.
`--format` explicite gagne toujours.

**Entrée**

- Lecture en **octets** puis décodage (BOM `utf-8-sig` géré) ; erreur de décodage → code 1 avec
  message explicite, jamais de remplacement silencieux des caractères.
- Le texte n'est **ni stripé ni normalisé** : les offsets renvoyés valident contre le fichier
  d'origine, octet pour caractère. Le newline final éventuel fait partie de l'entrée.
- Aucune entrée (STDIN vide, fichier vide) → code 1, message clair (ce n'est pas une extraction
  « 0 entité », c'est une erreur d'usage).

**Codes de sortie** : `0` succès (y compris 0 entité) ; `1` erreur d'exécution (fichier absent,
décodage, modèle) ; `2` erreur d'usage (argparse) ; `3` `--fail-on-empty` sans entité.

### 12.3 `info`

Équivalent terminal de `ner_info` : model ID, précision, provider, `max_tokens`, budget de
chunking, labels, version du paquet, licence et attribution, statut de chargement, chemin du
cache. Ne charge **pas** le modèle.

### 12.4 Contraintes

- **stdout = données, stderr = logs et progression.** Même règle que le transport MCP : un
  `print()` de progression dans stdout corromprait un pipe.
- Le CLI est testé sans modèle : les fonctions de rendu prennent un `ExtractionResult` en
  entrée, donc un résultat figé suffit à tester chaque format.
- `ner-mcp-fr extract` sur le même contenu via fichier ou via STDIN doit produire un JSON
  **identique** (test de non-régression sur le mode de lecture).

## 13. Décisions et questions ouvertes

### 13.1 Décisions déjà prises
- Python + `MCPServer` (SDK v2), transport **stdio uniquement** : c'est le mode d'intégration
  attendu par les hôtes MCP locaux, et il ne demande aucune infrastructure. Pas de HTTP en v1 :
  pas de session, pas d'auth, pas de port à défendre.
- Profil **`fast` par défaut** = `NERmembert-base` (420 MB, 512 tokens), après invalidation de
  `NERmemberta` pour tokenizer cassé (§2) ; profil `large` disponible pour reproduire exactement
  le snippet et maximiser le F1.
- Distribution **locale via `uvx --from git+file://…`**, pas de publication PyPI en v1.
- Réimplémentation de l'agrégation plutôt que `pipeline(aggregation_strategy="simple")` :
  on ne peut pas garantir les offsets ni le chunking avec la pipeline telle quelle.
- Aucun outil « résumé » / « comptage » : les agents savent le faire.
- **`ner_extract_file`** existe parce que le coût dominant d'un outil MCP est le texte qui
  traverse le contexte : passer un chemin au lieu du contenu est un gain direct, pas une
  commodité.
- **La précision n'est pas configurable** : elle appartient à l'artefact publié (§5).
- La sous-commande du serveur s'appelle **`mcp`**, pas `serve` : pas de fausse promesse de
  serveur HTTP.
- **La ponctuation de bord est retirée** des entités par défaut (`trim_punctuation`). Le modèle
  produit des spans corrects mais absorbe le point final ; laisser `Grand Rex.` pollue toute
  déduplication en aval. L'option inverse existe pour l'audit du modèle brut.
- Entrée non française : **pas de détection de langue en v1** (ce serait une dépendance
  supplémentaire pour un service qui s'annonce déjà comme français). Le README documente le
  comportement dégradé. À rouvrir si l'usage réel le justifie.

### 13.2 Arbitrages tranchés (2026-11-05)
1. **Profil par défaut** → `fast` (NERmemberta-4entities). `large` en opt-in.
2. **Transport** → stdio uniquement.
3. **Distribution** → `uvx` depuis le dépôt git local ; PyPI hors périmètre v1.
4. **Langue** → pas de détection, documentation des limites.

## 14. Tests et critères d'acceptation

### Tests obligatoires
- **Golden test** sur le paragraphe JO du snippet : `Paris` → `LOC`, `Grand Rex` → `ORG` ou
  `LOC` (tolérance documentée), `Sylvain Boyer` → `PER`, `Royalties`/`Ecobranding` → `ORG`,
  `21 octobre 2019` → `MISC`, `1924` → `MISC`. Le test vérifie la présence des entités clés,
  pas l'exactitude parfaite (le modèle n'est pas déterministe parfait sur les frontières).
- **Invariant d'offset** sur 100 % des entités de tous les tests.
- `labels` filtre correctement ; un label inconnu est rejeté par le schéma.
- `min_score` filtre et compte correctement.
- Texte > `max_tokens` du modèle (1024 en profil `fast`, 512 en `large`) : `chunks > 1`,
  `truncated == false`, entité à cheval sur deux fenêtres détectée une seule fois.
- **Déterminisme** : deux appels identiques donnent exactement le même JSON.
- Cas limites : chaîne vide (erreur de validation), texte d'un seul mot, emoji, texte en
  anglais (comportement dégradé mais pas de crash), 200 001 caractères (erreur claire).
- `ner_batch` : lot mixte valide/invalide, ordre préservé.
- Démarrage : le serveur répond à `initialize` + `tools/list` sans charger le modèle.

### Tests fichiers et serveur
- Lecture fichier : BOM, latin-1 explicite, fichier vide, répertoire, fichier trop gros,
  `NER_ALLOWED_DIRS` respecté.
- `ner_extract_file` : `source` = chemin absolu ; le texte ne figure pas dans la réponse.
- Smoke test stdio réel : handshake, `tools/list` (4 outils), `ner_info` sans chargement,
  `ner_extract_file` sur le texte JO, `ner_batch` avec une entrée invalide isolée.

### Tests CLI
- Chaque format (`json`, `jsonl`, `text`, `highlight`, `markdown`) rendu sur un
  `ExtractionResult` figé : nombre d'entités, présence des labels, pas de couleur si `NO_COLOR`.
- Fichier vs STDIN sur le même contenu → JSON identique.
- Codes de sortie : fichier absent → 1, argument inconnu → 2, `--fail-on-empty` sans entité → 3.
- `--labels` et `--min-score` transmis au moteur (test avec moteur factice).
- `--stats` écrit sur **stderr**, la donnée reste seule sur stdout.

### Définition de « terminé »
1. `uv run pytest` vert, `uv run ruff check` propre.
2. `uvx --from . ner-mcp-fr` démarre et répond à une extraction via un client MCP.
3. README avec : installation, config hôte (opencode / Claude Desktop), 2 exemples
   d'appels MCP, **2 exemples CLI (fichier + pipe)**, limites connues, attribution du modèle
   (MIT, CATIE-AQ).
4. Aucun `print()` sur stdout, aucun `TODO` orphelin, commentaires de code en anglais.
5. Un premier lancement sur machine vierge réussit (téléchargement 422 MB du profil `fast` compris).

## 15. Risques

| Risque | Impact | Mitigation |
|---|---|---|
| Téléchargement initial (422 MB / 1.34 GB en `large`) | Première expérience décevante | Profil `fast` par défaut, log de progression sur stderr, `NER_PRELOAD` |
| CamemBERT ~111 M (`fast`) / 336 M (`large`) | Latence sur CPU | MPS par défaut, `fast`, batch |
| Offsets SentencePiece faux d'un caractère | Contrat violé | Tests d'invariant systématiques, avance de `start` sur les espaces |
| `MISC` bruité (F1 0.812–0.834) | Fausses entités | `min_score`, documentation du label, pas de filtrage magique |
| Modèle `large` déprécié par ses auteurs | Dette | Profil `fast` sur `NERmembert-base` (vérifié), model ID configurable |
| **Tokenizer CamemBERTa sans `merges.txt`** | Modèle officiellement « successeur » inutilisable | Écarté, test de non-régression sur le profil, constat documenté en §2 |
| `ner_extract_file` utilisé sur du contenu hostile (injection de prompt) | Lecture de fichiers sensibles | `NER_ALLOWED_DIRS`, taille bornée, chemin résolu journalisé |
| Confusion `FastMCP` (v1) vs `MCPServer` (v2) | Code qui ne démarre pas | Cible explicite SDK v2, doc liée à la version |
| Texte découpé à la mauvaise frontière | Entité coupée | Chevauchement d'une phrase + déduplication |

## 16. Jalons

- **M0 — Spec validée** : ce fichier, arbitrages §13.2 tranchés. ✅
- **M1 — Moteur** ✅ : `engine.py` + `chunking.py` + tests. Invariant d'offset vérifié sur
  10 198 caractères (5 fenêtres, 146 entités, 0 violation, 0 doublon).
- **M2 — Serveur** ✅ : 4 tools + schémas, smoke test stdio réel (handshake, `tools/list`,
  `ner_extract_file`, `ner_batch` avec entrée invalide isolée).
- **M3 — CLI + packaging** ✅ : `extract`/`info`, 5 formats, `pyproject.toml`, script, README.
  Reste à faire : configuration de l'hôte via `uvx --from git+…`.
- **M4 — Durcissement** : mesures de latence documentées, revue, commit initial.
- **M5 — Distribution** ✅ : paquet Publiable, CI GitHub, publication PyPI sans secret.

## 17. Distribution et publication

**La contrainte à ne pas se raconter.** `uvx ner-mcp-fr` résout le nom **sur PyPI**, pas sur
GitHub. Sans publication, la seule commande honnête est
`uvx --from git+https://github.com/athoune/ner-mcp-fr ner-mcp-fr`. Annoncer « ça marche depuis
GitHub » sans paquet publié serait faux.

| Élément | Rôle |
|---|---|
| `pyproject.toml` (PEP 621/639) | métadonnées complètes, `license = "MIT"` + `license-files`, URLs, classifiers |
| `LICENSE` | MIT, requis par `license-files` |
| `.github/workflows/ci.yml` | `lint`, `test` (Linux + macOS, py3.12/3.13), `package` (build + `uv tool run --from wheel` + `info`) |
| `.github/workflows/release.yml` | tag `v*` → CI → build → TestPyPI ou PyPI → **vérification de `uvx ner-mcp-fr`** → release GitHub |
| `docs/PUBLISHING.md` | procédure exacte, y compris le *pending publisher* |
| `.github/dependabot.yml` | actions et dépendances `uv` à jour |
| `scripts/export_onnx.py` | export fp16 reproductible + SHA-256 |
| `athoune/ner-mcp-fr-onnx` (HF) | héberge `model-fp16.onnx`, 210 MB ; le paquet PyPI reste à 40 kB |

Décisions :

- **Trusted publishing (OIDC), aucun secret dans le dépôt** : le job déclare
  `permissions: id-token: write` et un `environment: pypi` qui doit correspondre au publisher
  configuré sur PyPI. Pas de `PYPI_API_TOKEN` à faire tourner.
- **Le tag doit égaler la version** (`v$(uv version --short)`), sinon `build` échoue. Une version
  publiée ne se réécrit pas.
- **Le nom du paquet est le nom de la commande** : `uvx <paquet>` exige que le script porte le
  nom du paquet. Renommer l'un sans l'autre casse la promesse.
- **La CI est bloquante pour la publication** (`uses: ./.github/workflows/ci.yml`) : on ne publie
  jamais du code rouge.
- **`verify-published` est un gate** : si `uvx ner-mcp-fr` échoue après publication, la release
  GitHub n'est pas créée.
- Le paquet n'embarque **aucun poids de modèle** : 26 kB de wheel, le modèle est téléchargé à
  l'exécution depuis Hugging Face. C'est ce qui rend `uvx` instantané à installer (hors `torch`).

## 18. Bascule ONNX : mesures et options écartées

Décision du 2026-11-05, après mesures sur `NERmembert-base` (Corpus : 40 phrases, 880 mots ;
texte de référence : le paragraphe JO).

| Variante | Taille | F1 vs torch | Verdict |
|---|---|---|---|
| torch fp32 (avant) | 778 MB installés | référence | abandonné : 581 MB de framework |
| ONNX fp32 | 419 MB + 149 MB installés | **1,000** (40/40 phrases identiques) | correct, téléchargement lourd |
| **ONNX fp16** | 210 MB + 149 MB installés | **1,000** (40/40) | **retenu** |
| ONNX int8 dynamique | 106 MB | 0,526 | rejeté |
| ONNX int8 statique (QDQ, calibration 40 phrases) | 106 MB | 0,000 | rejeté |

Latence (Mac, CPU) : 71 ms par paragraphe en fp32, 80 ms en fp16, 1,2 s pour 10 kB. torch + MPS
faisait 16 ms et 103 ms : c'est le prix de 5× moins de taille, assumé pour un usage par
paragraphe.

Providers locaux (Ollama, LM Studio, llama.cpp, MLX) : **écartés**. Ce sont des serveurs de
modèles génératifs ; il n'existe pas d'API de token-classification avec scores et offsets. Il
faudrait demander du JSON à un LLM, ce qui casse l'invariant `text[start:end] == entity.text`,
remplace une probabilité calibrée par une génération non déterministe, coûte des tokens par
document et déplace la taille (modèle plus gros + serveur à installer) au lieu de la réduire. MLX
est de surcroît macOS-only, donc incompatible avec un paquet `uvx` portable.

CoreML EP : mesuré plus lent que le CPU (29 nœuds couverts sur 597). Écarté.
