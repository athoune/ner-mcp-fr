# ner-mcp-fr

[![CI](https://github.com/athoune/ner-mcp-fr/actions/workflows/ci.yml/badge.svg)](https://github.com/athoune/ner-mcp-fr/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ner-mcp-fr)](https://pypi.org/project/ner-mcp-fr/)
[![Python](https://img.shields.io/pypi/pyversions/ner-mcp-fr)](https://pypi.org/project/ner-mcp-fr/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Reconnaissance d'entités nommées (NER) en français, exposée comme **serveur MCP** — avec un
**CLI** pour vérifier ce que le modèle fait réellement. Inférence **ONNX**, sans torch.

Quatre labels : `PER` (personne), `LOC` (lieu), `ORG` (organisation), `MISC` (divers).

Le modèle est
[`CATIE-AQ/NERmembert-base-4entities`](https://huggingface.co/CATIE-AQ/NERmembert-base-4entities)
(CamemBERT-base, 110 M de paramètres, fine-tuné sur `frenchNER_4entities`, licence MIT), exporté
en ONNX fp16 et hébergé séparément des poids PyTorch.

> ⚠️ **Ne pas utiliser `NERmemberta-*` ni `NERmembert2-*`.** Ces modèles, pourtant marqués
> `new_version` par leurs auteurs, publient un tokenizer byte-level sans règles de fusion : chaque
> mot est découpé en lettres (`Ġ S y l v a i n`) et les prédictions sont du bruit. Constat vérifié
> et documenté dans `SPECS.md` §2.

## Installation

Rien à installer globalement : `uvx` construit un environnement jetable à la volée.

```bash
uvx ner-mcp-fr info      # ne charge pas le modèle, sert de test de fumée
uvx ner-mcp-fr extract article.txt --format highlight
```

Empreinte : **~150 MB installés** (onnxruntime, tokenizers, numpy, mcp), contre ~780 MB avec
une pile PyTorch. Le premier appel télécharge les poids **fp16 (210 MB)** depuis Hugging Face,
puis les garde en cache. Compromis assumé : ~70 ms par paragraphe sur CPU (contre ~16 ms avec
torch + MPS sur Mac), pour un cinquième de la taille.

Tant que le paquet n'est pas publié sur PyPI, la même chose depuis GitHub :

```bash
uvx --from git+https://github.com/athoune/ner-mcp-fr ner-mcp-fr info
```

Pour développer :

```bash
git clone https://github.com/athoune/ner-mcp-fr
cd ner-mcp-fr
uv sync
uv run pytest
```

## Le CLI : vérifier une analyse

```bash
# un fichier, rendu lisible avec les entités colorées
ner-mcp-fr extract article.txt --format highlight

# depuis STDIN
cat article.txt | ner-mcp-fr extract

# un paragraphe collé directement
pbpaste | ner-mcp-fr extract --format highlight --stats
```

Exemple de sortie `--format text` :

```
LABEL  SCORE   SPAN        TEXT
-----  ------  ----------  -------------------
MISC   0.9456  [36,38)     'JO'
LOC    0.9993  [75,84)     'Grand Rex'
PER    0.9989  [165,178)   'Sylvain Boyer'
```

Formats disponibles :

| Format | Usage |
|---|---|
| `json` | schéma complet (défaut quand la sortie est redirigée) |
| `jsonl` | une entité par ligne, pour `jq` |
| `text` | tableau aligné (défaut sur un terminal) |
| `highlight` | texte d'origine annoté, en couleur |
| `markdown` | tableau + paragraphe annoté, pour un ticket |

Options utiles :

```bash
ner-mcp-fr extract article.txt --labels PER,LOC      # filtre les labels
ner-mcp-fr extract article.txt --min-score 0.9       # exige de la confiance
ner-mcp-fr extract article.txt --keep-punctuation    # spans bruts du modèle
ner-mcp-fr extract article.txt --onnx ./model.onnx   # poids locaux, sans téléchargement
ner-mcp-fr extract article.txt --stats               # chunks/timing sur stderr
ner-mcp-fr extract article.txt --fail-on-empty       # code 3 si aucune entité
```

Codes de sortie : `0` succès, `1` erreur d'exécution, `2` erreur d'usage, `3` `--fail-on-empty`
sans entité.

## Serveur MCP

`ner-mcp-fr` sans argument (ou `ner-mcp-fr mcp`) démarre le serveur sur **stdio**.
Configuration d'un hôte :

```json
{
  "mcpServers": {
    "ner-fr": {
      "command": "uvx",
      "args": ["ner-mcp-fr"]
    }
  }
}
```

Quatre outils :

| Outil | Rôle |
|---|---|
| `ner_extract` | un texte inline → entités |
| `ner_extract_file` | **un chemin** → entités, fichier lu côté serveur |
| `ner_batch` | jusqu'à 64 textes en une passe ; une entrée invalide n'annule pas le lot |
| `ner_info` | modèle, précision, provider, contexte, statut — sans charger |

**Pourquoi `ner_extract_file` existe** : un texte de 200 kB collé dans `ner_extract` traverse le
contexte du modèle deux fois (lecture + historique). Avec un chemin, il ne traverse jamais rien :
le serveur lit le fichier en local. `source` dans la réponse est le chemin absolu résolu, pour
tracer l'origine sans renvoyer le texte. `NER_ALLOWED_DIRS` restreint les chemins lisibles si le
serveur ingère du contenu non fiable.

Réponse de `ner_extract` :

```json
{
  "model": "CATIE-AQ/NERmembert-base-4entities",
  "precision": "fp16",
  "device": "CPUExecutionProvider",
  "entities": [
    { "text": "Paris", "label": "LOC", "start": 334, "end": 339, "score": 0.9991 }
  ],
  "source": "/Users/moi/article.txt",
  "text_length": 617,
  "chunks": 1,
  "truncated": false,
  "dropped_by_min_score": 0,
  "elapsed_ms": 214
}
```

**Invariant** : pour chaque entité, `text[start:end] == entity.text`. Les offsets sont des index
de *code points* Unicode (sémantique Python). Si un client JavaScript manipule des offsets sur du
texte contenant des emoji, préférer le champ `text`.

## Configuration

| Variable | Défaut | Effet |
|---|---|---|
| `NER_MODEL_ID` | `CATIE-AQ/NERmembert-base-4entities` | source du tokenizer et de la table de labels |
| `NER_ARTIFACT_REPO` | `athoune/ner-mcp-fr-onnx` | dépôt Hugging Face des poids ONNX |
| `NER_ARTIFACT_FILE` | `model-fp16.onnx` | fichier de poids dans ce dépôt |
| `NER_ONNX_PATH` | — | fichier ONNX local, court-circuite le téléchargement |
| `NER_DEVICE` | `auto` | `auto`, `cpu`, `cuda` (provider onnxruntime) |
| `NER_MAX_INPUT_CHARS` | `200000` | garde-fou de taille, aussi borne des fichiers |
| `NER_BATCH_SIZE` | `8` | fenêtres par passe |
| `NER_MIN_SCORE` | `0.0` | seuil de confiance par défaut |
| `NER_CACHE_DIR` | — | cache Hugging Face alternatif |
| `NER_PRELOAD` | `0` | charge le modèle au démarrage |
| `NER_OFFLINE` | `0` | `1` = aucun accès réseau |
| `NER_THREADS` | — | threads onnxruntime |
| `NER_LOG_LEVEL` | `INFO` | logs sur stderr (stdout = protocole) |
| `NER_ALLOWED_DIRS` | — | répertoires autorisés pour `ner_extract_file` (`:`) |

## Limites connues

- **Français uniquement.** Aucune détection de langue : sur de l'anglais, le modèle produit des
  entités plausibles mais non fiables.
- **Contexte 512 tokens** (~350 mots). Au-delà, le texte est découpé en fenêtres avec
  chevauchement d'une phrase ; `chunks` l'indique. Rien n'est tronqué silencieusement.
- **CPU uniquement**, sauf `onnxruntime-gpu` installé et `NER_DEVICE=cuda`. Le provider CoreML a
  été mesuré plus lent que le CPU sur ce graphe (29 nœuds accélérés sur 597).
- **fp16** : sortie identique au fp32 sur le corpus de référence, mais la quantization int8, elle,
  détruit le modèle (F1 0,53 en dynamique, 0,00 en statique). Ne pas la réintroduire sans mesure.
- `MISC` est le label le plus faiblement performant (F1 ≈ 0,81) : attendez-vous à du bruit.
- La ponctuation et le balisage Markdown de bord sont retirés des entités (`Grand Rex.` →
  `Grand Rex`, `Ebenezer**` → `Ebenezer`) ; `--keep-punctuation` rend les spans bruts.

## Développement

```bash
uv run pytest        # tests unitaires (aucun modèle téléchargé)
uv run ruff check .
```

Re-exporter les poids (seule étape qui demande torch) :

```bash
uv run --group export python scripts/export_onnx.py
# écrit dist/onnx/model-fp16.onnx et affiche son SHA-256
```

Architecture : `backend.py` (session ONNX + tokenizer), `artifacts.py` (téléchargement, cache,
vérification du SHA-256), `engine.py` (chunking, agrégation — sans MCP), `cli.py` + `render.py`
(CLI), `server.py` (quatre outils MCP). Le contrat complet est dans `SPECS.md`.

## Attribution

Modèles fine-tunés par [CATIE-AQ](https://huggingface.co/CATIE-AQ) sur le jeu
`frenchNER_4entities`, publiés sous licence MIT. Le graphe ONNX est un export de
`NERmembert-base-4entities`, distribué sous la même licence.
