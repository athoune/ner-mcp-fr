# ner-mcp-fr

[![CI](https://github.com/athoune/ner-mcp-fr/actions/workflows/ci.yml/badge.svg)](https://github.com/athoune/ner-mcp-fr/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ner-mcp-fr)](https://pypi.org/project/ner-mcp-fr/)
[![Python](https://img.shields.io/pypi/pyversions/ner-mcp-fr)](https://pypi.org/project/ner-mcp-fr/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Reconnaissance d'entités nommées (NER) en français, exposée comme **serveur MCP** — avec un
**CLI** pour vérifier ce que le modèle fait réellement.

Quatre labels : `PER` (personne), `LOC` (lieu), `ORG` (organisation), `MISC` (divers).

Le serveur n'entraîne rien : il s'appuie sur les modèles
[`CATIE-AQ/NERmembert-base-4entities`](https://huggingface.co/CATIE-AQ/NERmembert-base-4entities)
(défaut) et
[`CATIE-AQ/NERmembert-large-4entities`](https://huggingface.co/CATIE-AQ/NERmembert-large-4entities),
fine-tunés sur `frenchNER_4entities` (~385 k phrases), sous licence MIT.

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

Premier lancement : `uv` télécharge `torch` et `transformers` (~1 GB), puis le modèle
(**420 MB** en profil `fast`, 1.34 GB en `large`) au premier appel. Côté mémoire, comptez
~1.5 GB contre ~2.5 GB.

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

La publication PyPI est automatisée depuis GitHub (tag `v*` → CI → TestPyPI/PyPI → release) :
voir [`docs/PUBLISHING.md`](docs/PUBLISHING.md).

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
LOC    0.9991  [334,339)   'Paris'
ORG    0.9912  [361,370)   'Grand Rex'
PER    0.9987  [467,480)   'Sylvain Boyer'
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
ner-mcp-fr extract article.txt --stats               # chunks/timing sur stderr
ner-mcp-fr extract article.txt --profile large       # meilleur F1, contexte 512 tokens
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
      "args": ["ner-mcp-fr"],
      "env": { "NER_PROFILE": "fast" }
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
| `ner_info` | modèle, device, contexte, statut de chargement — sans charger |

**Pourquoi `ner_extract_file` existe** : un texte de 200 kB collé dans `ner_extract` traverse le
contexte du modèle deux fois (lecture + historique). Avec un chemin, il ne traverse jamais rien :
le serveur lit le fichier en local. Dès qu'un texte vient d'un fichier, préférez cette voie.
`source` dans la réponse est le chemin absolu résolu, pour tracer l'origine sans renvoyer le
texte. `NER_ALLOWED_DIRS` restreint les chemins lisibles si le serveur ingère du contenu non
fiable.

Réponse de `ner_extract` :

```json
{
  "model": "CATIE-AQ/NERmembert-base-4entities",
  "profile": "fast",
  "device": "mps",
  "source": "/Users/moi/article.txt",
  "entities": [
    { "text": "Paris", "label": "LOC", "start": 334, "end": 339, "score": 0.9991 }
  ],
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

Tout se règle par variables d'environnement (voir `SPECS.md` §11) :

| Variable | Défaut | Effet |
|---|---|---|
| `NER_PROFILE` | `fast` | `fast` (420 MB, 512 tokens) ou `large` (1.34 GB, 512 tokens) |
| `NER_MODEL_ID` | — | model ID explicite, prioritaire sur le profil |
| `NER_DEVICE` | `auto` | `auto`, `cpu`, `mps`, `cuda` |
| `NER_MAX_INPUT_CHARS` | `200000` | garde-fou de taille d'entrée |
| `NER_BATCH_SIZE` | `8` | fenêtres par passe |
| `NER_MIN_SCORE` | `0.0` | seuil de confiance par défaut |
| `NER_CACHE_DIR` | — | cache Hugging Face alternatif |
| `NER_PRELOAD` | `0` | charge le modèle au démarrage |
| `NER_OFFLINE` | `0` | `1` = aucun accès réseau |
| `NER_LOG_LEVEL` | `INFO` | logs sur stderr (stdout = protocole) |
| `NER_ALLOWED_DIRS` | — | répertoires autorisés pour `ner_extract_file` (`:`) |

## Limites connues

- **Français uniquement.** Aucune détection de langue : sur de l'anglais, le modèle produit des
  entités plausibles mais non fiables.
- **Contexte 512 tokens** dans les deux profils. Au-delà, le texte est découpé en fenêtres avec
  chevauchement d'une phrase ; `chunks` l'indique. Rien n'est tronqué silencieusement.
- La précision des poids n'est pas réglable : elle vient du checkpoint. Un half-precision forcé
  changerait les prédictions sans le dire.
- `MISC` est le label le plus faiblement performant (F1 ≈ 0.81) : attendez-vous à du bruit.
- La ponctuation et le balisage Markdown de bord sont retirés des entités (`Grand Rex.` →
  `Grand Rex`, `Ebenezer**` → `Ebenezer`, `| Paris |` → `Paris`) ; `--keep-punctuation` rend les
  spans bruts, utile pour auditer le modèle lui-même.
- Le profil `fast` privilégie vitesse et empreinte mémoire ; `large` reste le meilleur F1
  (0.982 / 0.964 / 0.919 / 0.834 contre 0.979 / 0.961 / 0.915 / 0.812).

## Développement

```bash
uv run pytest        # tests unitaires (aucun modèle téléchargé)
uv run ruff check .
```

Architecture : `engine.py` (modèle, chunking, agrégation — sans MCP), `cli.py` + `render.py`
(CLI), `server.py` (trois outils MCP). Le contrat complet est dans `SPECS.md`.

## Attribution

Modèles fine-tunés par [CATIE-AQ](https://huggingface.co/CATIE-AQ) sur le jeu
`frenchNER_4entities`, publiés sous licence MIT. Ce projet ne fait que les servir.
