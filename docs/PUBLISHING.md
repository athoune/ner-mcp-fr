# Publication

Objectif : `uvx ner-mcp-fr` doit fonctionner pour n'importe qui, sans clone ni installation
manuelle.

**Ce que `uvx` ne fait pas.** `uvx ner-mcp-fr` résout le nom **sur PyPI**. GitHub seul ne suffit
pas : tant que le paquet n'est pas publié, la commande équivalente est
`uvx --from git+https://github.com/athoune/ner-mcp-fr ner-mcp-fr`.

**Deux choses se publient, pas une.** Le paquet Python (PyPI, ~40 kB) et **les poids ONNX**
(Hugging Face, 210 MB). PyPI refuse les fichiers de plus de 100 MB, et `onnxruntime` ne sait pas
lire un checkpoint PyTorch : les poids ne peuvent pas vivre dans le paquet.

---

## Ce qu'il faut faire, dans l'ordre

| # | Où | Action |
|---|---|---|
| 1 | **GitHub** | créer `athoune/ner-mcp-fr` et pousser `main` |
| 2 | **Hugging Face** | créer `athoune/ner-mcp-fr-onnx` et y déposer `model-fp16.onnx` |
| 3 | **pypi.org** | déclarer un *pending publisher* |
| 4 | **TestPyPI** | même chose, pour répéter sans brûler de version (§5) |
| 5 | **GitHub Actions** | *Release* → `testpypi` (§6) |
| 6 | **Terminal** | taguer `v0.1.0`, la CI publie (§7) |
| 7 | **Terminal** | `uvx ner-mcp-fr info` (§8) |

Les étapes 1, 2, 3, 6, 7 sont le chemin minimal. 4 et 5 évitent de publier un `0.1.0` cassé sur
PyPI, où une version ne se réécrit pas.

---

## 1. Créer le dépôt et pousser `main`

```bash
# les commits existent déjà : il ne manque que le dépôt distant
gh repo create athoune/ner-mcp-fr --public --source=. --remote=origin --push

# ou, sans gh
git remote add origin git@github.com:athoune/ner-mcp-fr.git
git push -u origin main
```

## 2. Déposer les poids ONNX sur Hugging Face

**Pourquoi.** `onnxruntime` ne lit pas les `.safetensors` de CATIE-AQ : il lui faut un graphe
`.onnx`. On ne peut pas le déposer chez CATIE-AQ, et PyPI limite les fichiers à 100 MB. D'où un
dépôt Hugging Face à nous, public, qui contient **un seul fichier de 210 MB**.

Rien d'autre n'est nécessaire : `tokenizer.json` et `config.json` continuent d'être téléchargés
depuis `CATIE-AQ/NERmembert-base-4entities`.

**a. Produire le fichier** (si ce n'est pas déjà fait) :

```bash
uv run --group export python scripts/export_onnx.py
# → dist/onnx/model-fp16.onnx (210 MB) + son SHA-256
```

**b. Créer le dépôt** sur <https://huggingface.co/new> :

| Champ | Valeur |
|---|---|
| Owner | `athoune` |
| Model name | `ner-mcp-fr-onnx` |
| Visibility | **Public** (sinon chaque utilisateur aurait besoin d'un jeton) |
| License | MIT |

**c. Déposer le fichier**, au choix :

- **Interface web** : onglet *Files* → *Add file* → *Upload files* → glisser
  `dist/onnx/model-fp16.onnx` → *Commit changes*.
- **Ligne de commande** (plus fiable pour 210 MB) :

  ```bash
  uvx --from huggingface_hub hf auth login        # jeton avec droit d'écriture
  uvx --from huggingface_hub hf upload athoune/ner-mcp-fr-onnx \
      dist/onnx/model-fp16.onnx model-fp16.onnx
  ```

**d. Vérifier** que cette URL télécharge bien le fichier :

```bash
curl -sIL https://huggingface.co/athoune/ner-mcp-fr-onnx/resolve/main/model-fp16.onnx | head -3
```

**e. Vérifier le SHA-256.** `src/ner_mcp/config.py` épingle
`DEFAULT_ARTIFACT_SHA256`. Il doit correspondre à :

```bash
shasum -a 256 dist/onnx/model-fp16.onnx
# attendu: afd6f6dad996fc2585ad33615e8434ba2734a8270e9e7ebc0deb217e062cd2b5
```

Si tu réexportes les poids, le hash change : mets-le à jour dans `config.py` et dans un commit.
Un artefact remplacé sans mise à jour échoue **explicitement** au premier lancement, c'est voulu.

> Pas d'accord avec Hugging Face ? L'alternative est un asset de release GitHub, mais elle demande
> de modifier `src/ner_mcp/artifacts.py` (l'URL n'est plus un repo HF). Dis-le avant de publier.

## 3. Ce qu'il faut cliquer sur PyPI

C'est le seul passage qui ne peut pas être automatisé : PyPI doit savoir *à l'avance* de quel
dépôt il accepte des publications. Aucun mot de passe ni jeton n'est créé — l'authentification
passe par OIDC, GitHub prouve son identité au moment du job.

**Prérequis** : un compte sur <https://pypi.org> avec email vérifié et **2FA activée** (obligatoire
pour tous les comptes PyPI).

1. Se connecter, puis ouvrir <https://pypi.org/manage/account/publishing/> — barre latérale du
   compte, section **Publishing**.
2. **Add a pending publisher**.
3. Onglet **GitHub**.
4. Remplir **exactement** :

| Champ affiché | Valeur |
|---|---|
| PyPI Project Name | `ner-mcp-fr` |
| Owner | `athoune` |
| Repository name | `ner-mcp-fr` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

5. **Add**.

Ces valeurs ne sont pas décoratives : PyPI compare la revendication OIDC du job (dépôt, workflow,
environnement) à ce formulaire, à la chaîne près. Une faute de frappe donne
`Trusted publishing exchange failure` dans les logs, sans autre indice.

« Pending » parce que le projet n'existe pas encore : à la première publication réussie, PyPI
**crée** le projet et convertit ce publisher en publisher normal. Plus rien à configurer ensuite.

> ⚠️ Un pending publisher **ne réserve pas le nom** tant qu'il n'a pas servi. Si quelqu'un
> enregistre `ner-mcp-fr` avant ta première publication, ton publisher devient invalide.

## 4. Le workflow de publication, pour mémoire

Rien à faire : c'est déjà dans le dépôt.

- `.github/workflows/release.yml` se déclenche sur un tag `v*` (ou manuellement).
- Il appelle la CI, refuse un tag qui ne correspond pas à `pyproject.toml`, construit `dist/`,
  vérifie les métadonnées (`twine check`).
- `publish-pypi` déclare `permissions: id-token: write` et `environment: pypi` : ce sont ces deux
  éléments qui autorisent l'échange OIDC. **Aucun secret n'est stocké dans le dépôt.**

## 5. (Recommandé) Répéter sur TestPyPI

<https://test.pypi.org/manage/account/publishing/> → *Add a pending publisher* → GitHub, mêmes
champs, sauf `Environment name` = **`testpypi`**.

## 6. Répéter à blanc

Onglet **Actions** → *Release* → **Run workflow** → `testpypi`.

Contrôle (l'installation peut échouer : TestPyPI ne miroite pas toutes les dépendances, seule la
publication nous intéresse) :

```bash
uvx --index-url https://test.pypi.org/simple/ --from ner-mcp-fr ner-mcp-fr info
```

## 7. Publier pour de vrai

```bash
git tag v0.1.0
git push origin v0.1.0
```

Enchaînement : `check` (CI complète) → `build` (wheel + sdist, `twine check`, contrôle du tag) →
`publish-pypi` (OIDC) → `verify-published` (`uvx --refresh ner-mcp-fr --version`, sans checkout,
avec plusieurs tentatives le temps que PyPI propage) → `github-release`.

Si `verify-published` échoue, aucune release GitHub n'est créée : le contrat « `uvx ner-mcp-fr`
fonctionne » est bloquant, pas décoratif.

## 8. Vérifier, la seule commande qui compte

```bash
uvx ner-mcp-fr --version
uvx ner-mcp-fr info
```

Le premier `extract` télécharge 210 MB de poids depuis Hugging Face, puis les met en cache
(`~/.cache/huggingface`). `NER_OFFLINE=1` échoue proprement si le cache est vide.

## 9. Sortir une version suivante

```bash
uv version --bump patch     # met à jour pyproject.toml et uv.lock
git commit -am "chore: release $(uv version --short)"
git tag "v$(uv version --short)"
git push && git push --tags
```

Une version PyPI ne se réécrit pas : en cas d'erreur après publication, on incrémente, on ne
retague pas.

## Dépannage

| Symptôme | Cause probable |
|---|---|
| `cannot fetch model-fp16.onnx` | Dépôt HF absent, privé, ou nom de fichier différent |
| `checksum mismatch` | Poids réexportés sans mettre à jour `DEFAULT_ARTIFACT_SHA256`, ou téléchargement corrompu (supprimer le cache) |
| `Trusted publishing exchange failure` | Owner / repository / workflow / environment ne correspondent pas exactement au pending publisher |
| `File already exists` | Cette version est déjà publiée : incrémente |
| `Tag … does not match v…` | Le tag et `pyproject.toml` divergent |
| `uvx ner-mcp-fr` → `No solution found` | Publication absente, ou nom de paquet ≠ nom de commande |
| Le job `check` échoue | La CI est bloquante par construction |
| Le pending publisher a disparu | Le nom a été enregistré par quelqu'un d'autre entre-temps |
