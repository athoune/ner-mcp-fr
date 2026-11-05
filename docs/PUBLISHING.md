# Publication

Objectif : `uvx ner-mcp-fr` doit fonctionner pour n'importe qui, sans clone ni installation
manuelle.

**Ce que `uvx` ne fait pas.** `uvx ner-mcp-fr` résout le nom **sur PyPI**. GitHub seul ne suffit
pas : tant que le paquet n'est pas publié, la commande équivalente est
`uvx --from git+https://github.com/athoune/ner-my-french ner-mcp-fr`. Ce document met en place la
publication automatique depuis GitHub, puis la voie de secours.

---

## 1. Créer le dépôt et pousser `main`

```bash
# depuis le dépôt local
git add -A
git commit -m "Initial implementation: French NER MCP server and verification CLI"
git remote add origin git@github.com:athoune/ner-my-french.git
git push -u origin main
```

Le nom `ner-mcp-fr` est **libre sur PyPI** (vérifié le 2026-11-05). Si tu publies sous un autre
nom, change `[project] name` **et** `[project.scripts]` dans `pyproject.toml` : `uvx <nom>` exige
que la commande porte le nom du paquet.

## 2. Déclarer le *pending publisher* sur PyPI

Le workflow n'embarque aucun secret : il utilise le **trusted publishing** (OIDC). Encore
faut-il que PyPI sache à quel dépôt faire confiance.

Sur <https://pypi.org/manage/account/publishing/> → *Add a pending publisher* :

| Champ | Valeur |
|---|---|
| PyPI Project Name | `ner-mcp-fr` |
| Owner | `athoune` |
| Repository name | `ner-my-french` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

« Pending » parce que le projet n'existe pas encore : la première publication le crée.

## 3. (Recommandé) Même chose sur TestPyPI

<https://test.pypi.org/manage/account/publishing/> avec **Environment name = `testpypi`**.
Cela permet de répéter la publication sans brûler un numéro de version sur PyPI.

## 4. Répéter à blanc sur TestPyPI

Onglet **Actions** → *Release* → *Run workflow* → `testpypi`. Le workflow :

1. lance la CI complète (lint, tests Linux + macOS, build du wheel) ;
2. construit `dist/` et vérifie les métadonnées (`twine check`) ;
3. publie sur TestPyPI via OIDC.

Vérification, sans polluer l'environnement :

```bash
uvx --index-url https://test.pypi.org/simple/ --from ner-mcp-fr ner-mcp-fr info
```

TestPyPI ne résout pas toujours les dépendances (elles n'y sont pas toutes) : en cas d'échec à
l'installation, c'est normal, seule la publication nous intéresse ici.

## 5. Publier pour de vrai

```bash
git tag v0.1.0
git push origin v0.1.0
```

Le job `build` refuse le tag si `v$(uv version --short)` ne correspond pas exactement : impossible
de publier une version qui ne correspond pas au code. Puis :

1. `publish-pypi` pousse sur PyPI ;
2. `verify-published` exécute `uvx --refresh ner-mcp-fr --version` **sans checkout du dépôt**,
   avec plusieurs tentatives le temps que PyPI propage la version — c'est le test de la commande
   exacte que tapera un utilisateur ;
3. `github-release` attache `dist/*` à la release et génère les notes.

Si `verify-published` échoue, aucune release GitHub n'est créée : le contrat « `uvx ner-mcp-fr`
fonctionne » est bloquant, pas décoratif.

Vérification finale, la seule qui compte :

```bash
uvx ner-mcp-fr --version
uvx ner-mcp-fr info
```

## 6. Sortir une version suivante

```bash
uv version --bump patch     # ou minor / major ; met à jour pyproject.toml et uv.lock
git commit -am "chore: release $(uv version --short)"
git tag "v$(uv version --short)"
git push && git push --tags
```

Une version PyPI ne se réécrit pas : en cas d'erreur après publication, on incrémente, on ne
retague pas.

## Dépannage

| Symptôme | Cause probable |
|---|---|
| `Trusted publishing exchange failure` | Owner / repo / workflow / environment ne correspondent pas exactement au pending publisher |
| `File already exists` | Cette version est déjà publiée : incrémente |
| `Tag … does not match v…` | Le tag et `pyproject.toml` divergent |
| `uvx ner-mcp-fr` → `No solution found` | Publication absente ou nom de paquet ≠ nom de commande |
| Le job `check` échoue | La CI est bloquante par construction : on ne publie pas du code rouge |
