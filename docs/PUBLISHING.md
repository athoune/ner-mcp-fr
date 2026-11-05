# Publication

Objectif : `uvx ner-mcp-fr` doit fonctionner pour n'importe qui, sans clone ni installation
manuelle.

**Ce que `uvx` ne fait pas.** `uvx ner-mcp-fr` résout le nom **sur PyPI**. GitHub seul ne suffit
pas : tant que le paquet n'est pas publié, la commande équivalente est
`uvx --from git+https://github.com/athoune/ner-mcp-fr ner-mcp-fr`.

---

## Ce qu'il faut faire, dans l'ordre

| # | Où | Action |
|---|---|---|
| 1 | **GitHub** | créer le dépôt `athoune/ner-mcp-fr` et pousser `main` |
| 2 | **pypi.org** (web) | déclarer un *pending publisher* — le seul passage manuel obligatoire |
| 3 | **TestPyPI** (web) | même chose, pour répéter sans brûler de version (§4) |
| 4 | **GitHub Actions** | lancer *Release* → `testpypi` (§5) |
| 5 | **Terminal** | taguer `v0.1.0`, la CI publie (§6) |
| 6 | **Terminal** | `uvx ner-mcp-fr info` (§7) |

Les étapes 2, 3 et 4 sont optionnelles si tu veux publier du premier coup : les étapes 1, 2, 5 et 6
suffisent. 3 et 4 servent à éviter un `0.1.0` cassé sur PyPI, qui ne se réécrit pas.

---

## 1. Créer le dépôt et pousser `main`

```bash
# les commits existent déjà : il ne manque que le dépôt distant
gh repo create athoune/ner-mcp-fr --public --source=. --remote=origin --push

# ou, sans gh
git remote add origin git@github.com:athoune/ner-mcp-fr.git
git push -u origin main
```

Le nom du dépôt, celui du paquet et celui de la commande sont **le même** : `ner-mcp-fr`. C'est la
condition pour que `uvx ner-mcp-fr` ait un sens.

## 2. Ce qu'il faut cliquer sur PyPI

C'est le seul passage qui ne peut pas être automatisé : PyPI doit savoir *à l'avance* de quel
dépôt il accepte des publications. Aucun mot de passe ni jeton n'est créé — l'authentification
passe par OIDC, GitHub prouve son identité au moment du job.

**Prérequis** : un compte sur <https://pypi.org> avec email vérifié et **2FA activée** (obligatoire
pour tous les comptes PyPI).

1. Se connecter, puis ouvrir <https://pypi.org/manage/account/publishing/> — c'est dans la barre
   latérale du compte, section **Publishing**.
2. Cliquer sur **Add a pending publisher**.
3. Choisir l'onglet **GitHub**.
4. Remplir **exactement** ces cinq champs :

| Champ affiché | Valeur à saisir |
|---|---|
| PyPI Project Name | `ner-mcp-fr` |
| Owner | `athoune` |
| Repository name | `ner-mcp-fr` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

5. Cliquer sur **Add**.

Ces valeurs ne sont pas décoratives : PyPI compare la revendication OIDC du job (dépôt, fichier de
workflow, environnement) à ce formulaire, à la chaîne près. Une faute de frappe donne
`Trusted publishing exchange failure` dans les logs du job.

« Pending » parce que le projet n'existe pas encore : à la première publication réussie, PyPI
**crée** le projet et convertit ce publisher en publisher normal. Plus rien à configurer ensuite.

> ⚠️ Un pending publisher **ne réserve pas le nom** tant qu'il n'a pas servi. Si quelqu'un
> enregistre `ner-mcp-fr` avant ta première publication, ton publisher devient invalide. Le nom
> est libre au 2026-11-05 (vérifié), donc fais l'étape 5 peu après l'étape 2.

## 3. Le workflow de publication, pour mémoire

Rien à faire : c'est déjà dans le dépôt.

- `.github/workflows/release.yml` se déclenche sur un tag `v*` (ou manuellement).
- Il appelle la CI (`ci.yml`), refuse un tag qui ne correspond pas à `pyproject.toml`, construit
  `dist/`, vérifie les métadonnées (`twine check`).
- `publish-pypi` déclare `permissions: id-token: write` et `environment: pypi` : ce sont ces deux
  éléments qui autorisent l'échange OIDC. **Aucun secret n'est stocké dans le dépôt.**

## 4. (Recommandé) Répéter sur TestPyPI

<https://test.pypi.org/manage/account/publishing/> → *Add a pending publisher* → GitHub, mêmes
champs, sauf :

| Champ | Valeur |
|---|---|
| Environment name | `testpypi` |

Cela permet de répéter la publication sans consommer un numéro de version sur PyPI.

## 5. Répéter à blanc

Onglet **Actions** du dépôt → *Release* → **Run workflow** → `testpypi`.

Contrôle de la publication (l'installation peut échouer : TestPyPI ne miroite pas toutes les
dépendances, seule la publication nous intéresse) :

```bash
uvx --index-url https://test.pypi.org/simple/ --from ner-mcp-fr ner-mcp-fr info
```

## 6. Publier pour de vrai

```bash
git tag v0.1.0
git push origin v0.1.0
```

Enchaînement des jobs :

1. `check` — la CI complète ; si elle est rouge, rien n'est publié ;
2. `build` — wheel + sdist, `twine check`, et refus si `v$(uv version --short)` ≠ tag ;
3. `publish-pypi` — upload OIDC ;
4. `verify-published` — exécute `uvx --refresh ner-mcp-fr --version` **sans checkout du dépôt**,
   avec plusieurs tentatives le temps que PyPI propage ;
5. `github-release` — attache `dist/*` à la release GitHub.

Si `verify-published` échoue, aucune release GitHub n'est créée : le contrat « `uvx ner-mcp-fr`
fonctionne » est bloquant, pas décoratif.

## 7. Vérifier, la seule commande qui compte

```bash
uvx ner-mcp-fr --version
uvx ner-mcp-fr info
```

## 8. Sortir une version suivante

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
| `Trusted publishing exchange failure` | Owner / repository / workflow / environment ne correspondent pas exactement au pending publisher |
| `File already exists` | Cette version est déjà publiée : incrémente |
| `Tag … does not match v…` | Le tag et `pyproject.toml` divergent |
| `uvx ner-mcp-fr` → `No solution found` | Publication absente, ou nom de paquet ≠ nom de commande |
| Le job `check` échoue | La CI est bloquante par construction : on ne publie pas du code rouge |
| Le pending publisher a disparu | Le nom a été enregistré par quelqu'un d'autre entre-temps |
