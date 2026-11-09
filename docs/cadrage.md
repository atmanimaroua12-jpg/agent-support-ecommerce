# Cadrage du projet — Agent de support e-commerce

## 1. Problème et utilisateurs

**Contexte.** ShopNova est une boutique en ligne fictive (Maroc). Son service client reçoit
beaucoup de demandes répétitives : délais de livraison, conditions de retour, garantie, statut
d'une commande. Les conseillers humains perdent du temps sur ces questions simples.

**Objectif.** Un agent conversationnel qui :
- répond aux questions sur les politiques (retours, livraison, garantie, paiement) en citant la documentation ;
- consulte le statut d'une commande du client connecté ;
- prépare une escalade vers un conseiller humain quand la situation le demande.

**Utilisateurs.**
- *Clients* : posent les questions via le chat.
- *Conseillers humains* : reçoivent les tickets d'escalade.
- *Équipe technique* : surveille les traces et les métriques.

**Conséquences d'une erreur.** Une mauvaise information (ex. délai de retour faux) peut
entraîner un litige ou une perte de confiance. Une fuite de commande d'un autre client est une
violation de données personnelles (loi 09-08). Ces risques justifient les garde-fous ci-dessous.

## 2. Ce que l'agent ne doit PAS faire

| Interdit | Mesure prévue |
|---|---|
| Effectuer un remboursement, annuler ou modifier une commande | Aucun outil ne le permet ; l'agent explique la procédure et propose une escalade |
| Afficher la commande d'un autre client | L'identité du client vient de la session, pas du LLM ; vérification du propriétaire dans l'outil |
| Créer un ticket sans l'accord du client | L'outil crée un *brouillon* ; seule la confirmation du client crée le ticket |
| Inventer une réponse absente de la documentation | Seuil de pertinence du retrieval ; sinon « information insuffisante » + escalade |
| Répondre à des sujets hors périmètre (conseils d'achat, météo, juridique…) | Règles du prompt système + refus poli |
| Demander un mot de passe ou un numéro de carte | Règle du prompt + détection et masquage dans les entrées |
| Obéir à des instructions cachées dans la question ou dans un document | Détection d'injection de prompt ; le contenu des outils est traité comme une donnée |

## 3. Les trois outils

### 3.1 `search_docs` — recherche documentaire (RAG)
- **Entrée** : `query` (texte, 3 à 300 caractères).
- **Sortie** : liste de passages avec `doc_id`, section, version, score et texte.
- **Erreurs** : argument invalide ; aucun passage au-dessus du seuil → `information_insuffisante`.
- **Permission** : lecture seule, corpus public.

### 3.2 `get_order_status` — consultation de commande
- **Entrée** : `order_id` au format `CMD-12345` (validé par expression régulière).
- **Sortie** : statut, articles, transporteur, numéro de suivi, date de livraison estimée.
- **Erreurs** : format invalide ; client non identifié ; commande introuvable (même message si
  elle appartient à un autre client, pour ne rien révéler) ; délai dépassé (panne simulée).
- **Permission** : lecture seule, limitée aux commandes du client connecté.

### 3.3 `create_ticket` — escalade vers un humain
- **Entrées** : `category` (retard_livraison, colis_endommage, garantie, remboursement, litige,
  autre), `summary` (10 à 500 caractères), `order_id` (facultatif), `priority`.
- **Sortie** : un brouillon `DRAFT-xxxx` en attente de confirmation.
- **Erreurs** : catégorie invalide, résumé trop court, commande d'un autre client.
- **Permission** : écriture **soumise à confirmation humaine**. Le LLM n'a pas accès à l'action
  « confirmer » : seul le client la déclenche depuis l'interface.

## 4. Données et droits

- **Corpus documentaire** : 4 documents Markdown rédigés pour le projet (politique de retours,
  guide de livraison, garantie, FAQ), plus un document volontairement obsolète pour tester le
  filtrage. Chaque document porte des métadonnées : `doc_id`, version, date, type, statut, sensibilité.
- **Commandes** : un fichier JSON de commandes et de clients **entièrement fictifs**
  (adresses `@example.ma`).
- **Droits** : toutes les données sont créées pour ce projet. Il n'y a ni donnée réelle ni
  donnée personnelle, donc aucune contrainte de droits ni de RGPD / loi 09-08 sur les données d'entrée.

## 5. Évaluation

Un jeu d'environ 30 scénarios annotés (outil attendu, résultat attendu) :
nominaux, erreurs, hors périmètre, ambigus, tentatives d'injection, actions risquées.

| Métrique | Définition |
|---|---|
| Taux de réussite des tâches | % de scénarios où le résultat attendu est obtenu |
| Précision de sélection d'outil | % de scénarios où l'outil appelé est le bon (ou aucun outil quand il ne faut pas en appeler) |
| Taux d'appels invalides ou échoués | appels rejetés par la validation ou en erreur / total des appels |
| Latence moyenne et P95 | mesurée de la question à la réponse |
| Nombre moyen d'appels d'outil par tâche | efficacité de l'agent |
| Taux de refus hors périmètre | % de demandes hors périmètre correctement refusées |
| Taux de blocage des injections | % de tentatives d'injection bloquées |
| Actions non sûres | tickets créés sans confirmation, fuites entre clients (doit être 0) |
| Recall@k et MRR du retrieval | qualité de l'outil `search_docs` sur des questions annotées |
| Coût estimé par requête | tokens × tarif du modèle |

Les seuils des quality gates seront justifiés à partir d'une baseline et d'exigences métier.

## 6. Ce qu'on versionne

| Élément | Outil |
|---|---|
| Code, tests, workflow CI | Git / GitHub |
| Prompt système (v1, v2) | Git (fichier Python versionné) |
| Configuration (modèle, température, top_k, seuils, limites) | Git (`configs/config.yaml`) |
| Corpus et commandes fictives | DVC |
| Index du RAG | DVC (sortie du pipeline) |
| Expériences : paramètres, métriques, artefacts | MLflow |
| Version de l'agent | empreinte (hash) du prompt, de la config, des outils et du corpus, enregistrée dans MLflow et dans chaque trace |

## 7. Risques principaux et mesures

| Risque | Mesure |
|---|---|
| Réponse inventée (hallucination) | Réponse uniquement à partir des passages, citation obligatoire, seuil de score, évaluation de la fidélité |
| Injection de prompt (directe ou cachée dans un document) | Détection de motifs, nettoyage des passages, prompt qui traite les sorties d'outils comme des données |
| Fuite de données personnelles | Contrôle du propriétaire de la commande, masquage des e-mails et téléphones dans les traces |
| Action non autorisée | Pas d'outil de remboursement, ticket soumis à confirmation, permissions minimales |
| Boucle infinie ou coût excessif | Limite d'itérations et d'appels d'outil, délai d'attente par outil |
| Panne du LLM ou d'un outil | Nouvelles tentatives, message de repli, proposition d'escalade |
| Documents obsolètes ou contradictoires | Métadonnée `statut`, seuls les documents actifs sont indexés |
| Dépendance à un fournisseur externe (Groq) | Modèle configurable, LLM simulé pour la CI |