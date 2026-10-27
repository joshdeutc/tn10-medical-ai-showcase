# 🏥 Medical AI & Statistical Risk Control — R&D Showcase (Epiconcept TN10)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Quarto](https://img.shields.io/badge/Quarto-Technical_Manuals-39729E?logo=quarto&logoColor=white)](https://quarto.org/)
[![Docker](https://img.shields.io/badge/Docker-Reproducible_Capsules-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

> **FR :** Dépôt vitrine regroupant le code source méthodologique, les algorithmes de contrôle statistique du risque et les manuels techniques Quarto réalisés lors de mon stage de recherche R&D de fin d'études (6 mois, **Epiconcept**, Paris).  
> **Confidentialité & RGPD :** Conformément aux règles d'accès PhysioNet (MIMIC-IV) et RSNA, **aucune donnée patient, note clinique ni poids médical privé n'est inclus dans ce dépôt** — seuls les scripts d'ingénierie, les algorithmes mathématiques et la documentation technique reproductible sont publiés.
>
> **EN:** Public engineering and research showcase of my 6-month final-year R&D internship at **Epiconcept** (Paris, France), spanning two high-stakes clinical AI workstreams:
> 1. **High-Resolution Mammography Screening (`GMIC`) & Distribution-Free Selective Classification (`SGR` / Hoeffding Risk Bounds)**
> 2. **Grounded Clinical Information Extraction & Cross-Lingual Transfer (`CATH-LLM` for ECDC European Surveillance)**

---

## 🔬 Workstream 1 — Interpretability & Guaranteed Risk Control in Mammography (`part1_mammography_gmic_sgr/`)

### 1. Architecture GMIC (*Globally-Aware Multiple Instance Classifier*)
Fine-tuning et évaluation sur **> 300 Go de mammographies haute résolution** (`2944 × 1920`, 16 bits, dataset RSNA Screening Mammography, 54 706 images / 11 913 patientes, prévalence maligne `2,12 %`) :
- **Réseau Global (`ResNet-22`) :** Extraction d'une carte de saillance basse résolution $A \in [0,1]^{46 \times 30}$ et agrégation par *Top-$t\%$ pooling*.
- **Sélection Gloutonne de Patchs (*Retrieve ROI*) :** Extraction différentiable des $K=6$ régions d'intérêt haute résolution (`256 × 256`) sans chevauchement.
- **Réseau Local (`ResNet-18`) + Gated Attention :** Pondération par attention $\alpha_k$ des 6 patchs critiques et fusion globale-locale ($\hat{y}_{\text{fusion}}$).
- **Performance obtenue :** Passage de `AUC = 0,683` (SmallCNN baseline) et `AUC = 0,762` (GMIC pré-entraîné NYU zero-shot) à **`AUC = 0,889` (`IC 95 % : [0,858 ; 0,921]`)** par *full fine-tuning* en précision mixte 16 bits avec augmentations GPU et *Deep Ensemble* de 5 modèles (`nyu-m1..m5`).

### 2. Classification Sélective à Risque Garanti (Cadre SGR & Bornes de Hoeffding)
En dépistage médical, imposer `100 %` de couverture conduit à des erreurs inacceptables sur les cas ambigus. Implémentation des algorithmes de **Geifman & El-Yaniv (2017)** et extension **bi-seuil $(\tau_-, \tau_+)$** pour contrôler simultanément deux risques conditionnels asymétriques avec garantie en échantillon fini $1 - \delta$ via l'inversion de la borne de **Hoeffding** :
$$\hat{R}(g_\tau \mid S_{\text{cal}}) + \sqrt{\frac{\ln(1/\delta)}{2 \, m(\tau)}} \le r^*$$
- **Contrôle simultané :** $\text{FNR} \le 5\%$ sur les cas prédits bénins (zone verte) et $\text{FPR} \le 5\%$ sur les cas orientés en biopsie (zone rouge), avec abstention explicite dirigée vers une seconde lecture radiologique humaine.
- **Incertitude :** Comparaison *Softmax Response* vs entropie prédictive vs variance épistémique (**MC Dropout** `p = 0.3 / 0.5` et **Deep Ensembles**).

---

## 🧠 Workstream 2 — Grounded Clinical LLM Extraction & Zero-Shot Cross-Lingual Transfer (`part2_clinical_llm_extraction/`)

Développement d'un pipeline d'extraction d'infections associées aux soins (surveillance européenne **ECDC** des cathéters veineux centraux) sur notes cliniques **MIMIC-IV** sous contrainte matérielle stricte (**1× GPU NVIDIA A100 40 Go VRAM**, exécution 100 % locale pour conformité RGPD hospitalière) :

| Modèle Local / Quantization | Stratégie de Prompting | F1 Global (`infection_present`) | Conformité JSON & Grounding |
| :--- | :--- | :---: | :---: |
| **MedGemma-1.5-4B-IT** | Prompt structuré v2 (JSON + verbatim) | **0,91** | 100 % |
| **MedGemma-27B-Text-IT (`int8`)** | Prompt v2 + auto-vérification | **0,88** | 100 % |
| **Gemma-2-27B-IT (`int8`)** | Prompt v2 | 0,85 | 99,6 % |
| **Qwen-2.5-32B / Mistral-Small-24B** | Prompt v2 | 0,84 – 0,86 | 99,2 % |

### Innovations d'Ingénierie (`part2_clinical_llm_extraction/tools/` & `docs/`)
1. **Découpage Clinique Déterministe (`01-paragraphes.qmd`) :** Segmentation par expressions régulières des sections pertinentes (*Brief Hospital Course*, *Microbiology*, *Discharge Medications*), réduisant la fenêtre de contexte de `~65 %` sans perte de rappel.
2. **Protocole Anti-Hallucination (`05-verification.qmd`, `extraire.py`) :** Chaque décision clinique retournée par le LLM doit inclure une liste `supporting_evidence` contenant des citations exactes (*verbatim*). Un vérificateur déterministe (`verify_extractions.py`) normalise la casse/ponctuation et contrôle que chaque preuve est une sous-chaîne exacte du dossier patient (`evidence_is_grounded = True`).
3. **Transfert Cross-Lingue Zéro-Shot vers le Croate (`03-traducteurs.qmd`, `traduire_paragraphes.py`) :** Traduction automatisée par paragraphes et alignement de la vérité terrain via **NLLB-200-3.3B**, **TowerInstruct-7B** et **EuroLLM-9B**, démontrant le maintien d'un **`F1 = 0,88`** sur des comptes rendus cliniques en croate avec un prompt système resté en anglais.

---

## 📁 Structure du Dépôt

```text
tn10-medical-ai-showcase/
├── part1_mammography_gmic_sgr/
│   ├── _abstain_rc.py & _ens_riskcontrol.py   # Algorithmes SGR (Algo 1 & Algo 2 bi-seuil + Hoeffding)
│   ├── _gmic_bench.py & _build_ens_sgp.py     # Benchmark Deep Ensemble (5x GMIC NYU) & MC Dropout
│   ├── fine_tuning/                           # Scripts PyTorch d'entraînement GMIC 16-bit & ResNet
│   ├── scripts/ & utils/                      # Inférence, prétraitement DICOM/PNG 16-bit, validation
│   ├── docker/                                # Conteneurs reproductibles (inférence & code-server)
│   └── docs/script_notebook/                  # 9 notebooks Quarto (.qmd) documentant tout le pipeline
└── part2_clinical_llm_extraction/
    ├── tools/
    │   ├── extraire.py                        # Extraction clinique JSON contraint + vérification verbatim
    │   ├── traduire_paragraphes.py            # Pipeline de traduction clinique anglo-croate alignée
    │   └── multidoc.py                        # Agrégation multi-documents à l'échelle du séjour patient
    └── docs/
        ├── 01-paragraphes.qmd                 # Préparation & filtrage déterministe des notes MIMIC-IV
        ├── 02-redactions.qmd                  # Ingénierie des prompts cliniques & schémas JSON ECDC
        ├── 03-traducteurs.qmd                 # Benchmark de traduction médicale vers le croate
        ├── 04-extraction.qmd                  # Évaluation de 20 configurations LLMs sur A100 40 Go
        ├── 05-verification.qmd                # Contrôle déterministe anti-hallucination (grounding)
        └── a1-pieges.qmd                      # Retour d'expérience : pièges VRAM, biais et solutions
```
