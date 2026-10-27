"""Utilitaires réutilisables du pipeline (préparation des données, validation, I/O images).

Modules :
    preprocess        — DICOM/PNG → crop → resize, construction des .pkl GMIC
    validate_input    — validation d'un jeu d'images/CSV avant inférence
    test_validate_input — tests unitaires de validate_input (pytest)
    load_images       — chargement flexible d'images PNG (collect_images / load_image / load_all)
    preresize_images  — pré-resize sur disque des crops vers une résolution cible

Les fonctions sont importables, ex :
    from utils.load_images import load_all
    from utils.validate_input import check_image
"""
