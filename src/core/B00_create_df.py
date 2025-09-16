# src/core/excel_tools.py
import re
from collections import defaultdict
from functools import reduce

def detecter_gabarits_colonnes(dictionnaire_df):
    all_colnames = set()
    for df in dictionnaire_df.values():
        all_colnames.update(df.columns)
    all_colnames = list(all_colnames)
    gabarits = defaultdict(list)
    for col in all_colnames:
        gabarit = re.sub(r'\d+', '{X}', col)
        gabarits[gabarit].append(col)
    return dict(gabarits)



def extraire_et_fusionner_par_multi_gabarits_interactif(dictionnaire_df, groupes_predefinis=None):


    gabarits = detecter_gabarits_colonnes(dictionnaire_df)
    gabarits_tries = sorted(gabarits.items(), key=lambda x: x[0])

    print("Gabarits détectés (triés alphabétiquement) :")
    for i, (gabarit, colonnes) in enumerate(gabarits_tries):
        print(f"{i+1}: {gabarit} -> {len(colonnes)} colonnes")

    regroupements = {}

    # --- Mode non interactif si groupes_predefinis est fourni ---
    if groupes_predefinis is not None:
        listes_indices = []
        # Si on donne des indices (ex: [[1,3],[2,4,5]]), sinon on accepte aussi des noms de gabarits
        if isinstance(groupes_predefinis[0], int):
            listes_indices = [groupes_predefinis]
        elif isinstance(groupes_predefinis[0], (list, tuple)):
            listes_indices = groupes_predefinis
        else:
            raise ValueError("groupes_predefinis doit être une liste de listes d'indices de gabarits ou de noms")

        for indices in listes_indices:
            indices = sorted(set(indices))
            gabarits_choisis = [gabarits_tries[i][0] for i in indices]
            nom_combo = "_AND_".join(gabarits_choisis)
            colonnes_a_fusionner = []
            for g in gabarits_choisis:
                colonnes_a_fusionner += gabarits[g]
            dfs_a_fusionner = []
            for nom_feuille, df in dictionnaire_df.items():
                colonnes = [col for col in colonnes_a_fusionner if col in df.columns]
                if colonnes:
                    sous_df = df[colonnes].copy()
                    sous_df = sous_df.add_prefix(f"{nom_feuille}_")
                    dfs_a_fusionner.append(sous_df)
            if dfs_a_fusionner:
                df_merged = reduce(lambda left, right: left.join(right, how='inner'), dfs_a_fusionner)
                df_merged = df_merged[sorted(df_merged.columns)]
                regroupements[nom_combo] = df_merged
                print(f"Fusion réussie pour la combinaison '{nom_combo}' ({df_merged.shape[0]} lignes, colonnes : {df_merged.columns.tolist()})")
            else:
                regroupements[nom_combo] = None
                print(f"Aucune colonne trouvée pour la combinaison '{nom_combo}'.")
        return regroupements

    # --- Mode interactif (ancien fonctionnement) ---
    print("\nPour chaque fusion, entre une combinaison de numéros (ex : 1,24 ou 2-5), ou laisse vide pour finir.")
    while True:
        saisie = input("Entrez le(s) numéro(s) de gabarit(s) à fusionner ensemble : ").strip()
        if not saisie:
            break

        indices = set()
        morceaux = re.split(r'[,\s]+', saisie)
        for morceau in morceaux:
            if re.match(r'^\d+$', morceau):
                idx = int(morceau) - 1
                if 0 <= idx < len(gabarits_tries):
                    indices.add(idx)
            elif re.match(r'^\d+\s*-\s*\d+$', morceau):  # ex : 2-5
                deb, fin = re.split(r'-', morceau)
                deb, fin = int(deb.strip()) - 1, int(fin.strip()) - 1
                for idx in range(min(deb, fin), max(deb, fin) + 1):
                    if 0 <= idx < len(gabarits_tries):
                        indices.add(idx)

        if not indices:
            print("Aucun numéro valide.")
            continue

        indices = sorted(indices)
        gabarits_choisis = [gabarits_tries[i][0] for i in indices]
        nom_combo = "_AND_".join(gabarits_choisis)
        colonnes_a_fusionner = []
        for g in gabarits_choisis:
            colonnes_a_fusionner += gabarits[g]
        dfs_a_fusionner = []
        for nom_feuille, df in dictionnaire_df.items():
            colonnes = [col for col in colonnes_a_fusionner if col in df.columns]
            if colonnes:
                sous_df = df[colonnes].copy()
                sous_df = sous_df.add_prefix(f"{nom_feuille}_")
                dfs_a_fusionner.append(sous_df)
        if dfs_a_fusionner:
            df_merged = reduce(lambda left, right: left.join(right, how='inner'), dfs_a_fusionner)
            df_merged = df_merged[sorted(df_merged.columns)]
            regroupements[nom_combo] = df_merged
            print(f"Fusion réussie pour la combinaison '{nom_combo}' ({df_merged.shape[0]} lignes, colonnes : {df_merged.columns.tolist()})")
        else:
            regroupements[nom_combo] = None
            print(f"Aucune colonne trouvée pour la combinaison '{nom_combo}'.")
    return regroupements


def renommer_dataframes(df_dict, mapping_renommage=None, return_mapping=False):
    """
    Renomme les DataFrames (clés du dictionnaire).
    - mapping_renommage (dict) : mapping automatique {ancien_nom: nouveau_nom, ...}
    - return_mapping (bool) : si True, retourne aussi le mapping utilisé.
    - Sinon, mode interactif (console) : propose un renommage pour chaque DataFrame.
    """
    nouveaux_noms = {}
    new_df_dict = {}

    noms_liste = list(df_dict.keys())
    if mapping_renommage is not None:
        for old_name, df in df_dict.items():
            new_name = mapping_renommage.get(old_name, old_name)
            new_df_dict[new_name] = df
            nouveaux_noms[old_name] = new_name
    else:
        for i, old_name in enumerate(noms_liste):
            new_name = input(
                f"Nouveau nom pour df indice {i} ({old_name}) : "
            ).strip() or old_name
            new_df_dict[new_name] = df_dict[old_name]
            nouveaux_noms[old_name] = new_name

    return (new_df_dict, nouveaux_noms) if return_mapping else new_df_dict




def renommer_colonnes_df_dict(df_dict, mapping_renommage_colonnes=None, return_mapping=False):
    """
    Renomme les colonnes de chaque DataFrame d'un dictionnaire.
    Trie aussi les colonnes par ordre alphabétique.
    - mapping_renommage_colonnes : dict optionnel
        { nom_df: { ancien_nom_colonne: nouveau_nom_colonne, ... }, ... }
    - return_mapping : si True, retourne aussi le mapping utilisé.
    """
    mapping_utilise = {}
    capteur_pattern = re.compile(r'([CDE]\d+)')  # exemple : C4, D7, E8

    for nom_df, df in df_dict.items():
        mapping_cols = {}
        if mapping_renommage_colonnes and nom_df in mapping_renommage_colonnes:
            mapping_cols = mapping_renommage_colonnes[nom_df]
            df.rename(columns=mapping_cols, inplace=True)
        else:
            for col in df.columns:
                suggestion = None
                if "temperature" in col.lower() or "température" in col.lower():
                    match = capteur_pattern.search(col)
                    capteur = match.group(1) if match else "?"
                    suggestion = f"température {capteur}"
                else:
                    suggestion = col.split("-")[-1].strip()
                # si pas d'interaction (console), on garde juste la suggestion
                mapping_cols[col] = suggestion if suggestion else col
            df.rename(columns=mapping_cols, inplace=True)

        df.sort_index(axis=1, inplace=True)  # trie les colonnes
        mapping_utilise[nom_df] = mapping_cols

    return (df_dict, mapping_utilise) if return_mapping else df_dict


def filtrer_colonnes_par_numero(df_dict, selections_predefinies=None):
    """
    Supprime des colonnes par indices (par df) ; conserve le comportement console si selections_predefinies est None.
    """
    for nom_df, df in df_dict.items():
        print(f"\n--- DataFrame : {nom_df} ---")
        for i, col in enumerate(df.columns):
            print(f"{i}: {col}")

        if selections_predefinies and nom_df in selections_predefinies:
            selections = selections_predefinies[nom_df]
            print(f"Sélection prédéfinie pour {nom_df} : {selections}")
        else:
            selections = input(
                "Entrez les numéros ou plages de colonnes à supprimer (ex: 3,5-8,12-14). Laisse vide pour tout garder : "
            ).strip()

        if not selections:
            print("Aucune colonne supprimée.")
            continue

        indices_a_supprimer = set()
        if isinstance(selections, (list, tuple, set)):
            indices_a_supprimer = set(selections)
        else:
            for partie in str(selections).split(','):
                partie = partie.strip()
                if not partie:
                    continue
                if '-' in partie:
                    debut, fin = partie.split('-')
                    indices_a_supprimer.update(range(int(debut), int(fin) + 1))
                else:
                    indices_a_supprimer.add(int(partie))

        colonnes_a_drop = [df.columns[i] for i in sorted(indices_a_supprimer) if 0 <= i < len(df.columns)]
        print(f"→ Colonnes supprimées dans {nom_df} : {colonnes_a_drop}")
        df.drop(columns=colonnes_a_drop, inplace=True)

    return df_dict




def _pat_to_regex(pat: str) -> re.Pattern:
    q = re.escape(pat).replace(r"\{X\}", r"(\d+)")
    return re.compile(rf"^{q}$")



def expand_df_name_mapping(pattern_map: dict[str, str], actual_names: list[str]) -> dict[str, str]:
    """pattern_map: {"C{X} ... DepX": "C{X}.DepX"} -> applique aux noms réels."""
    out = {}
    compiled = [( _pat_to_regex(p), p, tgt) for p, tgt in pattern_map.items()]
    for name in actual_names:
        for rx, p, tgt in compiled:
            m = rx.match(name)
            if m:
                x = m.group(1)
                out[name] = tgt.replace("{X}", x)
                break
    return out



def expand_cols_mapping(pattern_map_nested: dict[str, dict[str, str]], df_name: str, actual_cols: list[str]) -> dict[str, str]:
    # trouve le bloc pour ce DF (clé avec {X})
    chosen = None
    xval = None
    for df_pat, colmap in pattern_map_nested.items():
        rx = _pat_to_regex(df_pat)
        m = rx.match(df_name)
        if m:
            chosen = colmap
            xval = m.group(1)
            break
    if not chosen:
        return {}

    out = {}
    compiled = [(_pat_to_regex(p), p, tgt) for p, tgt in chosen.items()]
    for col in actual_cols:
        # essai 1: match sur le nom complet
        candidates = [col]
        # essai 2: match sur la "queue" après le 1er underscore (préfixe DF retiré)
        if "_" in col:
            candidates.append(col.split("_", 1)[1])

        matched = False
        for cand in candidates:
            for rx, p, tgt in compiled:
                m = rx.match(cand)
                if m:
                    x = xval or m.group(1)
                    out[col] = tgt.replace("{X}", x)
                    matched = True
                    break
            if matched:
                break
    return out
