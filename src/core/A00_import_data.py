import os
import pandas as pd

def fusionner_excels_multifeuilles(
    dossier_source="excel_files",
    fichier_sortie="Data/fusion.xlsx"
):
    """
    Fusionne toutes les feuilles de tous les fichiers Excel d'un dossier
    dans un seul fichier Excel (chaque feuille restant séparée).
    Supprime 'Date (UTC)' et renomme 'Date (Europe/Paris)' en 'Timestamp'.
    """
    os.makedirs(os.path.dirname(fichier_sortie), exist_ok=True)
    fichiers = [f for f in os.listdir(dossier_source) if f.endswith('.xlsx')]
    print("Fichiers trouvés :", fichiers)
    with pd.ExcelWriter(fichier_sortie, engine='openpyxl') as writer:
        for fichier in fichiers:
            chemin_fichier = os.path.join(dossier_source, fichier)
            xls = pd.ExcelFile(chemin_fichier, engine="openpyxl")
            print(f"Lecture de {fichier}, feuilles : {xls.sheet_names}")
            for nom_feuille in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=nom_feuille, engine="openpyxl")
                # Supprime la colonne "Date (UTC)" si elle existe
                if "Date (UTC)" in df.columns:
                    df = df.drop(columns=["Date (UTC)"])
                # Renomme la colonne "Date (Europe/Paris)" en "Timestamp" si elle existe
                if "Date (Europe/Paris)" in df.columns:
                    df = df.rename(columns={"Date (Europe/Paris)": "Timestamp"})
                nom_feuille_final = f"{os.path.splitext(fichier)[0]}_{nom_feuille}"[:31]
                print(f"  Ajout de la feuille : {nom_feuille_final} (lignes: {len(df)})")
                df.to_excel(writer, sheet_name=nom_feuille_final, index=False)
    print(f"Fusion terminée ! Résultat dans : {fichier_sortie}")


def charger_feuillets_excel(chemin_fichier):
    """
    Charge toutes les feuilles d’un fichier Excel en tant que dictionnaire de DataFrames.
    """
    return pd.read_excel(chemin_fichier, sheet_name=None, engine='openpyxl')


def convertir_colonne_timestamp(df, nom_colonne='Timestamp', format_datetime='%Y-%m-%d %H:%M'):
    """
    Convertit une colonne en datetime uniforme, sans timezone, et la définit comme index.
    """
    if nom_colonne in df.columns:
        df[nom_colonne] = pd.to_datetime(df[nom_colonne], errors='coerce', format=format_datetime)
        df[nom_colonne] = df[nom_colonne].dt.tz_localize(None)
        df[nom_colonne] = df[nom_colonne].dt.floor('min')
        df = df.sort_values(nom_colonne).set_index(nom_colonne)  # ✅ définit Timestamp comme index
    else:
        print(f"⛔ Colonne '{nom_colonne}' absente du DataFrame")
    return df


def nettoyer_toutes_feuilles(dictionnaire_df, nom_colonne='Timestamp', format_datetime='%Y-%m-%d %H:%M'):
    """
    Applique la conversion de Timestamp à tous les DataFrames d’un dictionnaire.
    """
    for nom_feuille in dictionnaire_df:
        dictionnaire_df[nom_feuille] = convertir_colonne_timestamp(
            dictionnaire_df[nom_feuille],
            nom_colonne,
            format_datetime
        )
    return dictionnaire_df
