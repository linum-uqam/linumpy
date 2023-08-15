# -*- coding: utf-8 -*-

import os
import argparse
import SimpleITK as sitk

def tiff_folder_to_nifti(input_folder, output_folder):
    # Vérifier si le dossier de sortie existe, sinon le créer
    os.makedirs(output_folder, exist_ok=True)

    # Liste des fichiers TIFF dans le dossier d'entrée
    tiff_files = [file for file in os.listdir(input_folder) if file.lower().endswith('.tiff')]

    for tiff_file in tiff_files:
        input_tiff_path = os.path.join(input_folder, tiff_file)
        output_nifti_file = os.path.splitext(tiff_file)[0] + ".nii.gz"
        output_nifti_path = os.path.join(output_folder, output_nifti_file)

        # Charger l'image TIFF en utilisant SimpleITK
        image = sitk.ReadImage(input_tiff_path)

        # Sauvegarder l'image en format NIfTI
        sitk.WriteImage(image, output_nifti_path)

def main():
    parser = argparse.ArgumentParser(description="Convertit les images TIFF en fichiers NIfTI (.nii.gz)")
    parser.add_argument("input_folder", help="Chemin vers le dossier contenant les images TIFF")
    parser.add_argument("output_folder", help="Chemin vers le dossier de sortie pour les fichiers NIfTI")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

     # Vérifier si le dossier de sortie existe, sinon le créer
    os.makedirs(output_folder, exist_ok=True)

    # Liste des fichiers TIFF dans le dossier d'entrée
    tiff_files = [file for file in os.listdir(input_folder) if file.lower().endswith('.tiff')]

    for tiff_file in tiff_files:
        input_tiff_path = os.path.join(input_folder, tiff_file)
        output_nifti_file = os.path.splitext(tiff_file)[0] + ".nii.gz"
        output_nifti_path = os.path.join(output_folder, output_nifti_file)

        # Charger l'image TIFF en utilisant SimpleITK
        image = sitk.ReadImage(input_tiff_path)

        # Sauvegarder l'image en format NIfTI
        sitk.WriteImage(image, output_nifti_path)

if __name__ == "__main__":
    main()

