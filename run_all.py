#!/usr/bin/env python3
import os
import json
import subprocess
import argparse
# from config import load_config

def main():
    parser = argparse.ArgumentParser(description="Ejecutar múltiples scripts en orden, usando una configuración JSON.")
    parser.add_argument(
        "config_dir",
        nargs="?",
        default=os.getcwd(),
        help="Ruta al directorio donde se encuentra el archivo config.json. Si no se especifica, se usa el directorio actual."
    )
    args = parser.parse_args()

    config_directory = os.path.abspath(args.config_dir)
    if not os.path.isdir(config_directory):
        raise NotADirectoryError(f"El directorio proporcionado no existe: {config_directory}")
    print('- config_directory', config_directory)
    config_path = os.path.join(config_directory, "config.json")
    print('- config_path', config_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"No se encontró el archivo config.json en el directorio: {config_directory}")
    print('- config_path ok', config_path)
    scripts = [
        "reduccion.py",
        "creacion_array.py",
        "mezcla_array.py",
        "ia.py"
    ]
    # load_config(config_path)
    for script in scripts:
        subprocess.run(["python3", script, config_path])

if __name__ == "__main__":
    main()
