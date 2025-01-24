# Detection of Ascaris using CNN

## Description
This repository contains the code to assistance in parasitological diagnosis, offering an efficient and accurate solution for parasite detection through microscopic image analysis.
Author of the related research are:

- Giovanni Gelber Martinez Pastor
- Cesar Roberto Ancco Ruelas
- Eveling Castro-Gutierrez
- Victor Luis Vásquez Huerta

## Research Article
[Automatic Detection of Ascaris Lumbricoides in Microscopic Images using Convolutional Neural Networks (CNN)](https://thesai.org/Publications/ViewPaper?Volume=15&Issue=5&Code=IJACSA&SerialNo=90#:~:text=10.14569/IJACSA.2024.0150590)

## Local

1. Define your dataset and change its source in `reduccion.py `

```python
  carpeta = f'{current_dir}/DATASET-UNCINARIAS'
```

2. Run `setup_directories` with the name of a new folder `<MY-TEST-FOLDER>`. This folder will be created in `__tests` and contain an structure to save the data generated in each step.

```python
./setup_directories.py <MY-TEST-FOLDER>
```

3. Finally run next command. By default, the generated model will be in `__tests/<MY-TEST-FOLDER>/GENERATED_MODELS`

```python
python3 run_all.py ./__tests/<MY-TEST-FOLDER>
```
