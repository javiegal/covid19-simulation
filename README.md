# covid19-simulation

## Description

Project developed for the Data Processes course of the Master's Programme in Data Science of UPM.

This repository contains all the work done for the Data Processes assignment described in the [assignment guide](./docs/assignment_guide.pdf).

## Deliverables
- [Project plan](./docs/project_plan.pdf).
- [Technical report](./docs/technical_report.pdf).
- [Python script](./__main__.py) with the data analysis.
- [KNIME package](./models/knime-export.knar) with the prediction model.

## Requirements
The data analysis has been developed using the following software:
- Python 3.8.10
- Pandas 1.3.3
- Scikit-learn 1.0
- matplotlib 3.4.3
- Seaborn 0.11.2
- Lifelines 0.26.4

Prediction models have been developed using the following software:
- KNIME 4.5.0


## Usage
You can reproduce the data analysis performed by executing the following command:
```
$ python covid19-simulation.zip
```

Otherwise, you can extract the folder and execute the command:
```
$ python EXTRACTED_FOLDER_PATH
```
This will produce the console output, images (in `covid19-simulation/imgs` folder) and processed data (in `covid19-simulation/data` folder) explained in the [technical report](./docs/technical_report.pdf).

## Authors
- María Ayuso Luengo ([@mariaayuso](https://github.com/mariaayuso))
- Doga Cengiz ([@dogacengiz](https://github.com/dogacengiz))
- Javier Gallego Gutiérrez ([@javiegal](https://github.com/javiegal))
- Pablo Hernández Carrascosa ([@pablohdez98](https://github.com/pablohdez98))
