# About this repo

All data required for this github project can be downloaded by using the provided DaterLoader script. <br>
Simply calling: <br>
```
DataLoader().download_datasets(root_dir)
DataLoader().download_trained_models(root_dir)
DataLoader().generateAllJsonDataAnnotations(root_dir)
``` 
Will download the datasets from the OMR-Datasets repository, <br>
download all the trained models as specified in the overview and <br>
generate the json annotations for the datasets.

## Faster R-CNN and ResNet-50 backbone
|   Model Name        |   Iterations  |    mAP   |    AP75  |    AP50  |
|:-------------------:|:-------------:|:--------:|:--------:|:--------:|
|   System measures   |     5700      |  95.578  |  98.952  |  98.970  |
|   Stave measures    |     9000      |  87.510  |  96.744  |  98.020  |
|      Staves         |     5700      |  93.173  |  100.00  |  100.00  |
|      Combined       |     TODO      |  TODO  |  TODO  |  TODO  |

## scores for the combined dataset per category
| Category        | mAP    | Category       | mAP    | Category   | mAP    |
|:---------------:|:------:|:--------------:|:------:|:----------:|:------:|
| system_measures | TODO | stave_measures | TODO | staves     | TODO |

## Faster R-CNN and ResNet-101 backbone
|   Model Name        |   Iterations  |    mAP   |    AP75  |    AP50  |
|:-------------------:|:-------------:|:--------:|:--------:|:--------:|
|   System measures   |     8700      |  96.401  |  98.864  |  98.909  |
|   Stave measures    |     6300      |  87.476  |  96.823  |  98.020  |
|      Staves         |     15600     |  94.293  |  100.00  |  100.00  |
|      Combined       |     TODO      |  TODO  |  TODO  |  TODO  |

## scores for the combined dataset per category
| Category        | mAP    | Category       | mAP    | Category   | mAP    |
|:---------------:|:------:|:--------------:|:------:|:----------:|:------:|
| system_measures | TODO | stave_measures | TODO | staves     | TODO |