# Models

The trained models have to be downloaded seperately, because they cannot be stored in git. <br>
Their size is around 450MB and can be downloaded from my google drive.

## Overview

|   Name              |   Iterations  |     Dataset     |   mAP   |   AP75  |   AP50  |    Download    |
|:-------------------:|:-------------:|:---------------:|:-------:|:-------:|:-------:|:--------------:|
|                     |               |    Validation   |  96.47  |  98.88  |  98.90  |                |
|   System measures   |     10500     |    Test         |  96.54  |  98.95  |  98.95  |  coming soon   |
|                     |               |    Entire       |  96.70  |  98.96  |  98.97  |                |
|                     |               |    Validation   |  86.09  |  96.59  |  97.95  |                |
|   Stave measures    |     10800     |    Test         |  86.95  |  96.77  |  98.02  |  coming soon   |
|                     |               |    Entire       |  85.13  |  94.77  |  96.02  |                |
|                     |               |    Validation   |  91.72  |  98.98  |  100.00 |                |
|   Staves            |     12300     |    Test         |  92.03  |  98.98  |  100.00 |  coming soon   |
|                     |               |    Entire       |  91.95  |  98.98  |  100.00 |                |
|                     |               |    Validation   |  76.11  |  83.16  |  83.76  |                |
|   Combined          |     14500     |    Test         |  76.48  |  83.65  |  83.81  |  coming soon   |
|                     |               |    Entire       |  76.22  |  82.95  |  83.48  |                |


## scores for the combined dataset per category

|    Dataset    | Category        | mAP    | Category       | mAP    | Category   | mAP    |
|:-------------:|:---------------:|:------:|:--------------:|:------:|:----------:|:------:|
|   Validation  | system_measures | 90.453 | stave_measures | 61.559 | staves     | 76.322 |
|   Test        | system_measures | 90.517 | stave_measures | 62.195 | staves     | 76.740 |
|   Entire      | system_measures | 90.681 | stave_measures | 62.041 | staves     | 75.944 |