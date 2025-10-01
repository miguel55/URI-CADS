# URI-CADS

## Description
This repository contains the code and documentation to implement URI-CADS (Ultrasound Renal Imaging Computer Aided Diagnosis System), a benchmark for US kidney pathology detection. 

```
URI-CADS: a fully automated computer-aided diagnosis system for ultrasound renal imaging,
Miguel Molina-Moreno, Iván González-Díaz, Maite Rivera Gorrín, Víctor Burguera Vion, Fernando Díaz-de-María
Journal of Imaging Informatics in Medicine 37:1458–1474, doi: https://doi.org/10.1007/s10278-024-01055-4, 2024. 
```

This code is partly based on the Pytorch implementations of Faster R-CNN and Mask R-CNN and developed in Pytorch.

## License

URI-CADS code is released under the CC BY-NC 4.0 License (refer to the `LICENSE` file for details).

## Citing URI-CADS

If you find URI-CADS useful in your research, please consider citing:

	@ARTICLE{uricads,
		title = {URI-CADS: a fully automated computer-aided diagnosis system for ultrasound renal imaging},
		journal = {Journal of Imaging Informatics in Medicine},
		volume = {37},
		pages = {1458–1474},
		year = {2024},
		doi = {https://doi.org/10.1007/s10278-024-01055-4},
		author = {Miguel Molina-Moreno and Iván González-Díaz and Maite Rivera Gorrín and Víctor Burguera Vion and Fernando Díaz-de-María}
	}
  
## Dataset and models

Download the models and database, available under request in [https://doi.org/10.21950/0E0CTF](https://doi.org/10.21950/0E0CTF), and uncompress them in models/ and bbdd/ directories.

## Requirements

URI-CADS is implemented to not require any additional modules. The Python code has been tested with Pytorch 1.3.1, a modified version of the torchvision library released with the code and CUDA 10.1.

Before executing the Python code, it is necessary to uncompress the torchvision_05.zip file.

## Demo

To test our approach with the provided database and models, follow the steps below. 

1. First, execute the `detect_inference.py` script. It applies the trained model for each training fold to its corresponding test fold, providing: a) the kidney and lesion segmentations masks in the `results_test_kidney` and `results_test_lesions` directories, b) the results for kidney and lesion detection (locations, scores and ground-truth),c)  final score matrix for the different classes (healthy, cyst, pyramid, hydronephrosis, others, poor corticomedullary differentation and hyperecogenia), d) ground-truth matrix for evaluation, and e) alpha parameter values for interpretation.
2. Second, the `evaluate.py` script provides the area under the Sensitivity-Specificity curve for each one of the classes (AUC_{SP-SENS}) and the SP_{SENS95} value (specificity at a sensibility of 95 %).

The reported performance for each class (global and local pathologies) with our benchmark is presented below:

| Class (pathology)                      |  AUC_{SP-SENS} (%)  |   SP_{SENS95} (%)  |
|----------------------------------------|---------------------|--------------------|
| Healthy                                |        87.41        |        60.59       |
| Cyst*                                  |        79.59        |        28.19       |
| Pyramid*                               |        86.61        |        48.19       |
| Hydronephrosis*                        |        93.04        |        63.19       |
| Others*                                |        69.32        |        14.05       |
| Poor corticomedullary differentiation* |        78.65        |        29.28       |
| Hyperecogenia*                         |        84.15        |        43.63       |
| Multi-pathological (average*)          |        81.90        |        37.76       |

## Installation

To start using URI-CADS, download and unzip this repository.
```
git clone https://github.com/miguel55/URI-CADS
```

## More info

See `doc\doc.tex` for more details.
