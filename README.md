The idea of this project is based on the differences found when characterizing microorganisms by MALDI-TOF mass spectrometry between different environments. 
A microorganism can be characterized by MALDI-TOF mass spectrometry. Thus, for each mass spectrum analyzed, a peak associated with the time of flight is obtained (check references). 
In this way, a model can be trained to, given these characteristics (represented as a one-dimensional vector), be able to predict whether that particular microorganism is going to be resistant or vulnerable to an antibiotic.

The differences found between two MALDI-TOFs of the same kind of microbe can be due to several reasons, such as specific qualities of strains according to their origin, the process and setting of the laboratory material to generate the MALDI-TOF, etc.
The objective of this project is to investigate whether employing Adversarial Training allows the models to be less sensitive to small variations in the dimensions of the instances (associated with changes in the environment). 
In this way, it is intended to improve their robustness and with the objective that models trained with datasets from a specific environment can be applied in other environments offering good performance.

In this project, experiments will be performed using a dataset widely known in the field that has a wide variety of microorganisms and instances (MALDI-TOF of each of them).
In particular, efforts will focus on one bacterium in particular, Klebsiella Pneumoniae.
To evaluate the generalization capacity and robustness of the models, instances gathered from two Spanish hospitals located in Madrid (Hospital Gregorio Marañón and Hospital Ramón y Cajal) will be used as the test set.
