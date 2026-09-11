FUZZY PREFETCHER SELECTION
==========================

DATASETS
--------
The dataset_calib and dataset_testes datasets are available in the
Releases section of this repository.


DIRECTORY STRUCTURE
--------------------
The script must be executed from within the folder that contains both the
dataset_calib and dataset_testes directories.

- Calibration of the Fuzzy Prefetcher Selection system is performed using
  the files in the dataset_calib directory.
- Testing (evaluation) is performed using the files in the dataset_testes
  directory.


SELECTING TRACES
-----------------
To select which trace(s) will be processed by the Fuzzy Prefetcher
Selection, uncomment the corresponding line(s) of the TRACES_SELECIONADOS
variable:

    TRACES_SELECIONADOS = [
    #       TRACES TESTE
            '602.gcc_s-1850B.champsimtrace.xz',
    #       '603.bwaves_s-2931B.champsimtrace.xz',
    #       '605.mcf_s-994B.champsimtrace.xz',
    #       '619.lbm_s-4268B.champsimtrace.xz',
    #       '620.omnetpp_s-874B.champsimtrace.xz',
    #       '623.xalancbmk_s-592B.champsimtrace.xz',
    #       '649.fotonik3d_s-7084B.champsimtrace.xz',

    ]
