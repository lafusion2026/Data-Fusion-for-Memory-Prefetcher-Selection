The dataset_calib and dataset_testes are available in the releases of this repository.

To select the TRACE to be processed by the Fuzzy Prefetcher Selection, uncomment the lines for the variable:

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

The calibration of the Fuzzy Prefetcher Selection is performed using files from the `dataset_calib` directory, and testing is conducted using files from the `dataset_testes` directory.
The script must be executed from within the folder containing these two directories.
