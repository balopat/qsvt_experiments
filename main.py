from typing import List, Generator, Iterator, Iterable

import cirq
import numpy as np
from matplotlib import pyplot as plt




from pyqsp.angle_sequence import QuantumSignalProcessingPhases
ang_seq = QuantumSignalProcessingPhases([10, 20], signal_operator="Wz")
print(ang_seq)