# Exploring Quantum Singular Value Transformation

This is the code repository for a school project during my PhD training at Duke University for the class
 Introduction to Quantum Engineering by Prof Kenneth R. Brown. It will potentially become the repo for a
 write-up on my blog, I do not intend to make this code into a QSVT library. Feel free to use bits and
 pieces from it. If you do, please give credit!


See [the write-up](./balint_pato_exploring_qsvt_2021.pdf) for class for now.

# Setup

To setup dependencies, run:
```
pip install -r requirements.txt
```

# Entry points

- [bb1.py](./bb1.py): single qubit BB1 experiment reproduced from Martyn et al
- [qsp.py](./qsp.py): single qubit QSP experiments and code to convert between Wx and reflection QSP conventions
- [simple_qubitization.py](./qsp.py): two-qubit amplitude amplification experiments
- [plots/](./plots): the output for plots produced by the experiments. Lots of polynomials.




