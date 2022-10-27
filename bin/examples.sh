#!/bin/sh
echo "RUNNING EXAMPLE 1: h'=kh+(1-k)cos(t)-(1+k)sin(t), k=10, t=[0,1]"
python examples/test_equation.py

echo "RUNNING EXAMPLE 2(NFE): h'=-Lh, L=[1,-2,1], t=[0,1]"
python examples/discrete_laplacian.py

echo "RUNNING EXAMPLE 3(Convergence): h'=-Lh, L=[1,-2,1], t=[0,1]"
python examples/optimizer_convergence.py

echo "GENERATING EXAMPLE GRAPHICS"
python examples/plot_examples.py

echo "BASH TASK(S) COMPLETED."

read -p "$*"