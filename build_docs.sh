cd docs
cp ../README.md ./source/
# sphinx-apidoc -MEf -o ./source ../MultiHMCGibbs ../MultiHMCGibbs/tests
make clean
make html
cd ..
