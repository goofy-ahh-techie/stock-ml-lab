Some basic commands to run on doing the basic setup for running the application, as we are using poetry for env. 

1. python --version  
2. pip --version
3. pip install poetry

4. poetry config virtualenvs.in-project true
5. poetry init 
6. poetry env use python
7. poetry add pandas numpy scikit-learn matplotlib yfinance pyarrow
8. poetry add --group dev jupyter ipykernal ruff black pytest

---- Now, go to puproject.toml, add this at bottom(we don't wanna use poetry for packages) : 
[tool.poetry]
package-mode = false

9. poetry install