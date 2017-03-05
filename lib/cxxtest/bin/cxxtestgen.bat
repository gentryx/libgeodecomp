@echo off
rem Just run the python script

echo "ANDI1 %PYTHON%"
dir "%PYTHON%"
echo "ANDI2"
echo "ANDI3" %0 %*

IF "%PYTHON%" == "" GOTO DEFAULT_PYTHON
:PYTHON_OVERRIDE
"%PYTHON%" %0 %*
GOTO END
:DEFAULT_PYTHON
python %0 %*
GOTO END
:END
