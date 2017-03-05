@echo off
rem Just run the python script

echo "ANDI1 %PYTHON%"

IF "%PYTHON%" == "" GOTO DEFAULT_PYTHON
:PYTHON_OVERRIDE
"%PYTHON%" %0 %*
GOTO END
:DEFAULT_PYTHON
python %0 %*
GOTO END
:END
