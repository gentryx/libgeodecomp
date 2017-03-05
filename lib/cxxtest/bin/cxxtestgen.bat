@echo off
rem Just run the python script

echo "ANDI1 %PYTHON%"
dir C:\Python35
echo "ANDI2"
dir C:\
echo "ANDI3"

IF "%PYTHON%" == "" GOTO DEFAULT_PYTHON
:PYTHON_OVERRIDE
"%PYTHON%" %0 %*
GOTO END
:DEFAULT_PYTHON
python %0 %*
GOTO END
:END
