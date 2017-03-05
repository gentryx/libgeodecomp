@echo off
rem Just run the python script

echo "ANDI1 %PYTHON%"
dir "%PYTHON%"
echo "ANDI2"
echo "ANDI3" %0 %*
dir
echo "ANDI4"

IF "%PYTHON%" == "" GOTO DEFAULT_PYTHON
:PYTHON_OVERRIDE
echo "ANDI5"
"%PYTHON%" %0 %*
echo "andi555 %ERRORLEVEL%"
GOTO END
:DEFAULT_PYTHON
echo "ANDI6"
python %0 %*
GOTO END
:END

echo "ANDI7"
exit /B 0

