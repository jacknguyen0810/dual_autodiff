@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
    set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

REM Install requirements first
python -m pip install -r requirements.txt
where pandoc >nul 2>&1
if errorlevel 1 (
    echo Installing pandoc...
    conda install -c conda-forge pandoc -y
)

if "%1" == "" goto help
if "%1" == "clean" (
    rmdir /s /q %BUILDDIR%
    goto end
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
