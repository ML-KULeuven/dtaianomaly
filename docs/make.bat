@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build
set SPHINXPROJ=dtaianomaly

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

if "%1" == "" goto help

REM Handle "clean" argument
set CLEAN=0
set TARGET=%1

:parse_args
if "%1"=="" goto after_args
if /I "%1"=="clean" (
	set CLEAN=1
) else (
	set TARGET=%1
)
shift
goto parse_args

:after_args
if %CLEAN%==1 (
	echo.Removing build directory "%BUILDDIR%"...
	if exist "%BUILDDIR%" (
		rmdir /s /q "%BUILDDIR%"
		echo.Build directory removed.
	) else (
		echo.No build directory found.
	)

    echo.Removing "api\auto_generated" directory...
	if exist "api\auto_generated" (
		rmdir /s /q "api\auto_generated"
		echo."api\auto_generated" directory removed.
	) else (
		echo.No "api\auto_generated" directory found.
	)
)

%SPHINXBUILD% -M %TARGET% %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
echo.
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
echo.
echo.Add the optional argument "clean" to remove "%BUILDDIR%" and "api\auto_generated" before running `target'
echo.For example: `make target clean'.
echo.

:end
popd
