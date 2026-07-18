@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0generate-cv.ps1" %*
