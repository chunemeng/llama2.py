@echo off
REM ===============================================
REM Triton kernel ncu 采样 - Windows 批处理
REM ===============================================

REM --- 配置部分 ---
SET NCUBAT="C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.0\ncu.bat"
SET PYTHON="C:\Users\aa\AppData\Local\Programs\Python\Python311\python.exe"
SET SCRIPT=./run.py ./stories110M.bin -t 0 -p 0.9 -n 256 -i "One day, Lily met a Shoggoth"
SET OUTPUT="profile_gemm1"

REM --- 调用 ncu 采样 ---
REM --set full: 收集完整指标
REM --target-processes all: 包含子进程（PyTorch/Triton）
call %NCUBAT% --set full --target-processes all -o %OUTPUT% -f %PYTHON% %SCRIPT%

REM --- 输出完成提示 ---
echo.
echo ===============================================
echo Nsight Compute profile done, output file: %OUTPUT%.ncu-rep
echo ===============================================