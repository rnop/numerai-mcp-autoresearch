@echo off
cd /d "C:\Users\nopro\OneDrive\Desktop\numerai-mcp-autoresearch"
"C:\Users\nopro\anaconda3\envs\numerai_rag_env\python.exe" autoresearch-src/train.py --walkforward --dynamic-features --dynamic-target --top-k 120 --trailing-eras 142 > run_exp1.log 2>&1
echo TASK_EXIT=%ERRORLEVEL%>> run_exp1.log
