export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/run/current-system/sw/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/run/current-system/sw/lib/gcc/x86_64-linux-gnu/13

uvicorn src.__main__:app --reload

