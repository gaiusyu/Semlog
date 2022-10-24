FROM python:3.7
 
WORKDIR /work
 
ADD . .
 
RUN pip install -r requirements.txt

CMD ["python","/work/benchmark/Semlog_benchmark.py"]
 
