# d2s-demo-backend

docker run -d --init -p 8070:8070 -p 8071:8071 --name grobid lfoppiano/grobid:0.6.1

uvicorn app:app --reload --port=5000
