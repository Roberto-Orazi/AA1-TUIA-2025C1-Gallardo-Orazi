Before creating the enviroment remember to use python 3.10 because tensorflow is not compatible with python 3.13

```bash
python3.10 -m venv aa1

source aa1/bin/activate

pip install -r requirements.txt
```


For running Docker

First we are gonna build, for that we have to be in the folder named docker.

So if we are in the principal folder we do:
```bash
cd docker
```
And then we do the build:
```bash
docker build -t predictor-lluvia .
```

After we successfully build the image we are gonna do the run with some error handling for the output file.

If you dont have the folder call output on docker we can create it doing:
```bash
mkdir output
```

After that we can run:
```bash
docker run --rm -v $(pwd)/output:/app/output predictor-lluvia sh -c "python inferencia.py; cp prediccion_resultado.json /app/output/ 2>/dev/null || echo 'Archivo no encontrado'"
```
