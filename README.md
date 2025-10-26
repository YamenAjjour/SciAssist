# SciAssist

SciAssist a question answering system for NLP Papers that is implemented using Retrieval Augmented Generation. It can be deployed using 
Gemini API or locally using an open source model such as a backbone to answer questions on scientific NLP Papers. The paper pool that the underlines 
the rag system contains 72k papers from the [ACL Anthology](https://github.com/shauryr/ACL-anthology-corpus?tab=readme-ov-file). You can also supply your own Papers by adding a folder that contains the papers to the configuration under `path_dataset_own_domain`and setting `own_domain` to `True`. 
The system is based on a central [configuration](scirag/config.yaml) where parameters such as the size of the paper chunks and the size of retrieved paper chunks are stored.

![img_1.png](img_1.png)


### RESTFUL API Server

```
./run_server.sh
```

Then you can access the server as follows 

```
http://127.0.0.1:8585/chat?q=What are typical design patterns of RAG?
```

The server returns a dictionary as follows
```
{
asnwer: "The answer
}
```

### Streamlit Question Answering Client 

The chat ui client is implemented using streamlit and is deployed as a single docker container

to build run the following



```
./run_client.sh
```
Then you should be able to access the web client using 127.0.0.1:8501

### Locally using CLI
To deploy the Retrieval Augmented Generation pipeline via clli which allows you to chat with the interface via command line you
can run the following

```
python scirag/setup_rag.py --path-index data/index --path-dataset data/acl-publication-info.74k.parquet
  --path-model /bigwork/nhwpajjy/pre-trained-models/DeepSeek-R1-Distill-Qwen-1.5B
```



# Docker 

### Server 

### Locally as a webservice using docker
```
./scirag/deploy_sciarg_cpu.sh
```

The web service can be then accessed as follows
```
http://127.0.0.1:80/chat?q= What are typical software designs of RAG   
```



### Client 

```
docker build -t sciassist-client -f chatui/docker/Dockerfile .
```

To run the client use
```
docker run  -p 8081:8051 sciassist-client
```


# AWS
```
1) you can start an EC2 instance g4dn.xlarge.
2) docker run -it --rm --gpus all -p 80:80 --name "sciassist-cnt" --tty "yamenajjour:/sciassist-img:17"

```
